# Muon: MomentUm Orthogonalized by Newton-Schulz (Jordan et al. 2024,
# https://kellerjordan.github.io/posts/muon/). Only defined for 2D matrix
# params (the model's 'muon' group); embeddings, head and vectors keep
# AdamW — MuonAdamW below composes the two behind the single-optimizer
# contract from this package's __init__.
#
# Speed: the momentum update runs through foreach kernels, and Newton-
# Schulz runs once per (shape, dtype, device) bucket on a stacked batch —
# a transformer's many identically-shaped block matrices share one
# batched matmul chain instead of launching a small chain each.

import functools

import torch
import torch.distributed as dist
from torch import Tensor
from torch.optim.optimizer import Optimizer, ParamsT


def newton_schulz(G: Tensor, steps: int = 5, eps: float = 1e-7) -> Tensor:
    '''Orthogonalize G (..., m, n) by quintic Newton-Schulz in bf16.

    Returns an approximation of U V^T from the SVD of G (batched over
    leading dims). The quintic coefficients maximize convergence speed at
    the price of the singular values only landing in ~[0.7, 1.3], which
    is enough for Muon. bf16 throughout: the iteration is stable in low
    precision and matmuls are the whole cost.
    '''
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.to(torch.bfloat16)
    transposed = X.size(-2) > X.size(-1)
    if transposed:                       # keep X @ X.mT the small gram matrix
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)
    for _ in range(steps):
        A = X @ X.mT
        X = a * X + (b * A + c * (A @ A)) @ X
    if transposed:
        X = X.mT
    return X


@functools.cache
def _compiled_newton_schulz():
    # One compiled graph per input shape (few bucket shapes, stable across
    # steps). The matmuls stay cuBLAS; compile fuses the a*X + (...)@X
    # elementwise epilogue and the normalization into fewer kernels.
    return torch.compile(newton_schulz)


def muon_update(
    params: list[Tensor],
    grads: list[Tensor],
    momentum_bufs: list[Tensor],
    *,
    lr: float,
    momentum: float,
    weight_decay: float,
    nesterov: bool,
    ns_steps: int,
    eps: float,
    lr_adjust: str = 'shape',
    compile_ns: bool = False,
    world_size: int = 1,
    rank: int = 0,
) -> None:
    '''Functional Muon over one param group (all tensors 2D).

    With world_size > 1 the Newton-Schulz work is sharded: each rank runs
    NS on its slice of every same-shape stack and the slices are
    all-gathered, so replicas apply byte-identical updates. Requires the
    DDP invariant (identical grads and momentum on every rank, params in
    the same order) and every rank stepping in lockstep.
    '''
    # EMA momentum in lerp form: buf <- momentum*buf + (1-momentum)*grad.
    # Newton-Schulz normalizes scale away, so only the direction of the
    # update matters and the (1-momentum) factor is harmless.
    torch._foreach_lerp_(momentum_bufs, grads, 1 - momentum)  # type: ignore[attr-defined]
    if nesterov:
        updates = torch._foreach_lerp(grads, momentum_bufs, momentum)  # type: ignore[attr-defined]
    else:
        updates = momentum_bufs                # stack below copies; no aliasing

    if weight_decay != 0:                      # decoupled, as in AdamW
        torch._foreach_mul_(params, 1 - lr * weight_decay)  # type: ignore[attr-defined]

    ns = _compiled_newton_schulz() if compile_ns else newton_schulz

    # Bucket order derives from param order, so it is identical on every
    # rank — the sharding below depends on that.
    buckets: dict[tuple, list[int]] = {}
    for i, u in enumerate(updates):
        buckets.setdefault((u.shape, u.dtype, u.device), []).append(i)
    for (shape, dtype, device), idx in buckets.items():
        stack = torch.stack([updates[i] for i in idx])
        n = stack.size(0)
        if world_size > 1:
            chunk = -(-n // world_size)          # ceil; NS of the zero
            local = stack[rank * chunk:(rank + 1) * chunk]  # padding is zero
            if local.size(0) < chunk:
                local = torch.cat(
                    [local, stack.new_zeros((chunk - local.size(0), *shape))])
            local = ns(local, steps=ns_steps, eps=eps)
            out = torch.empty((chunk * world_size, *shape),
                              dtype=local.dtype, device=device)
            dist.all_gather(list(out.chunk(world_size)), local.contiguous())
            O = out[:n]
        else:
            O = ns(stack, steps=ns_steps, eps=eps)

        if lr_adjust == 'shape':
            # Reference impl (Jordan): unit spectral-ish norm, boosted for
            # tall matrices so row-wise RMS survives m > n.
            scale = max(1.0, shape[0] / shape[1]) ** 0.5
        else:  # 'rms' — Moonlight/Kimi: match AdamW's ~0.2 update RMS so
            #   Muon can reuse AdamW's lr and schedule directly.
            scale = 0.2 * max(shape[0], shape[1]) ** 0.5
        torch._foreach_add_(  # type: ignore[attr-defined]
            [params[i] for i in idx],
            list(O.to(dtype).unbind(0)),
            alpha=-lr * scale)


class Muon(Optimizer):
    '''Muon for 2D parameters. Everything of dim != 2 is rejected at
    construction — route those params to AdamW (see MuonAdamW).'''

    def __init__(
        self,
        params: ParamsT,
        lr: float = 0.02,
        momentum: float = 0.95,
        weight_decay: float = 0.0,
        *,
        nesterov: bool = True,
        ns_steps: int = 5,
        eps: float = 1e-7,
        lr_adjust: str = 'shape',
        compile_ns: bool | None = None,
        shard: bool = True,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f'Invalid learning rate: {lr}')
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f'Invalid momentum: {momentum}')
        if weight_decay < 0.0:
            raise ValueError(f'Invalid weight_decay: {weight_decay}')
        if ns_steps < 1:
            raise ValueError(f'Invalid ns_steps: {ns_steps}')
        if lr_adjust not in ('shape', 'rms'):
            raise ValueError(f'Invalid lr_adjust: {lr_adjust!r} '
                             f"(expected 'shape' or 'rms')")
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay,
                        nesterov=nesterov, ns_steps=ns_steps, eps=eps,
                        lr_adjust=lr_adjust)
        super().__init__(params, defaults)
        # Machine-local execution knobs, deliberately kept out of defaults
        # (and thus out of state_dict): a checkpoint must not carry the
        # writer's compile/shard setup onto the reader.
        self._compile_ns = (torch.cuda.is_available()
                            if compile_ns is None else compile_ns)
        self._shard = shard
        for group in self.param_groups:
            for p in group['params']:
                if p.dim() != 2:
                    raise ValueError(
                        f'Muon only handles 2D params, got shape '
                        f'{tuple(p.shape)}; give this param to AdamW instead')

    def __setstate__(self, state) -> None:
        super().__setstate__(state)
        for group in self.param_groups:   # fill keys absent in old checkpoints
            group.setdefault('nesterov', True)
            group.setdefault('ns_steps', 5)
            group.setdefault('eps', 1e-7)
            group.setdefault('lr_adjust', 'shape')

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if self._shard and dist.is_available() and dist.is_initialized():
            world_size, rank = dist.get_world_size(), dist.get_rank()
        else:
            world_size, rank = 1, 0

        for group in self.param_groups:
            params: list[Tensor] = []
            grads: list[Tensor] = []
            bufs: list[Tensor] = []
            for p in group['params']:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError('Muon does not support sparse gradients')
                state = self.state[p]
                if not state:              # lazy init, like Adam
                    state['momentum_buffer'] = torch.zeros_like(p)
                params.append(p)
                grads.append(p.grad)
                bufs.append(state['momentum_buffer'])
            if params:
                muon_update(
                    params, grads, bufs,
                    lr=group['lr'], momentum=group['momentum'],
                    weight_decay=group['weight_decay'],
                    nesterov=group['nesterov'], ns_steps=group['ns_steps'],
                    eps=group['eps'], lr_adjust=group['lr_adjust'],
                    compile_ns=self._compile_ns,
                    world_size=world_size, rank=rank)
        return loss


class MuonAdamW:
    '''Muon on the block matrices + AdamW on the rest, presented as one
    optimizer: step / zero_grad / state_dict / load_state_dict /
    param_groups, with every member group surfaced (the lr schedule and
    metrics iterate param_groups directly).'''

    def __init__(self, muon: Muon, adamw: torch.optim.AdamW) -> None:
        self.muon = muon
        self.adamw = adamw

    @property
    def param_groups(self):
        return self.muon.param_groups + self.adamw.param_groups

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self.muon.step()
        self.adamw.step()
        return loss

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.muon.zero_grad(set_to_none=set_to_none)
        self.adamw.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> dict:
        return {'muon': self.muon.state_dict(),
                'adamw': self.adamw.state_dict()}

    def load_state_dict(self, state_dict: dict) -> None:
        self.muon.load_state_dict(state_dict['muon'])
        self.adamw.load_state_dict(state_dict['adamw'])
