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
#
# Two distributed regimes, chosen per parameter by its type:
#   plain tensors (DDP: every rank holds the full param and grad) —
#     muon_update shards the Newton-Schulz WORK across the world and
#     all-gathers the results, so replicas apply byte-identical updates.
#   DTensors (FSDP2 / HSDP: the param and its grad are row-sharded over
#     the mesh's shard dim, replicated over any other dim) —
#     muon_update_sharded keeps momentum and decay on the local shard,
#     all-gathers the bf16 momentum of each same-shape bucket over the
#     shard group, splits Newton-Schulz across that group as above, and
#     each rank applies its own row slice of the orthogonalised update
#     (Moonlight's "distributed Muon", arXiv:2502.16982). The traffic is
#     2x the matrix bytes per step on the shard group's links, which
#     HSDP places inside the node.

import functools

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed.tensor import DTensor, Shard
from torch.optim.optimizer import Optimizer, ParamsT


# Coefficient schemes for the quintic iteration. 'jordan' is the constant
# tuple from the reference impl (slope at 0 maximized; the bulk of the
# spectrum lands in a ~[0.68, 1.2] band and never tightens further).
# 'polar_express' is the minimax-optimal 5-step schedule (Amsel et al.,
# arXiv:2505.16932, values from modded-nanogpt): ~2x closer to UV^T at
# identical cost; its schedule length fixes the step count, and it wants
# the 2% norm slack it was optimized under.
_NS_SCHEMES: dict[str, tuple[list[tuple[float, float, float]], float]] = {
    'jordan': ([(3.4445, -4.7750, 2.0315)], 0.0),  # repeated `steps` times
    'polar_express': ([
        (8.156554524902461, -22.48329292557795, 15.878769915207462),
        (4.042929935166739, -2.808917465908714, 0.5000178451051316),
        (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
        (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
        (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
    ], 2e-2),
}


def newton_schulz(G: Tensor, steps: int = 5, eps: float = 1e-7,
                  scheme: str = 'jordan') -> Tensor:
    '''Orthogonalize G (..., m, n) by quintic Newton-Schulz in bf16.

    Returns an approximation of U V^T from the SVD of G (batched over
    leading dims). Inexact by design: the coefficients favor slope at 0
    (inflating small singular values fast) over exact convergence to 1,
    which is enough for Muon. bf16 throughout: the iteration is stable in
    low precision and matmuls are the whole cost.
    '''
    coeffs, slack = _NS_SCHEMES[scheme]
    if len(coeffs) == 1:
        coeffs = coeffs * steps          # constant scheme: `steps` applies
    X = G.to(torch.bfloat16)
    transposed = X.size(-2) > X.size(-1)
    if transposed:                       # keep X @ X.mT the small gram matrix
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * (1 + slack) + eps)
    for a, b, c in coeffs:
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
    ns_coeffs: str = 'jordan',
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
            local = ns(local, steps=ns_steps, eps=eps, scheme=ns_coeffs)
            out = torch.empty((chunk * world_size, *shape),
                              dtype=local.dtype, device=device)
            dist.all_gather(list(out.chunk(world_size)), local.contiguous())
            O = out[:n]
        else:
            O = ns(stack, steps=ns_steps, eps=eps, scheme=ns_coeffs)

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


def _shard_layout(p: DTensor):
    '''(group, world_size, rank) of the mesh dim a DTensor is sharded
    over; (None, 1, 0) when it is replicated everywhere. Only row
    sharding (Shard(0), FSDP2's layout) on a single mesh dim is
    supported.'''
    dims = [i for i, pl in enumerate(p.placements) if isinstance(pl, Shard)]
    if not dims:
        return None, 1, 0
    assert len(dims) == 1 and p.placements[dims[0]].dim == 0, \
        f'Muon expects row sharding on one mesh dim, got {p.placements}'
    mesh = p.device_mesh
    return mesh.get_group(dims[0]), mesh.size(dims[0]), mesh.get_local_rank(dims[0])


def _lr_scale(shape, lr_adjust: str) -> float:
    if lr_adjust == 'shape':
        # Reference impl (Jordan): unit spectral-ish norm, boosted for
        # tall matrices so row-wise RMS survives m > n.
        return max(1.0, shape[0] / shape[1]) ** 0.5
    # 'rms' — Moonlight/Kimi: match AdamW's ~0.2 update RMS so Muon can
    # reuse AdamW's lr and schedule directly.
    return 0.2 * max(shape[0], shape[1]) ** 0.5


def muon_update_sharded(
    params: list[DTensor],
    grads: list[DTensor],
    momentum_bufs: list[DTensor],
    *,
    lr: float,
    momentum: float,
    weight_decay: float,
    nesterov: bool,
    ns_steps: int,
    eps: float,
    lr_adjust: str = 'shape',
    ns_coeffs: str = 'jordan',
    compile_ns: bool = False,
) -> None:
    '''muon_update for row-sharded DTensor params (FSDP2). Elementwise
    work (momentum, decay, the final add) runs on the local shards;
    Newton-Schulz needs whole matrices, so per bucket the bf16 momentum
    shards are all-gathered over the shard group, NS is split across
    that group and its outputs all-gathered back. Every rank of the
    shard group must call this with the same params in the same order
    (the DDP invariant, now per shard group); the replicate groups do
    identical work on identical inputs.

    Requires each matrix's row count to be a multiple of the shard
    group size (FSDP2 pads uneven shards; the stacked gather here needs
    equal local shapes) — true for every block matrix at our widths,
    and any skinny exception belongs to AdamW anyway.'''
    P = [p.to_local() for p in params]
    G = [g.to_local() for g in grads]
    Bf = [b.to_local() for b in momentum_bufs]
    for p, g, b in zip(params, grads, momentum_bufs):
        assert g.placements == p.placements and b.placements == p.placements, \
            f'grad/momentum layout {g.placements}/{b.placements} differs from param {p.placements}'

    torch._foreach_lerp_(Bf, G, 1 - momentum)  # type: ignore[attr-defined]
    if nesterov:
        updates = torch._foreach_lerp(G, Bf, momentum)  # type: ignore[attr-defined]
    else:
        updates = Bf
    if weight_decay != 0:
        torch._foreach_mul_(P, 1 - lr * weight_decay)  # type: ignore[attr-defined]

    ns = _compiled_newton_schulz() if compile_ns else newton_schulz

    buckets: dict[tuple, list[int]] = {}
    for i, p in enumerate(params):
        buckets.setdefault((tuple(p.shape), p.dtype, p.device, id(p.device_mesh), p.placements), []).append(i)
    for (shape, dtype, device, _, _), idx in buckets.items():
        group, ws, rank = _shard_layout(params[idx[0]])
        rows, cols = shape
        n = len(idx)
        # NS runs in bf16; gathering bf16 halves the traffic at no cost
        local = torch.stack([updates[i] for i in idx]).to(torch.bfloat16)   # (n, rows_l, cols)
        if ws > 1:
            rows_l = rows // ws
            assert rows % ws == 0 and local.shape[1] == rows_l, \
                f'{shape} is not evenly row-sharded {ws} ways (local {tuple(local.shape[1:])})'
            gathered = torch.empty((ws * n, rows_l, cols), dtype=torch.bfloat16, device=device)
            dist.all_gather_into_tensor(gathered, local.contiguous(), group=group)
            full = gathered.view(ws, n, rows_l, cols).transpose(0, 1).reshape(n, rows, cols)
        else:
            rows_l = rows
            full = local

        chunk = -(-n // ws)                     # ceil; NS of the zero padding is zero
        mine = full[rank * chunk:(rank + 1) * chunk]
        if mine.size(0) < chunk:
            mine = torch.cat([mine, full.new_zeros((chunk - mine.size(0), rows, cols))])
        mine = ns(mine, steps=ns_steps, eps=eps, scheme=ns_coeffs)
        if ws > 1:
            out = torch.empty((chunk * ws, rows, cols), dtype=mine.dtype, device=device)
            dist.all_gather_into_tensor(out, mine.contiguous(), group=group)
            O = out[:n, rank * rows_l:(rank + 1) * rows_l]
        else:
            O = mine[:n]

        torch._foreach_add_(  # type: ignore[attr-defined]
            [P[i] for i in idx],
            list(O.to(dtype).unbind(0)),
            alpha=-lr * _lr_scale(shape, lr_adjust))


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
        ns_coeffs: str = 'jordan',
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
        if ns_coeffs not in _NS_SCHEMES:
            raise ValueError(f'Invalid ns_coeffs: {ns_coeffs!r} '
                             f'(expected one of {sorted(_NS_SCHEMES)})')
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay,
                        nesterov=nesterov, ns_steps=ns_steps, eps=eps,
                        lr_adjust=lr_adjust, ns_coeffs=ns_coeffs)
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
            group.setdefault('ns_coeffs', 'jordan')

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
            sharded: list[DTensor] = []
            sgrads: list[DTensor] = []
            sbufs: list[DTensor] = []
            for p in group['params']:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError('Muon does not support sparse gradients')
                state = self.state[p]
                if not state:              # lazy init, like Adam (a DTensor
                    state['momentum_buffer'] = torch.zeros_like(p)   # param gets a DTensor buffer)
                if isinstance(p, DTensor):
                    sharded.append(p); sgrads.append(p.grad); sbufs.append(state['momentum_buffer'])
                else:
                    params.append(p); grads.append(p.grad); bufs.append(state['momentum_buffer'])
            common = dict(
                lr=group['lr'], momentum=group['momentum'],
                weight_decay=group['weight_decay'],
                nesterov=group['nesterov'], ns_steps=group['ns_steps'],
                eps=group['eps'], lr_adjust=group['lr_adjust'],
                ns_coeffs=group['ns_coeffs'], compile_ns=self._compile_ns)
            if params:
                muon_update(params, grads, bufs, **common,
                            world_size=world_size, rank=rank)
            if sharded:
                muon_update_sharded(sharded, sgrads, sbufs, **common)
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

    @property
    def members(self) -> list[torch.optim.Optimizer]:
        '''The torch optimizers inside, for APIs that want real
        Optimizer objects (the DCP state-dict functions).'''
        return [self.muon, self.adamw]

    @property
    def state(self):
        '''Merged per-param state view (fresh dict each access, so writes
        to the mapping itself are lost — mutate member .state instead).
        Lets state-inspecting metric hooks treat the composite like a
        single optimizer.'''
        return {**self.muon.state, **self.adamw.state}

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
