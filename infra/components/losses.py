# Loss computation that fuses the lm_head projection into the loss.
#
# The standard path materialises the full (N, vocab) logits twice — once in
# bf16 (saved for backward) and once in fp32 inside cross_entropy's autocast
# upcast. At 32k vocab that is the largest single allocation of a training
# step (the micro-16 OOM of job 29680266). Here rows are processed in
# chunks under torch.utils.checkpoint: forward keeps only the per-chunk
# loss, backward recomputes each chunk's logits, so peak memory holds one
# chunk of logits instead of all of them. Cost: the head matmul runs once
# more in backward (a few percent of model FLOPs at 340M).

import inspect

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

# CUDA/Triton-only, installed manually on GPU boxes (like triton/flash_attn)
try:
    from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
except ImportError:
    LigerFusedLinearCrossEntropyLoss = None


def _chunk_loss_sums(h, weight, targets, z_coef):
    # bf16 matmul under autocast; the fp32 upcast is chunk-sized
    logits = F.linear(h, weight).float()
    ce = F.cross_entropy(logits, targets, reduction='sum')
    z = (z_coef * torch.logsumexp(logits, dim=-1).pow(2).sum()
         if z_coef else logits.new_zeros(()))
    return ce, z


def chunked_cross_entropy(
        hidden: torch.Tensor,     # (..., d) pre-head hidden states
        weight: torch.Tensor,     # (vocab, d) lm_head weight
        targets: torch.Tensor,    # (...) int64
        *,
        chunk_rows: int = 4096,
        z_loss: float = 0.0,
        return_z: bool = False,
    ):
    '''Mean cross-entropy (plus optional z-loss term) computed from hidden
    states without ever materialising the full logits matrix. Numerically
    equivalent to F.cross_entropy(F.linear(hidden, weight), targets) with
    the same fp32 softmax path; per-chunk summation may differ by float
    rounding only. With return_z, returns (loss, z_term) where loss still
    includes the z term and z_term is its mean value alone.'''
    h = hidden.reshape(-1, hidden.shape[-1])
    t = targets.reshape(-1)
    n = h.shape[0]
    ce_total = torch.zeros((), device=h.device, dtype=torch.float32)
    z_total = torch.zeros((), device=h.device, dtype=torch.float32)
    for i in range(0, n, chunk_rows):
        ce, z = checkpoint(
            _chunk_loss_sums, h[i:i + chunk_rows], weight, t[i:i + chunk_rows],
            z_loss, use_reentrant=False)
        ce_total = ce_total + ce
        z_total = z_total + z
    loss = (ce_total + z_total) / n
    return (loss, z_total / n) if return_z else loss


class CELoss:
    '''Callable loss path with z-term tracking. __call__(hidden,
    head_weight, targets) returns the mean combined objective (CE + z
    term); when tracks_z, last_z holds the detached z term of the LAST
    call, so pure CE is loss - last_z. Cross-run loss comparisons must use
    the CE value: the z term's magnitude (coef * lse^2, ~1e-3..1e-2 nats
    at 1e-4) rivals the effects being measured.'''

    def __init__(self, fn, tracks_z: bool):
        self._fn = fn                # fn(...) -> (loss, detached z | None)
        self.tracks_z = tracks_z
        self.last_z: torch.Tensor | None = None

    def __call__(self, hidden, weight, targets) -> torch.Tensor:
        loss, z = self._fn(hidden, weight, targets)
        if z is not None:
            self.last_z = z
        return loss


def make_ce(impl: str, *, z_loss: float = 0.0, chunk_rows: int = 4096,
            compute_dtype: torch.dtype | None = torch.bfloat16) -> CELoss:
    '''Factory for the loss path (a CELoss; the reported loss is always
    the combined objective, see CELoss for the z-term split contract).
    'liger' needs return_z_loss support (>= 0.5.2) to split; older
    kernels train identically but report the combined loss only.

    'full' materialises the logits (reference path, fastest, largest
    peak); 'chunked' is the dependency-free fused fallback; 'liger'
    (CUDA/Triton) trades ~1.3x loss-path time for ~96% of its peak memory.
    The three are numerically interchangeable at bf16-rounding level —
    measured in meter/examples/bench_ce.py.'''
    if impl == 'full':
        def full_fn(hidden, weight, targets):
            # identical to head(hidden) + F.cross_entropy: autocast runs
            # the matmul in bf16 and upcasts the CE internals to fp32
            logits = F.linear(hidden, weight)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                   targets.reshape(-1))
            if not z_loss:
                return loss, None
            z = z_loss * torch.logsumexp(logits, dim=-1).pow(2).mean()
            return loss + z, z.detach()
        return CELoss(full_fn, tracks_z=bool(z_loss))
    if impl == 'chunked':
        def chunked_fn(hidden, weight, targets):
            loss, z = chunked_cross_entropy(
                hidden, weight, targets, chunk_rows=chunk_rows,
                z_loss=z_loss, return_z=True)
            return loss, z.detach() if z_loss else None
        return CELoss(chunked_fn, tracks_z=bool(z_loss))
    if impl == 'liger':
        if LigerFusedLinearCrossEntropyLoss is None:
            raise ImportError("ce_impl 'liger' requires liger-kernel "
                              "(CUDA-only; pip install liger-kernel)")
        split_z = bool(z_loss) and 'return_z_loss' in inspect.signature(
            LigerFusedLinearCrossEntropyLoss.__init__).parameters
        flce = LigerFusedLinearCrossEntropyLoss(
            lse_square_scale=z_loss,
            **({'return_z_loss': True} if split_z else {}))
        def liger_fn(hidden, weight, targets):
            # liger ignores autocast; cast explicitly (grads flow back to
            # the fp32 masters through the casts)
            dt = compute_dtype or hidden.dtype
            out = flce(weight.to(dt), hidden.reshape(-1, hidden.shape[-1]).to(dt),
                       targets.reshape(-1))
            if isinstance(out, torch.Tensor):
                return out, None
            # (loss, z) tuple on 0.5.x, CrossEntropyOutput on newer liger;
            # the kernel's loss already includes the z term in both
            loss, z = (out if isinstance(out, tuple)
                       else (out.loss, out.z_loss))
            return loss, z.detach()
        return CELoss(liger_fn, tracks_z=split_z)
    raise KeyError(f"unknown fused ce impl '{impl}'")
