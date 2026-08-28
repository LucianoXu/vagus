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

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def _chunk_loss_sum(h, weight, targets, z_coef):
    # bf16 matmul under autocast; the fp32 upcast is chunk-sized
    logits = F.linear(h, weight).float()
    loss = F.cross_entropy(logits, targets, reduction='sum')
    if z_coef:
        loss = loss + z_coef * torch.logsumexp(logits, dim=-1).pow(2).sum()
    return loss


def chunked_cross_entropy(
        hidden: torch.Tensor,     # (..., d) pre-head hidden states
        weight: torch.Tensor,     # (vocab, d) lm_head weight
        targets: torch.Tensor,    # (...) int64
        *,
        chunk_rows: int = 4096,
        z_loss: float = 0.0,
    ) -> torch.Tensor:
    '''Mean cross-entropy (plus optional z-loss term) computed from hidden
    states without ever materialising the full logits matrix. Numerically
    equivalent to F.cross_entropy(F.linear(hidden, weight), targets) with
    the same fp32 softmax path; per-chunk summation may differ by float
    rounding only.'''
    h = hidden.reshape(-1, hidden.shape[-1])
    t = targets.reshape(-1)
    n = h.shape[0]
    total = torch.zeros((), device=h.device, dtype=torch.float32)
    for i in range(0, n, chunk_rows):
        part = checkpoint(
            _chunk_loss_sum, h[i:i + chunk_rows], weight, t[i:i + chunk_rows],
            z_loss, use_reentrant=False)
        assert part is not None
        total = total + part
    return total / n
