# Scoring a continuation with or without a context in memory, on the
# Decodable protocol through the Generator (as every eval task does):
# the context is prefilled without logits, then the continuation block
# is fed with logits, so the logits materialised are only those of Y.

import torch

from ..inference import Generator


@torch.no_grad()
def score_continuation(gen: Generator, y: torch.Tensor, ctx: torch.Tensor | None = None) -> torch.Tensor:
    '''Logits (B, Ly, V) predicting y (B, Ly) on fresh streams that first
    read ctx (B, Lx), or nothing. Bit-identical to score_ids(cat(ctx, y))
    sliced to the last Ly positions. The streams are left holding
    ctx + y with y[:, -1] pending, as prefill_ids would.'''
    assert gen.start_id is not None, 'needs a start token (the fresh stream is never empty)'
    assert y.dim() == 2 and y.shape[1] >= 1
    B, Ly = y.shape
    Lx = 0 if ctx is None else int(ctx.shape[1])
    y = y.to(gen.device, dtype=torch.int64)
    gen.reset(B, max_len=1 + Lx + Ly)
    if Lx > 0:
        assert ctx is not None
        gen.prefill_ids(ctx)
    assert gen.pending is not None
    block = torch.cat([gen.pending[:, None], y[:, :-1]], dim=1)
    out = gen.model.decode_step(block, return_logits=True)
    gen.fed_len += block.shape[1]
    gen.pending = y[:, -1].clone()
    assert out is not None
    return out


def nll_positions(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    '''(B, Ly) fp32 -log p(y_t) from logits (B, Ly, V).'''
    logp = logits.float().log_softmax(-1)
    return -logp.gather(-1, y.to(logp.device)[..., None]).squeeze(-1)


def bucket_means(nll: torch.Tensor, edges: tuple[int, ...]) -> dict[str, torch.Tensor]:
    '''Per-row means over position buckets [0, e0), [e0, e1), ..., [ek, Ly):
    the benefit of a context is front-loaded in Y, and the buckets show
    how fast it fades. Keys like `b0_64`, `b64_256`, `b256_512`.'''
    Ly = nll.shape[1]
    bounds = [0, *[e for e in edges if e < Ly], Ly]
    out = {}
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        out[f'b{lo}_{hi}'] = nll[:, lo:hi].mean(1)
    return out
