# How much of a stream a recurrent memory can still absorb, as a function
# of position — the capacity reading.
#
# The delta rule writes beta_t k_t (v_t - S^T k_t)^T, so the residual
# ratio |v_t - S^T k_t| / |v_t| is what the memory failed to predict about
# the value it is about to store (GDNLM.surprise). Run along a stream from
# a blank state, averaged over sequences, it falls while the state still
# has room and flattens once it does not. Where it flattens is the
# memory's practical horizon, and that is a measurement, not a
# derivation: the rank bound says a d_k x d_v state separates at most d_k
# pairs, but the learned decay can evict long before that bound bites.
#
# Two families of numbers come back:
#   frac@p     the share of the curve's TOTAL fall completed by position
#              p. Normalised per subject, so it survives the d_v
#              confound below and is the number to compare across
#              geometries.
#   plateau    the level the curve settles at (mean over the tail).
#              NOT comparable across different d_v: the ratio is
#              normalised by |v_t|, so a wider value head has more norm
#              to predict per slot and sits higher for reasons that have
#              nothing to do with capacity. Compare levels only at equal
#              d_v.
#
# Measured on the LAX1 / LAX3 pair (2026-09-12, 32 sequences x 2048):
# LAX1 (d_k 128) completes 97.5% of its fall by position 128, LAX3
# (d_k 256, same params) 95.9% — doubling d_k did not move the point.
# The reason is in the decay, not the rank: LAX3's median realised
# channel timescale is 4.1 tokens against a 256-slot bound, so 98.5% of
# its channels forget before they could ever fill it.
#
# Items are per (layer, bucket) means in a fixed order, so the runner
# pairs subjects on them; the per-layer curve is in the witness, since
# depth is where the timescale hierarchy lives.

from typing import Any, Sequence

import numpy as np
import torch

from ..core import EvalCtx, TaskResult

DEFAULT_EDGES = (1, 2, 4, 8, 16, 32, 64, 128, 192, 256, 384, 512, 1024)
DEFAULT_MARKS = (64, 128, 192, 256, 384, 512)


def _windows(store, rng, n: int, length: int) -> np.ndarray:
    '''n contiguous windows of `length` tokens, shard-weighted by token
    count. Contiguous rather than document-aligned on purpose: this is
    the distribution the model was trained on.'''
    weights = np.asarray(store.shard_tokens, dtype=np.float64)
    weights /= weights.sum()
    out = []
    for _ in range(n):
        s = int(rng.choice(len(weights), p=weights))
        hi = int(store.shard_tokens[s]) - length - 1
        out.append(store.read_window(s, int(rng.integers(0, hi)), length))
    return np.stack(out).astype(np.int64)


def run(ctx: EvalCtx, data_dir: str, shards: list[str] | None = None, n_seqs: int = 32,
        length: int = 2048, batch_size: int = 8, edges: Sequence[int] = DEFAULT_EDGES,
        marks: Sequence[int] = DEFAULT_MARKS, tail_from: int = 1024) -> TaskResult:
    model: Any = ctx.subject.model     # GDNLM; Decodable does not declare surprise/config
    assert hasattr(model, 'surprise'), \
        f'{type(model).__name__} has no surprise(); saturation needs an all-GDN linear model'
    store = ctx.store(data_dir, shards)
    ids = _windows(store, ctx.rng('windows'), n_seqs, length)
    assert tail_from < length, f'tail_from {tail_from} must be inside length {length}'

    n_layers = int(model.config['layer_count'])
    acc = np.zeros((n_layers, length))
    seen = 0
    with torch.no_grad():
        for b in range(0, n_seqs, batch_size):
            tok = torch.from_numpy(ids[b:b + batch_size]).to(ctx.device)
            # (B, L, layers, H) -> mean over batch and heads -> (layers, L)
            s = model.surprise(tok).mean(dim=(0, 3)).permute(1, 0).float().cpu().numpy()
            acc += s * tok.shape[0]
            seen += tok.shape[0]
            ctx.log(f'saturation: {seen}/{n_seqs} sequences')
    curve = acc / seen                                   # (layers, length)
    mean = curve.mean(0)

    lo_hi = [(lo, hi) for lo, hi in zip(edges[:-1], edges[1:]) if hi <= length]
    lo_hi.append((edges[-1], length))
    buckets = {f'b{lo}_{hi}': float(mean[lo:hi].mean()) for lo, hi in lo_hi}

    start, plateau = float(mean[0]), float(mean[tail_from:].mean())
    total = start - plateau
    # A curve that does not fall has no fraction-of-fall to report, and an
    # untrained model's does not (the state only adds noise, so the residual
    # rises). Emit the frac@ keys only when they mean something rather than
    # putting NaN — which is not valid JSON — into a versioned record;
    # total_fall is always there and says why they are missing.
    fracs = {}
    if total > 0:
        for p in marks:
            if p < length:
                near = mean[max(1, p - 8):p + 8].mean()
                fracs[f'frac@{p}'] = float((start - near) / total)

    scalars = {'start': start, 'plateau': plateau, 'total_fall': total,
               'fall_to_128': float(start - mean[120:136].mean()),
               'fall_after_128': float(mean[120:136].mean() - plateau), **fracs, **buckets}
    args = model.config
    return TaskResult(
        scalars=scalars,
        # per (layer, bucket), layer-major — the runner pairs subjects on this order
        items={'bucket_mean': [float(curve[l, lo:hi].mean())
                               for l in range(n_layers) for lo, hi in lo_hi]},
        witness={
            'n_seqs': seen, 'length': length, 'tail_from': tail_from,
            'edges': list(edges), 'marks': list(marks),
            'buckets': [f'{lo}-{hi - 1}' for lo, hi in lo_hi],
            'geometry': {k: args[k] for k in ('head_count', 'key_head_dim', 'value_head_dim')},
            'rank_bound_d_k': int(args['key_head_dim']),
            'per_layer_plateau': [float(curve[l, tail_from:].mean()) for l in range(n_layers)],
            'level_comparable_only_at_equal_d_v': int(args['value_head_dim']),
        },
    )
