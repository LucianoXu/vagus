# Where does a delta-rule memory stop learning about the stream?
#
# The state is a d_k x d_v matrix whose TOKEN capacity is d_k: retrieval
# is S^T q = sum_i v_i (k_i . q), so pairs stay separable only while the
# keys are independent, and the keys live in d_k dimensions. The
# observable consequence is the per-token residual
#
#     rho_t = |v_t - S_{t-1}^T k_t| / |v_t|
#
# — what the memory failed to predict about the value it is about to
# store. From a blank state rho falls as the memory fills; once the
# store is saturated it stops falling, because every further write goes
# into a state that is already full. So the POSITION at which rho
# flattens is a direct, structural read of the memory's capacity, and
# unlike a recall task it does not need a model strong enough to solve
# anything: a half-trained model saturates at the same place.
#
# Measured on the trained LAX1-340M (d_k 128) on 2026-09-10: rho falls
# 0.92 -> 0.715 over positions 1-128 and is then flat to 2048 (0.698).
# 92% of everything that memory ever learns about a stream it has
# learned by position 128 = d_k. LAX3 doubles d_k to 256, and the
# falsifiable prediction is that its knee moves out toward 256 and its
# plateau sits lower.
#
# Usage (one A100; the token loop is launch-bound, not FLOP-bound):
#   python -m infra.meter.examples.saturation_curve \
#       --ckpt LAX1=runs/.../ckpt-00005000.pt --ckpt LAX3=runs/.../ckpt-00005000.pt \
#       --data data/tokenized/fineweb-edu-100BT-mistral32k --batches 4
#
# The recurrence is written out here rather than imported so the probe
# has no branch dependency: it must match GatedDeltaNet's own scan, and
# the assertion that it does is that both come from the same four lines
# (decay the state, read S^T k, form the residual, write beta k r^T).

import argparse
import json

import numpy as np
import torch

from ...components.linear_attention import GatedDeltaNet
from ...dataset.loader import TokenStore
from ...models.io import load_model

BUCKETS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)


@torch.no_grad()
def residual_by_position(model, tokens: torch.Tensor) -> torch.Tensor:
    '''(layers, L) mean residual ratio, averaged over batch and heads.

    Mirrors GatedDeltaNet.forward: the mixer sees blk.rmsnorm1(x), and
    the state evolves S <- diag(a_t) S + beta_t k_t (v_t - S^T k_t)^T.'''
    x = model.embedding(tokens)
    out = []
    for blk in model.blocks:
        att = blk.att
        assert isinstance(att, GatedDeltaNet), \
            f'{type(att).__name__}: this probe is for all-GDN layouts'
        h = blk.rmsnorm1(x)
        _, k, v = att._heads(att.wq(h), att.wk(h), att.wv(h))
        g, beta = att._gates(h)
        beta, w = att._write(h, k, g, beta)
        B, L, H, dk = k.shape
        dv = v.shape[-1]
        k32, v32 = k.float(), v.float()
        S = torch.zeros(B, H, dk, dv, device=k.device, dtype=torch.float32)
        rho = torch.empty(B, L, H, device=k.device, dtype=torch.float32)
        for t in range(L):
            if g is not None:
                a = g[:, t].float().exp()
                S = S * (a[..., None] if a.dim() == 3 else a[..., None, None])
            kt, vt = k32[:, t], v32[:, t]
            r = vt - torch.einsum('bhk,bhkv->bhv', kt, S)
            rho[:, t] = r.norm(dim=-1) / vt.norm(dim=-1).clamp(min=1e-6)
            wt = w[:, t].float() if w is not None else kt * beta[:, t].float()[..., None]
            S = S + torch.einsum('bhk,bhv->bhkv', wt, r)
        out.append(rho.mean(dim=(0, 2)))
        x = x + blk.att(h)
        x = x + blk.ffn(blk.rmsnorm2(x))
    return torch.stack(out)


def log_bins(L: int) -> list[tuple[int, int]]:
    '''[lo, hi) index ranges, log-spaced, covering position 0 upward.'''
    edges = [0] + [e for e in BUCKETS if e < L] + [L]
    return [(a, b) for a, b in zip(edges[:-1], edges[1:]) if b > a]


def knee(rho: np.ndarray, frac: float) -> tuple[int, int]:
    '''The log-spaced bin in which rho first reaches `frac` of its total
    fall, as 1-indexed positions.

    Binned, not raw: a per-position curve fluctuates by more than the
    tail of the fall it is being compared against, so a raw first-crossing
    fires on the first lucky dip (on 2 sequences of LAX1 it reported 19
    against a knee the table plainly puts near 128). The floor is the mean
    over the last quarter of the stream.'''
    rinf = float(rho[len(rho) * 3 // 4:].mean())
    r0 = float(rho[0])
    target = r0 - frac * (r0 - rinf)
    for lo, hi in log_bins(len(rho)):
        if float(rho[lo:hi].mean()) <= target:
            return lo + 1, hi
    return len(rho), len(rho)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--ckpt', action='append', required=True, metavar='LABEL=PATH')
    ap.add_argument('--data', required=True)
    ap.add_argument('--context', type=int, default=2048)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--batches', type=int, default=4)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--out', default=None, help='write the curves here as json')
    args = ap.parse_args()

    store = TokenStore(args.data)
    L, curves = args.context, {}
    for spec in args.ckpt:
        label, path = spec.split('=', 1)
        model, meta = load_model(path, device=args.device)
        ma = meta['model_args']
        # the same windows for every subject: the curve is a paired read
        rng = np.random.default_rng(args.seed)
        acc = None
        for _ in range(args.batches):
            ids = np.stack([
                store.read_window(int(rng.integers(len(store.tokens))),
                                  int(rng.integers(0, 10_000_000)), L)
                for _ in range(args.batch)]).astype(np.int64)
            r = residual_by_position(model, torch.from_numpy(ids).to(args.device))
            acc = r if acc is None else acc + r
        rho = (acc / args.batches).float().cpu().numpy()          # (layers, L)
        curves[label] = dict(
            per_layer=rho.tolist(), mean=rho.mean(0).tolist(),
            d_k=int(ma['key_head_dim']), d_v=int(ma['value_head_dim']),
            heads=int(ma['head_count']), step=meta.get('step'),
            tokens_seen=meta.get('tokens_seen'))
        del model
        print(f'{label}: d_k={ma["key_head_dim"]} d_v={ma["value_head_dim"]} '
              f'H={ma["head_count"]} step={meta.get("step")}', flush=True)

    print(f'\nresidual |v - S^T k| / |v| by position, {args.batch * args.batches} '
          f'sequences of {L}\n')
    labels = list(curves)
    head = f'{"positions":>14s} ' + ' '.join(f'{x:>9s}' for x in labels)
    print(head + (f' {"delta":>9s}' if len(labels) == 2 else ''))
    print('-' * len(head))
    for lo, hi in log_bins(L):
        vals = [float(np.mean(curves[x]['mean'][lo:hi])) for x in labels]
        row = f'{lo + 1:6d}-{hi:<7d} ' + ' '.join(f'{v:9.4f}' for v in vals)
        if len(vals) == 2:
            row += f' {vals[1] - vals[0]:+9.4f}'
        print(row)

    print(f'\n{"":10s} ' + ' '.join(f'{x:>9s}' for x in labels))
    for name, fn in (('rho(1)', lambda m: m[0]),
                     ('plateau', lambda m: float(np.mean(m[len(m) * 3 // 4:]))),
                     ('total fall', lambda m: m[0] - float(np.mean(m[len(m) * 3 // 4:])))):
        vals = [fn(curves[x]['mean']) for x in labels]
        print(f'{name:10s} ' + ' '.join(f'{v:9.4f}' for v in vals))
    for frac in (0.90, 0.95, 0.99):
        cells = [knee(np.asarray(curves[x]['mean']), frac) for x in labels]
        print(f'knee@{int(frac * 100)}%   ' + ' '.join(f'{a:4d}-{b:<4d}' for a, b in cells))
    print(f'\nd_k: ' + ' '.join(f'{curves[x]["d_k"]:9d}' for x in labels)
          + '   <- the knee should track this if the rank bound is what binds')

    if args.out:
        with open(args.out, 'w') as f:
            json.dump({'context': L, 'sequences': args.batch * args.batches,
                       'curves': curves}, f)
        print(f'\nwrote {args.out}')


if __name__ == '__main__':
    main()
