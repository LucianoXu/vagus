# Where on the readout RMSNorm's curve does a trained linear mixer read?
#
# Every GatedDeltaNet head emits o = RMSNorm(r) * SiLU(W_g x) with the
# read r = scale q^T S. RMSNorm with eps is
#     r / sqrt(mean(r^2) + eps)
# — scale-invariant for mean(r^2) >> eps, linear for mean(r^2) << eps.
# Only in the second regime does a weaker memory read as a weaker signal.
# The consolidation thread found the first regime everywhere on LAX1
# (halving the memory: per-hop KL 0.04; joint modes could not release
# the memory without a cliff), and LAX5 moves the knee on purpose. This
# script measures both sides of that argument on real checkpoints:
#
#   read energy   e = mean(r^2) per (token, head), in units of a
#                 full-strength read of the token's own value,
#                 e_full = scale^2 |v|^2 / dv (= 1/(dk dv) for unit v).
#                 e ~ 1: the query hits one stored association at full
#                 strength; e << 1: a weak or diffuse read.
#   eps (same units)  where the model's own RMSNorm knee sits.
#   linearity     read_linearity at c = 0.5: 0 = halving the read changes
#                 nothing after the norm, 1 = it halves the output.
#   tau(lin=0.5)  the read_floor that would put the MEDIAN read at
#                 linearity 0.5: eps / m = 0.35 there, so tau = 0.35 e_p50.
#                 This is how LAX5's floor is calibrated from LAX1.
#
#   python -m infra.meter.examples.read_regime \
#       --ckpt LAX1=runs/lax1-340M-fwe15B-c9af2f44/model-final.pt \
#       --data data/tokenized/fineweb-edu-100BT-mistral32k --out runs/read-regime.json

import argparse
import json

import numpy as np
import torch

from ...components.linear_attention import GatedDeltaNet, chunk_scan, read_linearity
from ...dataset.loader import TokenStore
from ...models.io import load_model


@torch.no_grad()
def reads(model, tokens: torch.Tensor, warmup: int):
    '''Per layer: (e, lin) each (tokens, heads) with e in full-read units,
    and the layer's eps in the same units (per head).'''
    x = model.embedding(tokens)
    out = []
    for blk in model.blocks:
        att = blk.att
        assert isinstance(att, GatedDeltaNet), \
            f'{type(att).__name__}: this probe is for all-GDN layouts'
        h = blk.rmsnorm1(x)
        q, k, v = att._heads(att.wq(h), att.wk(h), att.wv(h))
        g, beta = att._gates(h)
        beta, w = att._write(h, k, g, beta)
        assert w is None, 'symmetric write only (the torch chunk path)'
        q, k, v = q.float(), k.float(), v.float()
        g = None if g is None else g.float()
        beta = None if beta is None else beta.float()
        o, _ = chunk_scan(q, k, v, g, beta, None, scale=att.scale, delta=att.delta,
                          chunk_size=att.chunk_size)
        m = o.pow(2).mean(-1)[:, warmup:]                                # (B, L', H)
        dv = v.shape[-1]
        e_full = (att.scale ** 2 * v.pow(2).sum(-1) / dv)[:, warmup:]    # (B, L', H)
        unit = e_full.mean(dim=(0, 1))                                   # per head
        eps = att.read_eps()
        eps = eps.to(unit) if torch.is_tensor(eps) else eps
        out.append(dict(
            e=(m / unit).reshape(-1, m.shape[-1]).cpu().numpy(),
            lin=read_linearity(o[:, warmup:], eps).reshape(-1, m.shape[-1]).cpu().numpy(),
            eps_units=(eps / unit).cpu().numpy()))
        x = x + blk.att(h)
        x = x + blk.ffn(blk.rmsnorm2(x))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--ckpt', action='append', required=True, metavar='LABEL=PATH')
    ap.add_argument('--data', required=True)
    ap.add_argument('--context', type=int, default=2048)
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--batches', type=int, default=4)
    ap.add_argument('--warmup', type=int, default=64)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    store = TokenStore(args.data)
    rows = []
    for spec in args.ckpt:
        label, path = spec.split('=', 1)
        model, meta = load_model(path, device=args.device)
        rng = np.random.default_rng(args.seed)          # same windows per subject
        per_layer = None
        for _ in range(args.batches):
            ids = np.stack([
                store.read_window(int(rng.integers(len(store.tokens))),
                                  int(rng.integers(0, 10_000_000)), args.context)
                for _ in range(args.batch)]).astype(np.int64)
            got = reads(model, torch.from_numpy(ids).to(args.device), args.warmup)
            if per_layer is None:
                per_layer = got
            else:
                for acc, new in zip(per_layer, got):
                    acc['e'] = np.concatenate([acc['e'], new['e']])
                    acc['lin'] = np.concatenate([acc['lin'], new['lin']])
        assert per_layer is not None
        layers = []
        for i, d in enumerate(per_layer):
            e, lin = d['e'], d['lin']
            layers.append(dict(
                layer=i, e_p10=float(np.percentile(e, 10)), e_p50=float(np.percentile(e, 50)),
                e_p90=float(np.percentile(e, 90)),
                eps_units=float(np.median(d['eps_units'])),
                lin_mean=float(lin.mean()), lin_p90=float(np.percentile(lin, 90))))
        e_all = np.concatenate([d['e'].ravel() for d in per_layer])
        lin_all = np.concatenate([d['lin'].ravel() for d in per_layer])
        row = dict(label=label, step=meta.get('step'),
                   v_norm=meta['model_args'].get('v_norm', False),
                   read_floor=meta['model_args'].get('read_floor'),
                   e_p10=float(np.percentile(e_all, 10)), e_p50=float(np.percentile(e_all, 50)),
                   e_p90=float(np.percentile(e_all, 90)),
                   lin_mean=float(lin_all.mean()),
                   tau_for_median_lin_half=0.35 * float(np.percentile(e_all, 50)),
                   layers=layers)
        rows.append(row)
        del model

        print(f'\n{label}  step {row["step"]}  v_norm={row["v_norm"]}  read_floor={row["read_floor"]}')
        print(f'{"layer":>5s} {"e p10":>9s} {"e p50":>9s} {"e p90":>9s} {"eps":>9s} {"lin":>6s} {"lin p90":>7s}')
        for L in layers:
            print(f'{L["layer"]:5d} {L["e_p10"]:9.2e} {L["e_p50"]:9.2e} {L["e_p90"]:9.2e} '
                  f'{L["eps_units"]:9.2e} {L["lin_mean"]:6.3f} {L["lin_p90"]:7.3f}')
        print(f'  all: e p10/p50/p90 {row["e_p10"]:.2e} / {row["e_p50"]:.2e} / {row["e_p90"]:.2e}'
              f'   linearity mean {row["lin_mean"]:.4f}')
        print(f'  read_floor putting the median read at linearity 0.5: '
              f'{row["tau_for_median_lin_half"]:.3g}')

    if args.out:
        with open(args.out, 'w') as f:
            json.dump(rows, f, indent=1)
        print(f'\nwrote {args.out}')


if __name__ == '__main__':
    main()
