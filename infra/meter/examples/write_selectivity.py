# Is the memory's write gate selective, or does every token cost the same?
#
# The delta rule writes  S <- S + w_t (v_t - S^T k_t)^T,  so the state
# change a token buys is |w_t| |r_t| with r_t the residual and, for the
# symmetric write, |w_t| = beta_t (the keys are L2-normalised). A memory
# with 128 slots and a 2048-token stream is oversubscribed 16x, so what
# it spends those slots on is the whole question.
#
# Measured on the trained LAX1-340M (2026-09-10, 46k tokens): beta is
# near-saturated (mean 0.80, median 0.90) and carries no information
# about the residual — corr(beta, rho) = -0.031, write gain 0.993 — and
# the write mass is nearly uniform: the top 10% of tokens by surprise
# take 15.0% of it against a uniform 10%. Nothing is selective.
#
# That measurement is what motivated LAX2 (Kalman Delta Networks): the
# gain there is a function of accumulated evidence, not of the token
# alone, so if it does anything these numbers must move. This script is
# that read, so the LAX1 baseline and the LAX2 endpoint are the same
# measurement rather than two scripts that agree by hand.
#
#   python -m infra.meter.examples.write_selectivity \
#       --ckpt LAX1=runs/.../model-final.pt --ckpt LAX2=runs/.../model-final.pt \
#       --data data/tokenized/fineweb-edu-100BT-mistral32k --batches 6
#
# What each number means:
#   corr(w, rho)   per head, over tokens. The direct question: does the
#                  gate know what the memory failed to predict?
#   write gain     mean |w| on the top-10% most surprising tokens over
#                  mean |w| on the bottom half. 1.0 is no selectivity.
#   concentration  share of total |w| * rho held by the top k% of tokens.
#                  Uniform is k%. This is the one that matters: it counts
#                  the automatic weighting by rho as well as the gate's,
#                  so it says whether the memory is selective AT ALL.
#
# The recurrence is written out rather than imported, as in
# saturation_curve.py, and it goes through GatedDeltaNet._write, so a
# Kalman mixer measures correctly (|w| is beta for the symmetric write).

import argparse
import json

import numpy as np
import torch

from ...components.linear_attention import GatedDeltaNet
from ...dataset.loader import TokenStore
from ...models.io import load_model


@torch.no_grad()
def write_and_residual(model, tokens: torch.Tensor, warmup: int):
    '''(|w|, rho) each (layers, tokens, heads), positions < warmup dropped.

    warmup exists because the state starts blank: rho is exactly 1.0 at
    position 0 and still falling steeply for the first few dozen tokens,
    which would dominate any correlation with a quantity that is not.'''
    x = model.embedding(tokens)
    ws, rhos = [], []
    for blk in model.blocks:
        att = blk.att
        assert isinstance(att, GatedDeltaNet), \
            f'{type(att).__name__}: this probe is for all-GDN layouts'
        h = blk.rmsnorm1(x)
        _, k, v = att._heads(att.wq(h), att.wk(h), att.wv(h))
        g, beta = att._gates(h)
        beta, w = att._write(h, k, g, beta)
        B, L, H, dk = k.shape
        k32, v32 = k.float(), v.float()
        S = torch.zeros(B, H, dk, v.shape[-1], device=k.device, dtype=torch.float32)
        rho = torch.empty(B, L, H, device=k.device, dtype=torch.float32)
        wn = torch.empty(B, L, H, device=k.device, dtype=torch.float32)
        for t in range(L):
            if g is not None:
                a = g[:, t].float().exp()
                S = S * (a[..., None] if a.dim() == 3 else a[..., None, None])
            kt, vt = k32[:, t], v32[:, t]
            r = vt - torch.einsum('bhk,bhkv->bhv', kt, S)
            rho[:, t] = r.norm(dim=-1) / vt.norm(dim=-1).clamp(min=1e-6)
            wt = w[:, t].float() if w is not None else kt * beta[:, t].float()[..., None]
            wn[:, t] = wt.norm(dim=-1)
            S = S + torch.einsum('bhk,bhv->bhkv', wt, r)
        ws.append(wn[:, warmup:].reshape(-1, H).cpu())
        rhos.append(rho[:, warmup:].reshape(-1, H).cpu())
        x = x + blk.att(h)
        x = x + blk.ffn(blk.rmsnorm2(x))
    return torch.stack(ws).numpy(), torch.stack(rhos).numpy()


def share(a: np.ndarray, pct: float) -> float:
    '''Mean over heads of the share of total mass held by the top pct%.'''
    out = []
    for li in range(a.shape[0]):
        for h in range(a.shape[2]):
            v = np.sort(a[li, :, h])[::-1]
            out.append(v[:max(1, int(len(v) * pct / 100))].sum() / max(v.sum(), 1e-12))
    return 100 * float(np.mean(out))


def report(label: str, w: np.ndarray, rho: np.ndarray) -> dict:
    NL, T, H = w.shape
    corr = np.full((NL, H), np.nan)
    gain = np.ones((NL, H))
    for li in range(NL):
        for h in range(H):
            a, b = w[li, :, h], rho[li, :, h]
            if a.std() < 1e-8 or b.std() < 1e-8:
                continue
            corr[li, h] = np.corrcoef(a, b)[0, 1]
            hi, lo = b >= np.percentile(b, 90), b <= np.percentile(b, 50)
            gain[li, h] = a[hi].mean() / max(a[lo].mean(), 1e-12)
    mass = w * rho
    out = dict(
        label=label, tokens=int(T), layers=int(NL), heads=int(H),
        w_mean=float(w.mean()), w_p50=float(np.percentile(w, 50)),
        w_p90=float(np.percentile(w, 90)),
        rho_mean=float(rho.mean()), rho_p50=float(np.percentile(rho, 50)),
        corr_mean=float(np.nanmean(corr)), corr_median=float(np.nanmedian(corr)),
        corr_se=float(1 / np.sqrt(T)),
        gain_median=float(np.median(gain)),
        mass_top10=share(mass, 10), mass_top25=share(mass, 25),
        w_top10=share(w, 10), rho_top10=share(rho, 10))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--ckpt', action='append', required=True, metavar='LABEL=PATH')
    ap.add_argument('--data', required=True)
    ap.add_argument('--context', type=int, default=1024)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--batches', type=int, default=6)
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
        W, R = [], []
        for _ in range(args.batches):
            ids = np.stack([
                store.read_window(int(rng.integers(len(store.tokens))),
                                  int(rng.integers(0, 10_000_000)), args.context)
                for _ in range(args.batch)]).astype(np.int64)
            w, r = write_and_residual(model, torch.from_numpy(ids).to(args.device),
                                      args.warmup)
            W.append(w); R.append(r)
        row = report(label, np.concatenate(W, axis=1), np.concatenate(R, axis=1))
        row['write'] = meta['model_args'].get('write', 'delta')
        row['step'] = meta.get('step')
        rows.append(row)
        del model
        print(f'{label}: write={row["write"]} step={row["step"]} '
              f'{row["tokens"]} samples/layer', flush=True)

    print('\nwrite strength |w| (= beta for the symmetric write)')
    hdr = f'{"":16s} ' + ' '.join(f'{r["label"]:>10s}' for r in rows)
    print(hdr); print('-' * len(hdr))
    for name, key in (('mean', 'w_mean'), ('median', 'w_p50'), ('p90', 'w_p90'),
                      ('rho mean', 'rho_mean'), ('rho median', 'rho_p50')):
        print(f'{name:16s} ' + ' '.join(f'{r[key]:10.4f}' for r in rows))

    print('\ndoes the gate track what the memory failed to predict?')
    for name, key in (('corr(w, rho)', 'corr_mean'), ('  median', 'corr_median'),
                      ('  se/head', 'corr_se'), ('write gain', 'gain_median')):
        print(f'{name:16s} ' + ' '.join(f'{r[key]:+10.4f}' for r in rows))
    print('   write gain = mean |w| on top-10% surprise / bottom-50%; 1.0 = none')

    print('\nwrite-mass concentration (share held by the top k% of tokens)')
    for name, key, uni in (('|w| * rho top10', 'mass_top10', 10),
                           ('|w| * rho top25', 'mass_top25', 25),
                           ('rho alone top10', 'rho_top10', 10),
                           ('|w| alone top10', 'w_top10', 10)):
        print(f'{name:16s} ' + ' '.join(f'{r[key]:9.1f}%' for r in rows)
              + f'   (uniform {uni}%)')

    if args.out:
        with open(args.out, 'w') as f:
            json.dump(rows, f, indent=1)
        print(f'\nwrote {args.out}')


if __name__ == '__main__':
    main()
