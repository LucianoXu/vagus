# Does a scaled memory read as a weaker memory? The fade diagnostic.
#
# The joint-mode rounds found no predictively benign path from the full
# memory m to the blank one: halving m barely moved the prediction
# (per-hop KL 0.04), then the landscape fell off a cliff. The first
# explanation — the per-head readout RMSNorm makes the read scale-
# invariant — was refuted at the whole-read level on 2026-09-24
# (read_regime on main: LAX1's median read sits near the RMSNorm knee,
# mean linearity 0.39). That probe scaled the WHOLE read. What
# consolidation scales is only the carried memory; the in-window writes
# of Y are untouched. This script measures exactly that, at three
# levels, on X | Y items (items.py):
#
#   1. prediction, all layers: m -> c m on every layer, score Y,
#      KL(p(.|m) || p(.|c m)) per position bucket and nll. The shape of
#      KL(c) / KL(0) over c is the question: graded fading rises
#      steadily as c falls; a gauge stays near 0 and jumps late.
#   2. prediction, one layer at a time: only layer l's memory scaled
#      (c = 0.5 and 0), everything else intact. Tells which layers the
#      prediction depends on and whether single-layer effects add up.
#   3. readout, one layer at a time, upstream frozen: layer l's mixer
#      output on the SAME inputs h_l (captured from the full-memory
#      pass) with its memory at c m_l. The memory's contribution is
#      d(c) = out(c m_l) - out(0), and
#          gain(c) = |d(c)| / |d(1)|,  cos(c) = cos(d(c), d(1)).
#      A linear readout gives gain = c, cos = 1; a scale-invariant one
#      gain ~ 1 until c is small. This is the readout-level
#      memory-only linearity, the thing LAX5's floor was meant to change.
#
#   python -m infra.consolidate.fade --ckpt runs/gen-eval/lax1-model-final.pt \
#       --data data/tokenized/fineweb-edu-100BT-mistral32k --out runs/eval/fade-lax1.json

import argparse
import copy
import json
import time

import numpy as np
import torch

from ..dataset.loader import TokenStore
from ..inference import Generator
from .items import item_ids, sample_items
from .memory import block_states, degrade, with_block_states
from .probe import nll_positions

LEVELS = (1.0, 0.75, 0.5, 0.25, 0.1, 0.03, 0.01, 0.0)
LAYER_LEVELS = (0.5, 0.0)
READ_LEVELS = (0.75, 0.5, 0.25, 0.1, 0.03, 0.01)


def kl_positions(p_logits: torch.Tensor, q_logits: torch.Tensor) -> torch.Tensor:
    '''(B, L) KL(p || q) in fp32.'''
    lp = p_logits.float().log_softmax(-1)
    lq = q_logits.float().log_softmax(-1)
    return (lp.exp() * (lp - lq)).sum(-1)


def buckets(t: torch.Tensor, edges: tuple[int, ...]) -> dict[str, float]:
    L = t.shape[1]
    bounds = [0, *[e for e in edges if e < L], L]
    return {f'b{lo}_{hi}': float(t[:, lo:hi].mean()) for lo, hi in zip(bounds[:-1], bounds[1:])}


class Fade:
    def __init__(self, gen: Generator):
        self.gen = gen
        self.model = gen.model
        self.atts = [blk.att for blk in self.model.blocks]

    @torch.no_grad()
    def read(self, x: torch.Tensor, max_len: int) -> dict:
        self.gen.reset(x.shape[0], max_len=max_len)
        self.gen.prefill_ids(x.to(self.gen.device))
        return self.gen.export_state()

    @torch.no_grad()
    def score(self, m: dict | None, y: torch.Tensor, max_len: int, capture: bool = False):
        '''Logits of y from memory m (None = fresh stream), and with capture the
        per-layer mixer inputs h_l of this pass.'''
        if m is None:
            self.gen.reset(y.shape[0], max_len=max_len)
        else:
            self.gen.load_state(m)
        assert self.gen.pending is not None
        block = torch.cat([self.gen.pending[:, None], y[:, :-1].to(self.gen.device)], dim=1)
        hs: list[torch.Tensor] = []
        undo = []
        if capture:
            for att in self.atts:
                orig = att.decode_step

                def wrapped(h, _orig=orig):
                    hs.append(h.detach().clone())
                    return _orig(h)
                att.decode_step = wrapped
                undo.append(att)
        try:
            out = self.model.decode_step(block, return_logits=True)
        finally:
            for att in undo:
                del att.decode_step          # back to the class method
        assert out is not None
        return out, hs

    @torch.no_grad()
    def layer_out(self, l: int, m: dict, S: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        '''Layer l's mixer output on inputs h, starting from m's cache with the
        matrix state replaced by S.'''
        att = self.atts[l]
        cache = copy.copy(m['cache']['blocks'][l]['att'])
        cache['state'] = S
        att.load_cache(cache, None)
        return att.decode_step(h).float()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--label', default=None)
    ap.add_argument('--data', required=True)
    ap.add_argument('--n-items', type=int, default=32)
    ap.add_argument('--x-len', type=int, default=2048)
    ap.add_argument('--y-len', type=int, default=512)
    ap.add_argument('--gap', type=int, nargs='+', default=[0, 256])
    ap.add_argument('--batch', type=int, default=2)
    ap.add_argument('--buckets', type=int, nargs='+', default=[64, 256])
    ap.add_argument('--dtype', default='float32')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    edges = tuple(args.buckets)

    gen, meta = Generator.from_checkpoint(args.ckpt, device=args.device, dtype=args.dtype,
                                          with_meta=True)
    fade = Fade(gen)
    NL = len(fade.atts)
    store = TokenStore(args.data)
    max_len = 1 + args.x_len + args.y_len
    record = dict(ckpt=args.ckpt, label=args.label, step=meta.get('step'),
                  model_args=meta['model_args'], x_len=args.x_len, y_len=args.y_len,
                  n_items=args.n_items, dtype=args.dtype, seed=args.seed, levels=list(LEVELS),
                  layer_levels=list(LAYER_LEVELS), read_levels=list(READ_LEVELS), gaps={})
    t0 = time.time()
    for gap in args.gap:
        rng = np.random.default_rng(args.seed)                 # same documents across gaps
        items = sample_items(store, rng, args.n_items, args.x_len, args.y_len, gap)
        acc_kl = {c: [] for c in LEVELS}
        acc_nll = {c: [] for c in LEVELS}
        acc_nll3 = []
        acc_layer = {(l, c): [] for l in range(NL) for c in LAYER_LEVELS}
        acc_gain = {(l, c): [] for l in range(NL) for c in READ_LEVELS}
        acc_cos = {(l, c): [] for l in range(NL) for c in READ_LEVELS}
        acc_dnorm = {l: [] for l in range(NL)}
        for b0 in range(0, len(items), args.batch):
            pairs = [item_ids(store, it) for it in items[b0:b0 + args.batch]]
            x = torch.stack([p[0] for p in pairs])
            y = torch.stack([p[1] for p in pairs]).to(gen.device)
            m = fade.read(x, max_len)
            full, hs = fade.score(m, y, max_len, capture=True)
            assert len(hs) == NL
            # 1. all layers scaled
            for c in LEVELS:
                logits = full if c == 1.0 else fade.score(degrade(m, c, 'scale'), y, max_len)[0]
                acc_kl[c].append(kl_positions(full, logits).cpu())
                acc_nll[c].append(nll_positions(logits, y).cpu())
            blank, _ = fade.score(None, y, max_len)
            acc_nll3.append(nll_positions(blank, y).cpu())
            # 2. one layer scaled
            Ss = block_states(m)
            for l in range(NL):
                for c in LAYER_LEVELS:
                    new = [S * c if i == l else S for i, S in enumerate(Ss)]
                    logits, _ = fade.score(with_block_states(m, new), y, max_len)
                    acc_layer[(l, c)].append(kl_positions(full, logits).cpu())
            # 3. readout of layer l on frozen inputs
            for l in range(NL):
                S = Ss[l]
                o1 = fade.layer_out(l, m, S, hs[l])
                o0 = fade.layer_out(l, m, torch.zeros_like(S), hs[l])
                d1 = o1 - o0
                n1 = d1.norm(dim=-1)
                acc_dnorm[l].append((n1 / o1.norm(dim=-1).clamp(min=1e-12)).cpu())
                for c in READ_LEVELS:
                    dc = fade.layer_out(l, m, c * S, hs[l]) - o0
                    acc_gain[(l, c)].append((dc.norm(dim=-1) / n1.clamp(min=1e-12)).cpu())
                    acc_cos[(l, c)].append(torch.nn.functional.cosine_similarity(dc, d1, dim=-1).cpu())
            print(f'gap {gap}: {b0 + len(pairs)}/{len(items)} items  {time.time() - t0:.0f}s', flush=True)

        cat = lambda xs: torch.cat(xs)
        nll1 = cat(acc_nll[1.0])
        nll3 = cat(acc_nll3)
        kl0 = buckets(cat(acc_kl[0.0]), edges)
        out = dict(
            benefit=buckets(nll3 - nll1, edges),
            kl={str(c): buckets(cat(acc_kl[c]), edges) for c in LEVELS},
            kl_frac={str(c): {k: v / max(kl0[k], 1e-12) for k, v in buckets(cat(acc_kl[c]), edges).items()}
                     for c in LEVELS},
            nll={str(c): buckets(cat(acc_nll[c]), edges) for c in LEVELS},
            nll_blank=buckets(nll3, edges),
            layer_kl={str(c): [buckets(cat(acc_layer[(l, c)]), edges) for l in range(NL)]
                      for c in LAYER_LEVELS},
            read_gain={str(c): [buckets(cat(acc_gain[(l, c)]), edges) for l in range(NL)]
                       for c in READ_LEVELS},
            read_cos={str(c): [buckets(cat(acc_cos[(l, c)]), edges) for l in range(NL)]
                      for c in READ_LEVELS},
            mem_share=[buckets(cat(acc_dnorm[l]), edges) for l in range(NL)],
        )
        record['gaps'][str(gap)] = out

        # --- print ---
        bk = list(kl0)
        print(f'\n=== gap {gap}: {len(items)} items, x {args.x_len}, y {args.y_len} ===')
        print('benefit nll3-nll1 ' + '  '.join(f'{k} {out["benefit"][k]:.3f}' for k in bk))
        print('\n1. all layers m -> c m: KL(full || c) and as a fraction of KL(blank)')
        print(f'{"c":>6s} ' + ' '.join(f'{k:>18s}' for k in bk))
        for c in LEVELS:
            print(f'{c:6.2f} ' + ' '.join(
                f'{out["kl"][str(c)][k]:8.4f} ({out["kl_frac"][str(c)][k]:5.2f})' for k in bk))
        print('\n2. one layer m_l -> c m_l: KL on all of Y')
        allb = lambda d: float(np.mean(list(d.values())))
        s05 = sum(allb(out['layer_kl']['0.5'][l]) for l in range(NL))
        s0 = sum(allb(out['layer_kl']['0.0'][l]) for l in range(NL))
        print(f'  sum over layers: c=0.5 {s05:.4f}  c=0 {s0:.4f}   joint: c=0.5 '
              f'{allb(out["kl"]["0.5"]):.4f}  c=0 {allb(out["kl"]["0.0"]):.4f}')
        print(f'{"layer":>5s} {"kl@.5":>8s} {"kl@0":>8s} {"mem share":>9s}   read gain at c (first bucket | rest), cos@0.1')
        for l in range(NL):
            g = ' '.join(f'{out["read_gain"][str(c)][l][bk[0]]:.2f}|{np.mean([out["read_gain"][str(c)][l][k] for k in bk[1:]]):.2f}'
                         for c in READ_LEVELS)
            print(f'{l:5d} {allb(out["layer_kl"]["0.5"][l]):8.4f} {allb(out["layer_kl"]["0.0"][l]):8.4f} '
                  f'{allb(out["mem_share"][l]):9.3f}   {g}   {allb(out["read_cos"]["0.1"][l]):.3f}')
        print(f'  read levels {READ_LEVELS}; linear readout = c, scale-invariant ~ 1')

    if args.out:
        with open(args.out, 'w') as f:
            json.dump(record, f, indent=1)
        print(f'\nwrote {args.out}')


if __name__ == '__main__':
    main()
