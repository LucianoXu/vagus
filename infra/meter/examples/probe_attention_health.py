'''
Attention health probe: where a model's softmax attention actually sits on
the sharpness scale, and which factor is driving it.

Motivation (HAX2-1.3B, 2026-09-11): attn_logit_max — the trainer's slow
metric — climbed 8 -> 36 over 12,600 steps and was still accelerating,
while the paired HAX1-340M run saturated at 26. The max alone cannot say
whether that is one odd head or a systematic entropy collapse, nor which
term grew. This script decomposes it.

With qk-norm, RMSNorm makes every head's q and k have the same length,
so the logit factorises exactly:

    logit(i,j,h) = q_h(i) . k_h(j) / sqrt(Dh)
                 = sqrt(Dh) * g_q * g_k * cos(q_h(i), k_h(j))

    g_q = RMS(gamma_q), g_k = RMS(gamma_k)   (one gamma per LAYER: the
    qk-norm gammas are shape (Dh,), broadcast over heads, so the scale
    term is shared by every head of a layer and only the cosine is per
    head)

    ceiling = sqrt(Dh) * g_q * g_k   is the largest logit the layer can
    produce; observed_max / ceiling is the largest cosine any head
    reaches. Growth in the gammas (a layer-wide scale the optimizer can
    push without limit, since 1-D params carry no weight decay) and
    growth in the cosine (heads genuinely aligning q with k) are
    different problems with different fixes, and this separates them.

What matters downstream is not the logit but the softmax it feeds, so
the probe also reports, per head, on real tokens:

    gap      mean over query positions of (top-1 - top-2) logit. A gap
             of ~20 nats means the row is one-hot to 2e-9: that head's
             attention pattern has stopped receiving gradient.
    entropy  mean row entropy in nats. log(i+1) is the uniform ceiling
             at position i; ~0 is full collapse.
    sink     mean attention mass on position 0 (the sink pathway gated
             attention is supposed to remove).
    gate     mean sigmoid output gate, and the fraction of (position,
             channel) pairs above 0.9 / below 0.1. The working
             hypothesis for HAX2 is that the gate decouples "does this
             head fire here" from "where does it look", removing the
             pressure that kept an ungated head's logits moderate; if
             so, the sharpest heads are the ones whose gate is mostly
             shut, which shows up as a negative gate/cosine correlation.

Run (one A100 is plenty; fp32 weights, scores materialised per layer):

    python -m infra.meter.examples.probe_attention_health \\
        runs/<run>/ckpt-00005000.pt runs/<run>/ckpt-00010000.pt \\
        --data-dir data/tokenized/fineweb-edu-100BT-mistral32k --seq 4096

Several checkpoints print a per-layer delta table, which is what says
whether the growth is in the scale or in the cosines.
'''

import argparse
import json
import math
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from infra.components.attention import SoftmaxAttention
from infra.components.parallel_mixer import ParallelMixer
from infra.dataset.loader import TokenStore, WindowLoader
from infra.models.io import load_model


def pick_device() -> str:
    return 'cuda' if torch.cuda.is_available() else 'cpu'


@torch.no_grad()
def probe_branch(att: SoftmaxAttention, h: torch.Tensor, skip: int) -> dict:
    '''Per-head attention statistics for one softmax branch on input h
    (the block's post-norm residual). Scores are materialised in fp32,
    one layer at a time: (B, H, L, L) at B=1, H=8, L=4096 is 0.5 GiB.'''
    B, L = h.shape[0], h.shape[1]
    H, Hkv = att.head_count, att.kv_head_count
    Dh = att.dim // att.head_count

    q = att.wq(h).reshape(B, L, H, Dh).transpose(1, 2)
    k = att.wk(h).reshape(B, L, Hkv, Dh).transpose(1, 2)
    if att.qk_norm:
        q, k = att.q_norm(q), att.k_norm(k)
    if att.rope is not None:
        q, k = att.rope(q), att.rope(k)
    if Hkv != H:
        k = k.repeat_interleave(H // Hkv, dim=1)
    q, k = q.float(), k.float()

    # the layer-wide scale term, and the ceiling it implies
    g_q = float(att.q_norm.gamma.float().pow(2).mean().sqrt()) if att.qk_norm else 1.0
    g_k = float(att.k_norm.gamma.float().pow(2).mean().sqrt()) if att.qk_norm else 1.0
    # RMS(gamma) gives the TYPICAL scale, so observed/ceiling can exceed 1 when a
    # head routes its q/k energy into the high-gamma channels. The hard bound uses
    # max|gamma|; outlier-channel growth under qk-norm is what Qwen3-Next reported
    # (a released Qwen3-0.6B k_norm gain reaches 96) and fixed with zero-centred
    # gains plus weight decay, so track both.
    m_q = float(att.q_norm.gamma.float().abs().max()) if att.qk_norm else 1.0
    m_k = float(att.k_norm.gamma.float().abs().max()) if att.qk_norm else 1.0
    ceiling = math.sqrt(Dh) * g_q * g_k
    hard = math.sqrt(Dh) * m_q * m_k
    # what the vectors actually are, as a check on the factorisation
    qn = float(q.pow(2).sum(-1).sqrt().mean())
    kn = float(k.pow(2).sum(-1).sqrt().mean())

    causal = torch.ones(L, L, dtype=torch.bool, device=h.device).tril()
    rows = slice(skip, L)          # early rows have too few keys to be informative
    heads = []
    for head in range(H):
        s = (q[:, head] @ k[:, head].transpose(-1, -2)) / math.sqrt(Dh)   # (B, L, L)
        s = s.masked_fill(~causal, float('-inf'))[:, rows]
        p = torch.softmax(s, dim=-1)
        top2 = s.topk(2, dim=-1).values
        ent = -(p * torch.log(p.clamp_min(1e-30))).sum(-1)
        heads.append({
            'logit_max': float(s.max()),
            'logit_p99': float(torch.quantile(s[s > -1e30].flatten().float(), 0.99)),
            'gap': float((top2[..., 0] - top2[..., 1]).mean()),
            'gap_p99': float(torch.quantile((top2[..., 0] - top2[..., 1]).flatten(), 0.99)),
            'entropy': float(ent.mean()),
            'ent_frac': float(ent.mean()) / math.log(L),
            'entropy_min': float(ent.min()),
            'sink': float(p[..., 0].mean()),
        })
        del s, p, top2, ent

    out = {'g_q': g_q, 'g_k': g_k, 'm_q': m_q, 'm_k': m_k, 'ceiling': ceiling,
           'hard': hard, 'q_norm': qn, 'k_norm': kn, 'heads': heads, 'Dh': Dh, 'H': H,
           'L': L, 'ent_uniform': math.log(L)}
    for rec in heads:
        rec['cos_max'] = rec['logit_max'] / ceiling if ceiling else float('nan')

    if att.out_gate:
        g = torch.sigmoid(att.wg(h).float())                    # (B, L, out_width)
        gh = g.reshape(B, L, H, att.out_width // H)
        out['gate'] = [{'mean': float(gh[..., i, :].mean()),
                        'open': float((gh[..., i, :] > 0.9).float().mean()),
                        'shut': float((gh[..., i, :] < 0.1).float().mean())}
                       for i in range(H)]
    return out


@torch.no_grad()
def probe_model(model, tokens: torch.Tensor, skip: int) -> dict:
    '''Walk the residual stream, probing every softmax branch on the way.
    Submodules are called directly (not Block.forward), so nothing here
    touches a compiled graph.'''
    x = model.embedding(tokens)
    layers = {}
    for i, blk in enumerate(model.blocks):
        h = blk.rmsnorm1(x)
        mixer = blk.att
        att = mixer.att if isinstance(mixer, ParallelMixer) else mixer
        if isinstance(att, SoftmaxAttention):
            layers[i] = probe_branch(att, h, skip)
        x = x + mixer(h)
        x = x + blk.ffn(blk.rmsnorm2(x))
    return layers


def batch_from_store(data_dir: str, seq: int, batch: int, seed: int, device) -> torch.Tensor:
    store = TokenStore(data_dir)
    loader = WindowLoader(store, seq, batch, seed=seed, shuffle=True, prefetch=0)
    x, _ = next(iter(loader))
    return x.to(device)


def report(name: str, layers: dict, verbose: bool) -> None:
    print(f'\n=== {name}')
    L0 = next(iter(layers.values()))
    print(f'(entropy ceiling at L={L0["L"]}: {L0["ent_uniform"]:.2f} nats; '
          f'entfr = mean row entropy / that ceiling)')
    print(f'{"layer":>5} {"gRMS":>6} {"gmax":>6} {"ceil":>7} {"hard":>7} '
          f'{"max":>7} {"cosmax":>7} {"gap":>7} {"gapp99":>7} {"ent":>6} {"entfr":>6} {"entmin":>7} {"sink":>6}'
          + (f' {"gate":>6} {"shut":>6}' if 'gate' in next(iter(layers.values())) else ''))
    for i, L in sorted(layers.items()):
        hs = L['heads']
        row = (f'{i:5d} {L["g_q"]:6.3f} {max(L["m_q"], L["m_k"]):6.3f} {L["ceiling"]:7.2f} '
               f'{L["hard"]:7.1f} '
               f'{max(h["logit_max"] for h in hs):7.2f} '
               f'{max(h["cos_max"] for h in hs):7.3f} '
               f'{sum(h["gap"] for h in hs)/len(hs):7.2f} '
               f'{max(h["gap_p99"] for h in hs):7.2f} '
               f'{sum(h["entropy"] for h in hs)/len(hs):6.2f} '
               f'{sum(h["ent_frac"] for h in hs)/len(hs):6.3f} '
               f'{min(h["entropy_min"] for h in hs):7.3f} '
               f'{max(h["sink"] for h in hs):6.3f}')
        if 'gate' in L:
            row += f' {sum(g["mean"] for g in L["gate"])/len(L["gate"]):6.3f} {max(g["shut"] for g in L["gate"]):6.3f}'
        print(row)
        if verbose:
            for j, h in enumerate(hs):
                g = f' gate {L["gate"][j]["mean"]:.3f} shut {L["gate"][j]["shut"]:.3f}' if 'gate' in L else ''
                print(f'      head {j}: max {h["logit_max"]:6.2f} cos {h["cos_max"]:.3f} '
                      f'gap {h["gap"]:6.2f} (p99 {h["gap_p99"]:6.2f}) ent {h["entropy"]:5.2f} '
                      f'min {h["entropy_min"]:.3f} sink {h["sink"]:.3f}{g}')

    # the hypothesis test: do the sharpest heads have the most-shut gates?
    if 'gate' in next(iter(layers.values())):
        cos, shut = [], []
        for L in layers.values():
            cos += [h['cos_max'] for h in L['heads']]
            shut += [g['shut'] for g in L['gate']]
        t = torch.tensor([cos, shut])
        if float(t[1].std()) < 1e-6 or float(t[0].std()) < 1e-6:
            print('corr(cos_max, gate-shut): undefined (no spread in one of them)')
        else:
            c = float(torch.corrcoef(t)[0, 1])
            print(f'corr(head cos_max, head gate-shut fraction) = {c:+.3f}  '
                  f'(near zero = the gate does not explain sharpness; '
                  f'strongly positive = the shut heads are the sharp ones)')


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('checkpoints', nargs='+')
    p.add_argument('--data-dir', default='data/tokenized/fineweb-edu-100BT-mistral32k')
    p.add_argument('--seq', type=int, default=4096)
    p.add_argument('--batch', type=int, default=1)
    p.add_argument('--skip', type=int, default=64,
                   help='query positions to skip (early rows see too few keys)')
    p.add_argument('--seed', type=int, default=1234)
    p.add_argument('--device', default=None)
    p.add_argument('--verbose', action='store_true', help='per-head lines')
    p.add_argument('--json', default=None, help='also write the raw stats here')
    args = p.parse_args()

    device = torch.device(args.device or pick_device())
    tokens = batch_from_store(args.data_dir, args.seq, args.batch, args.seed, device)
    print(f'probe batch {tuple(tokens.shape)} from {args.data_dir} (seed {args.seed}) on {device}')

    all_stats = {}
    for path in args.checkpoints:
        model, meta = load_model(path, device=device, dtype='float32')
        model.eval()
        name = f'{pathlib.Path(path).name} (step {meta.get("step")})'
        layers = probe_model(model, tokens, args.skip)
        report(name, layers, args.verbose)
        all_stats[path] = {'meta': {k: meta.get(k) for k in ('step', 'tokens_seen')},
                           'layers': {str(k): v for k, v in layers.items()}}
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    if len(args.checkpoints) > 1:
        a, b = args.checkpoints[0], args.checkpoints[-1]
        la, lb = all_stats[a]['layers'], all_stats[b]['layers']
        print(f'\n=== delta: {pathlib.Path(a).name} -> {pathlib.Path(b).name}')
        print(f'{"layer":>5} {"d_scale":>9} {"d_cosmax":>9} {"d_max":>8} {"d_gap":>8} {"d_ent":>8}')
        for key in sorted(la, key=int):
            A, B = la[key], lb[key]
            ca = max(h['cos_max'] for h in A['heads']); cb = max(h['cos_max'] for h in B['heads'])
            ga = sum(h['gap'] for h in A['heads'])/len(A['heads'])
            gb = sum(h['gap'] for h in B['heads'])/len(B['heads'])
            ea = sum(h['entropy'] for h in A['heads'])/len(A['heads'])
            eb = sum(h['entropy'] for h in B['heads'])/len(B['heads'])
            print(f'{int(key):5d} {B["ceiling"]/A["ceiling"]:9.3f} {cb/ca:9.3f} '
                  f'{max(h["logit_max"] for h in B["heads"]) - max(h["logit_max"] for h in A["heads"]):8.2f} '
                  f'{gb-ga:8.2f} {eb-ea:8.2f}')
        print('d_scale = ceiling ratio (the qk-norm gammas); d_cosmax = cosine ratio '
              '(genuine q/k alignment). Whichever is far from 1.0 is the driver.')

    if args.json:
        pathlib.Path(args.json).write_text(json.dumps(all_stats, indent=1))
        print(f'\nwrote {args.json}')


if __name__ == '__main__':
    main()
