# Budget-PPL evaluation for the eviction-only block (branch evict-only).
#
# Same protocol as budget_ppl.py (held-out = last manifest shard, NLL
# over positions >= block_len, fp32, per-sequence means kept for
# pairing), cells:
#   - full            : no management (exact softmax attention)
#   - m<budget>e_<sf> : eviction-only at that budget with score form sf
# The score form 'p2' reproduces budget_ppl's m<budget>e rows exactly
# (regression anchor); 'lin' is the deployable default.
#
#   python -m infra.eval.evict_ppl --ckpt runs/.../model-final.pt \
#       [--ckpt more.pt ...] --data_dir data/tokenized/... \
#       --out runs/eval/evict.json [--budgets 32 64 128 256 512] \
#       [--scores lin] [--seq_len 2048 4096] [--n_seq 128]

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F

from ..components.evict import EvictCfg, Health, stream_hidden
from ..dataset.loader import TokenStore
from .budget_ppl import holdout_sequences, load_model


@torch.no_grad()
def seq_nll(model, seqs: torch.Tensor, cfg: EvictCfg, manage: bool,
            device, batch: int = 8) -> tuple[list[float], dict]:
    out, health = [], Health()
    for i in range(0, len(seqs), batch):
        chunk = seqs[i:i + batch].to(device)
        x, y = chunk[:, :-1], chunk[:, 1:]
        hidden = stream_hidden(model, x, cfg, manage=manage, health=health)
        logits = model.head(hidden).float()
        nll = F.cross_entropy(logits.transpose(1, 2), y, reduction='none')
        out += nll[:, cfg.block_len:].mean(dim=1).tolist()
    return out, health.as_dict()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', action='append', required=True)
    ap.add_argument('--data_dir', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--budgets', type=int, nargs='*', default=[32, 64, 128, 256, 512])
    ap.add_argument('--scores', nargs='*', default=['lin'],
                    choices=['lin', 'sq', 'p2'])
    ap.add_argument('--seq_len', type=int, nargs='+', default=[2048, 4096])
    ap.add_argument('--no_full', action='store_true')
    ap.add_argument('--n_seq', type=int, default=128)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--block_len', type=int, default=256)
    ap.add_argument('--ring_window', type=int, default=32)
    ap.add_argument('--lookahead', type=int, default=0)
    ap.add_argument('--compile', action='store_true')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    full_store = TokenStore(args.data_dir)
    holdout = full_store.entries[-1]['file']
    store = TokenStore(args.data_dir, shards=[holdout])
    print(f'holdout shard: {holdout} ({store.total_tokens:,} tokens)')

    results = {'created': datetime.now().astimezone().isoformat(timespec='seconds'),
               'holdout_shard': holdout, 'n_seq': args.n_seq,
               'block_len': args.block_len, 'ring_window': args.ring_window,
               'lookahead': args.lookahead, 'scores': args.scores,
               'ckpts': {}}
    base = dict(block_len=args.block_len, ring_window=args.ring_window,
                lookahead=args.lookahead, compile_cell=args.compile)
    for ckpt in args.ckpt:
        model, meta = load_model(ckpt, device)
        entry = {'meta': meta, 'cells': {}}
        for L in args.seq_len:
            seqs = holdout_sequences(store, L, args.n_seq if L <= 2048
                                     else max(args.n_seq // 2, 16))
            cells = []
            if not args.no_full:
                cells.append(('full', EvictCfg(**base), False))
            for sf in args.scores:
                for m in args.budgets:
                    cells.append((f'm{m}e_{sf}',
                                  EvictCfg(**base, budget=m, score=sf), True))
            for name, cfg, manage in cells:
                nlls, health = seq_nll(model, seqs, cfg, manage, device,
                                       batch=args.batch)
                mean = sum(nlls) / len(nlls)
                entry['cells'][f'L{L}/{name}'] = {
                    'nll_mean': mean, 'ppl': math.exp(mean),
                    'nll_per_seq': nlls, 'health': health}
                print(f'{Path(ckpt).parent.name}/{Path(ckpt).name} '
                      f'L{L}/{name}: nll {mean:.4f} ppl {math.exp(mean):.2f} '
                      f'{health}', flush=True)
        results['ckpts'][ckpt] = entry
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(results, indent=1))
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
