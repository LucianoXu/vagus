# Budget-PPL evaluation for the unified block (Tier 1).
#
# For each checkpoint: streaming NLL on held-out sequences under
#   - full        : no management (the ceiling; exact softmax attention)
#   - m<budget>   : unified management (evict + demote), the floor
#   - m<budget>e  : eviction-only ablation (--evict_only_at)
# Sequences are contiguous windows of the LAST manifest shard (excluded
# from continued-training recipes via data_shards, so both ct arms never
# trained on it; the original 15B run saw ~13.5% of its windows once —
# constant across every compared checkpoint, so paired deltas are clean).
# NLL is averaged over positions >= block_len (every scored position
# reads a managed cache), fp32, no autocast. Per-sequence means are kept
# so any two protocols/checkpoints pair exactly.
#
# Usage:
#   python -m infra.eval.budget_ppl --ckpt runs/.../model-final.pt \
#       [--ckpt more.pt ...] --data_dir data/tokenized/... \
#       --out runs/eval/budget_ppl.json [--budgets 256 512 1024]
#       [--seq_len 2048 4096] [--n_seq 128]

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F

from ..components.unified import Health, ManageCfg, stream_hidden
from ..dataset.loader import TokenStore
from ..models import build_model


def load_model(path: str, device):
    state = torch.load(path, map_location='cpu', weights_only=False)
    args = state.get('model_args') or state.get('config', {}).get('model_args')
    name = state.get('model_name', 'TransformerPP')
    assert args, f'{path}: no model_args found'
    model = build_model(name, args)
    model.load_state_dict(state.get('model', state))
    model.to(device).eval().float()
    return model, {'step': state.get('step'), 'tokens_seen': state.get('tokens_seen')}


def holdout_sequences(store: TokenStore, seq_len: int, n_seq: int) -> torch.Tensor:
    '''n_seq non-overlapping (seq_len+1)-token rows from shard 0 of the
    (single-shard) store.'''
    total = store.shard_tokens[0]
    need = n_seq * (seq_len + 1)
    assert need <= total, f'holdout shard too small: {need} > {total}'
    rows = [torch.from_numpy(
        store.read_window(0, i * (seq_len + 1), seq_len + 1).astype('int64'))
        for i in range(n_seq)]
    return torch.stack(rows)


@torch.no_grad()
def seq_nll(model, seqs: torch.Tensor, mcfg: ManageCfg, manage: bool,
            device, batch: int = 8) -> tuple[list[float], dict]:
    '''Per-sequence mean NLL (nats) over positions >= block_len.'''
    out, health = [], Health()
    for i in range(0, len(seqs), batch):
        chunk = seqs[i:i + batch].to(device)
        x, y = chunk[:, :-1], chunk[:, 1:]
        hidden = stream_hidden(model, x, mcfg, manage=manage, health=health)
        logits = model.head(hidden).float()
        nll = F.cross_entropy(logits.transpose(1, 2), y, reduction='none')
        out += nll[:, mcfg.block_len:].mean(dim=1).tolist()
    return out, health.as_dict()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', action='append', required=True)
    ap.add_argument('--data_dir', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--budgets', type=int, nargs='*', default=[256, 512, 1024])
    ap.add_argument('--seq_len', type=int, nargs='+', default=[2048, 4096])
    ap.add_argument('--no_full', action='store_true',
                    help='skip the full-cache ceiling cell')
    ap.add_argument('--n_seq', type=int, default=128)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--block_len', type=int, default=256)
    ap.add_argument('--ring_window', type=int, default=32)
    ap.add_argument('--evict_only_at', type=int, nargs='*', default=[512],
                    help='budgets for eviction-only ablation rows; empty disables')
    ap.add_argument('--lam', type=float, default=1.0 / 1024,
                    help='offset-discount rate of the position measure (v2)')
    ap.add_argument('--lam_sens', type=float, default=1.0 / 256,
                    help='second lambda for the L4096/m512 sensitivity cell; 0 disables')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    full_store = TokenStore(args.data_dir)
    holdout = full_store.entries[-1]['file']
    store = TokenStore(args.data_dir, shards=[holdout])
    print(f'holdout shard: {holdout} ({store.total_tokens:,} tokens)')

    results = {'created': datetime.now().astimezone().isoformat(timespec='seconds'),
               'holdout_shard': holdout, 'n_seq': args.n_seq,
               'block_len': args.block_len, 'ring_window': args.ring_window,
               'lam': args.lam, 'lam_sens': args.lam_sens,
               'ckpts': {}}
    for ckpt in args.ckpt:
        model, meta = load_model(ckpt, device)
        entry = {'meta': meta, 'cells': {}}
        for L in args.seq_len:
            seqs = holdout_sequences(store, L, args.n_seq if L <= 2048
                                     else max(args.n_seq // 2, 16))
            cells = []
            if not args.no_full:
                cells.append(('full', ManageCfg(block_len=args.block_len,
                                                ring_window=args.ring_window), False))
            for m in args.budgets:
                cells.append((f'm{m}', ManageCfg(
                    block_len=args.block_len, budget=m,
                    ring_window=args.ring_window, demote=True,
                    lam=args.lam), True))
            for m in args.evict_only_at:
                cells.append((f'm{m}e', ManageCfg(
                    block_len=args.block_len, budget=m,
                    ring_window=args.ring_window, demote=False,
                    lam=args.lam), True))
            if args.lam_sens and L == 4096:
                cells.append((f'm512_lam{round(1 / args.lam_sens)}', ManageCfg(
                    block_len=args.block_len, budget=512,
                    ring_window=args.ring_window, demote=True,
                    lam=args.lam_sens), True))
            for name, mcfg, manage in cells:
                nlls, health = seq_nll(model, seqs, mcfg, manage, device,
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
