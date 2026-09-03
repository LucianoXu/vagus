# Recall probes for the eviction-only line (branch evict-only,
# 2026-09-03): does management-aware CT buy *exact recall*, or only
# smooth-regime PPL? Two synthetic tasks, scored by teacher forcing
# (per-token argmax == target; all-correct == what greedy decoding
# would produce, since the conditioning prefix is then identical):
#
#   passkey : natural filler (Mohtashami & Jaggi), "The pass key is
#             NNNNN. Remember it. NNNNN is the pass key." inserted at
#             depth d, "What is the pass key? The pass key is" at the
#             end; score the 5 digit tokens.
#   kv      : MQAR-style associative recall — K random (key, value)
#             token pairs inserted at depth d, natural filler around
#             them, then Q queried keys at the end each followed by its
#             value; score each value token (induction-head copying).
#
# Protocols: full (manage=False, exact softmax) and m<b>e (eviction-only,
# score lin, block 256). Every sequence is exactly seq_len tokens (a
# block multiple), BOS first (the training data's document convention).
# Under management the needle / pairs sit in earlier blocks and must
# survive the eviction decisions; the question is always in the last
# block (always visible, as in full attention).
#
#   python -m infra.eval.recall_probe --ckpt A.pt [--ckpt B.pt ...] \
#       --out runs/eval/recall.json [--tasks passkey kv] \
#       [--seq_len 1024 2048 4096] [--budgets 32 128 512] \
#       [--depths 0.1 0.3 0.5 0.7 0.9] [--n_trials 16]

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..components.evict import EvictCfg, stream_hidden
from ..tokenizers import mistral32k
from .budget_ppl import load_model

FILLER = ('The grass is green. The sky is blue. The sun is yellow. '
          'Here we go. There and back again. ')
BOS = mistral32k.BOS_ID


class Builder:
    def __init__(self):
        self.tok = mistral32k.load()
        self.filler = self.enc(FILLER)

    def enc(self, text):
        return self.tok.encode(text, add_special_tokens=False).ids

    def filler_stream(self, n):
        reps = n // len(self.filler) + 2
        return (self.filler * reps)[:n]

    def passkey(self, L, depth, rng):
        n = int(rng.integers(10_000, 100_000))
        needle = self.enc(f'The pass key is {n}. Remember it. {n} is the pass key.')
        qa = self.enc(f'What is the pass key? The pass key is {n}')
        ans = qa[-5:]
        assert [self.tok.id_to_token(t) for t in ans] == list(str(n)), ans
        avail = L - 1 - len(needle) - len(qa)
        assert avail > 0
        pre = int(round(depth * avail))
        fs = self.filler_stream(avail)
        seq = [BOS] + fs[:pre] + needle + fs[pre:] + qa
        assert len(seq) == L
        score_pos = list(range(L - 5, L))        # target positions
        return seq, score_pos

    def kv(self, L, depth, rng, K=16, Q=8):
        keys = rng.choice(np.arange(300, 16_000), size=K, replace=False).tolist()
        vals = rng.choice(np.arange(16_000, 32_000), size=K, replace=False).tolist()
        pairs = [t for k, v in zip(keys, vals) for t in (k, v)]
        qidx = rng.choice(K, size=Q, replace=False).tolist()
        queries = [t for i in qidx for t in (keys[i], vals[i])]
        avail = L - 1 - len(pairs) - len(queries)
        assert avail > 0
        pre = int(round(depth * avail))
        fs = self.filler_stream(avail)
        seq = [BOS] + fs[:pre] + pairs + fs[pre:] + queries
        assert len(seq) == L
        q0 = L - len(queries)
        score_pos = [q0 + 2 * i + 1 for i in range(Q)]   # the value slots
        return seq, score_pos


@torch.no_grad()
def score_batch(model, x, score_pos, cfg, manage, device):
    '''Returns per-sequence (all_correct, n_correct, n_scored, sum_nll).'''
    hidden = stream_hidden(model, x, cfg, manage=manage)
    pos = torch.tensor(score_pos, device=device)
    logits = model.head(hidden[:, pos - 1]).float()      # predict x[:, pos]
    tgt = x[:, pos]
    nll = F.cross_entropy(logits.transpose(1, 2), tgt, reduction='none')
    hit = logits.argmax(-1) == tgt
    return (hit.all(-1).float().tolist(), hit.sum(-1).tolist(),
            [hit.shape[1]] * hit.shape[0], nll.sum(-1).tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', action='append', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--tasks', nargs='*', default=['passkey', 'kv'])
    ap.add_argument('--seq_len', type=int, nargs='*', default=[1024, 2048, 4096])
    ap.add_argument('--budgets', type=int, nargs='*', default=[32, 128, 512])
    ap.add_argument('--depths', type=float, nargs='*', default=[0.1, 0.3, 0.5, 0.7, 0.9])
    ap.add_argument('--n_trials', type=int, default=16)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--block_len', type=int, default=256)
    ap.add_argument('--score', default='lin')
    ap.add_argument('--kv_pairs', type=int, default=16)
    ap.add_argument('--kv_queries', type=int, default=8)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    b = Builder()
    results = {'created': datetime.now().astimezone().isoformat(timespec='seconds'),
               'args': vars(args), 'ckpts': {}}
    cells = [('full', None)] + [(f'm{m}e', m) for m in args.budgets]

    for ckpt in args.ckpt:
        model, meta = load_model(ckpt, device)
        entry = {'meta': meta, 'cells': {}}
        for task in args.tasks:
            for L in args.seq_len:
                for depth in args.depths:
                    rng = np.random.default_rng([args.seed, L, int(depth * 1000),
                                                 0 if task == 'passkey' else 1])
                    seqs, sp = [], None
                    for _ in range(args.n_trials):
                        if task == 'passkey':
                            s, sp = b.passkey(L, depth, rng)
                        else:
                            s, sp = b.kv(L, depth, rng, K=args.kv_pairs, Q=args.kv_queries)
                        seqs.append(s)
                    X = torch.tensor(seqs, dtype=torch.long)
                    for name, m in cells:
                        cfg = EvictCfg(block_len=args.block_len, budget=m or 10**9,
                                       ring_window=32, score=args.score)
                        em, nc, ns, nl = [], 0, 0, 0.0
                        for i in range(0, len(seqs), args.batch):
                            x = X[i:i + args.batch].to(device)
                            e, c, n, l = score_batch(model, x, sp, cfg, m is not None, device)
                            em += e; nc += sum(c); ns += sum(n); nl += sum(l)
                        key = f'{task}/L{L}/d{depth}/{name}'
                        entry['cells'][key] = {
                            'exact': float(np.mean(em)), 'tok_acc': nc / ns,
                            'ans_nll': nl / ns, 'n': len(seqs), 'exact_per_trial': em}
                        print(f'{Path(ckpt).parent.name[:28]:28s} {key:36s} '
                              f'exact {np.mean(em):.3f} tok_acc {nc/ns:.3f} '
                              f'nll {nl/ns:.3f}', flush=True)
        results['ckpts'][ckpt] = entry
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(results, indent=1))
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
