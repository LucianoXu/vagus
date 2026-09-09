# Teacher-forced statistics on windows of real text: one scoring pass,
# many metrics. The windows are sampled by seed from the store (see the
# package header on why no reserved held-out data), read as L tokens and
# scored as the stream start_id + window[:-1] — exactly L stream
# positions, so a model's trained context length is scored at that
# length. The prediction of window[0] from the start token alone is
# dropped; the remaining L-1 positions see the same left context
# training used, plus the start token at the front.
#
# Metrics are a second registry, LOGIT_METRICS: fn(logp, targets, args)
# -> {name: (B,) tensor}, with logp the fp32 log-probabilities of one
# batch. Adding a statistic over the same logits is one function here.
#
#   nll            mean -log p(target); ppl = exp(nll) as a scalar
#   entropy        mean predictive entropy (nats)
#   top_p_support  mean size of the nucleus at `top_p` — the number of
#                  tokens sampling would actually choose among
#   rep_at_l       Welleck et al. 2019: fraction of positions whose
#                  argmax prediction occurs in the previous l gold tokens
#                  (rep_l), the same for the gold token itself
#                  (gold_rep_l, the text's own repetition), and the
#                  excess rep_l - gold_rep_l — a deterministic,
#                  sampling-free measure of the model's pull towards
#                  repeating its context.
#
# Several context lengths share the same window starts (nested
# prefixes), so a longer length scores the same text plus more; lengths a
# subject cannot stream (max_stream_len) are skipped and listed in the
# witness — this is how a NoPE model's extrapolation is read off.

import math
from collections import defaultdict
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

from ..core import EvalCtx, TaskResult, exposure


def metric_nll(logp, targets, args):
    return {'nll': -logp.gather(-1, targets[..., None]).squeeze(-1).mean(1)}


def metric_entropy(logp, targets, args):
    return {'entropy': -(logp.exp() * logp).sum(-1).mean(1)}


def metric_top_p_support(logp, targets, args):
    p = logp.exp().sort(dim=-1, descending=True).values
    cum = p.cumsum(-1)
    # same rule as sampling.sample: a token is kept while the mass before it is < top_p
    kept = ((cum - p) < args['top_p']).sum(-1)
    return {'top_p_support': kept.float().mean(1)}


def metric_rep_at_l(logp, targets, args):
    pred = logp.argmax(-1)
    out = {}
    for l in args['rep_l']:
        # the l gold tokens before each position (padded with -1 at the front)
        prev = F.pad(targets, (l, 0), value=-1).unfold(1, l, 1)[:, :-1]
        rep = (prev == pred[..., None]).any(-1).float().mean(1)
        gold = (prev == targets[..., None]).any(-1).float().mean(1)
        out[f'rep_{l}'] = rep
        out[f'gold_rep_{l}'] = gold
        out[f'rep_excess_{l}'] = rep - gold
    return out


LOGIT_METRICS = {
    'nll': metric_nll,
    'entropy': metric_entropy,
    'top_p_support': metric_top_p_support,
    'rep_at_l': metric_rep_at_l,
}


def run(ctx: EvalCtx, data_dir: str, shards: list[str] | None = None, n_seq: int = 128,
        context_len: int | list[int] = 2048, batch_size: int = 4,
        metrics: Sequence[str] = ('nll', 'entropy', 'top_p_support', 'rep_at_l'),
        top_p: float = 0.9, rep_l: Sequence[int] = (32, 128)) -> TaskResult:
    store = ctx.store(data_dir, shards)
    lengths = sorted({int(x) for x in ([context_len] if isinstance(context_len, int) else context_len)})
    L_max = lengths[-1]
    rng = ctx.rng('windows')
    weights = np.asarray(store.shard_tokens, dtype=np.float64)
    weights /= weights.sum()
    windows = []
    for _ in range(n_seq):
        s = int(rng.choice(len(weights), p=weights))
        windows.append((s, int(rng.integers(0, store.shard_tokens[s] - L_max))))

    gen = ctx.subject.generator
    limit = gen.model.max_stream_len
    args = {'top_p': top_p, 'rep_l': tuple(int(l) for l in rep_l)}
    result = TaskResult()
    ran, skipped = [], []
    for L in lengths:
        if limit is not None and L > limit:
            skipped.append(L)
            continue
        per: dict[str, list[float]] = defaultdict(list)
        for b0 in range(0, n_seq, batch_size):
            batch = windows[b0:b0 + batch_size]
            ids = torch.tensor(np.stack([store.read_window(s, st, L).astype(np.int64)
                                         for s, st in batch]))
            logits = gen.score_ids(ids)[:, 1:]          # drop the start-token-only prediction
            targets = ids[:, 1:].to(logits.device)
            logp = logits.float().log_softmax(-1)
            del logits
            for name in metrics:
                for k, v in LOGIT_METRICS[name](logp, targets, args).items():
                    per[k].extend(v.float().cpu().tolist())
            del logp
        for k, v in per.items():
            result.items[f'{k}@{L}'] = v
            result.scalars[f'{k}@{L}'] = float(np.mean(v))
        if 'nll' in per:
            result.scalars[f'ppl@{L}'] = math.exp(result.scalars[f'nll@{L}'])
        ran.append(L)
        ctx.log(f'  L={L}: ' + ' '.join(f'{k}={result.scalars[f"{k}@{L}"]:.4f}' for k in per))

    result.witness = {
        'n_seq': n_seq,
        'lengths': ran,
        'lengths_skipped': skipped,
        'metrics': list(metrics),
        'top_p': top_p,
        'rep_l': list(args['rep_l']),
        'store': {'dir': str(store.dir.resolve()), 'source': store.manifest.get('source'),
                  'tokenizer': store.manifest.get('tokenizer', {}).get('id'),
                  'shards': [e['file'] for e in store.entries], 'total_tokens': store.total_tokens},
        'exposure': exposure(ctx.subject, store),
        'windows': [[store.entries[s]['file'], st] for s, st in windows],
    }
    return result
