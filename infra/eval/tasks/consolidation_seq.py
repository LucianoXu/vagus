# Sequential consolidation — the continual-learning reading of the
# unit test. Documents X_1..X_n are consolidated one after another
# into the same weights (nothing restored in between); after each
# consolidation k every earlier continuation Y_j (j <= k) is scored
# with a blank memory. nll2[k][j] against nll1[j] (memory intact,
# original weights) and nll3[j] (no memory, original weights) gives
# the fraction of document j's context benefit still in the weights
# after k - j further consolidations — the forgetting curve by age.
#
# docs_per_sleep groups the documents: one sleep consolidates a group
# jointly (their replays pooled, the per-document step budget kept),
# the scores are taken after each group. 1 is strictly sequential; n
# is one sleep over everything.
#
# Scalars: kept_age@a = ratio of means over all scored (j, k) with
# k - j = a of (nll3_j - nll2_j^(k)) / (nll3_j - nll1_j); kept_final
# over all documents after the last sleep; retain_delta_final: LM loss
# on one fixed set of store windows, after the last sleep minus before
# the first (the per-sleep witnesses draw their own windows and are
# not comparable across sleeps). Items: kept_final per document
# (paired across subjects / methods on the same seed), retain_lm after
# each sleep. Witness: the full nll2 matrix, the per-sleep witnesses.

from collections import defaultdict
from typing import Sequence

import numpy as np
import torch

from ...consolidate import (SleepConfig, Sleeper, item_ids, nll_positions, sample_items,
                            score_continuation)
from ..core import EvalCtx, TaskResult, exposure


def run(ctx: EvalCtx, data_dir: str, shards: list[str] | None = None, n_docs: int = 16,
        x_len: int = 2048, y_len: int = 512, gap: int = 256, batch_size: int = 2,
        method: str = 'replay_kl', sleep: dict | None = None, docs_per_sleep: int = 1,
        ages: Sequence[int] = (0, 1, 2, 4, 8, 16, 32)) -> TaskResult:
    store = ctx.store(data_dir, shards)
    items = sample_items(store, ctx.rng('items'), n_docs, x_len, y_len, gap)
    gen = ctx.subject.generator
    limit = gen.model.max_stream_len
    assert limit is None or 1 + x_len + gap + y_len <= limit, f'{x_len} + {gap} + {y_len} exceeds the stream limit {limit}'

    # reference scores under the original weights
    nll1, nll3 = [], []
    for b0 in range(0, n_docs, batch_size):
        batch = items[b0:b0 + batch_size]
        y = torch.stack([item_ids(store, it)[1] for it in batch])
        x = torch.stack([item_ids(store, it)[0] for it in batch])
        nll3.extend(nll_positions(score_continuation(gen, y), y).mean(1).tolist())
        nll1.extend(nll_positions(score_continuation(gen, y, x), y).mean(1).tolist())
    benefit = np.array(nll3) - np.array(nll1)
    ctx.log(f'  nll3={np.mean(nll3):.4f} nll1={np.mean(nll1):.4f} benefit={benefit.mean():.4f}')

    cfg = SleepConfig(method=method, **(sleep or {}))
    sleeper = Sleeper(gen, store, cfg, log=ctx.log)
    model = gen.model
    snapshot = {k: v.detach().clone() for k, v in model.state_dict().items()}   # type: ignore[attr-defined]
    retain = sleeper._windows(max(cfg.retain_windows, 1), cfg.lm_len)
    with torch.no_grad():
        retain_lm = [float(sleeper._lm_loss(retain))]
    nll2 = np.full((n_docs, n_docs), np.nan)                                     # [k][j], j <= k
    witnesses = []
    for g0 in range(0, n_docs, docs_per_sleep):
        ks = list(range(g0, min(g0 + docs_per_sleep, n_docs)))
        k = ks[-1]
        w = sleeper.consolidate([item_ids(store, items[i])[0] for i in ks])
        witnesses.append({kk: v for kk, v in w.items() if kk != 'config'} | {'docs': ks})
        with torch.no_grad():
            retain_lm.append(float(sleeper._lm_loss(retain)))
        for b0 in range(0, k + 1, batch_size):
            js = list(range(b0, min(b0 + batch_size, k + 1)))
            y = torch.stack([item_ids(store, items[j])[1] for j in js])
            nll2[k, js] = nll_positions(score_continuation(gen, y), y).mean(1).cpu().numpy()
        kept_now = sum(nll3[i] - nll2[k, i] for i in ks) / max(sum(benefit[i] for i in ks), 1e-9)
        kept_first = (nll3[0] - nll2[k, 0]) / benefit[0] if abs(benefit[0]) > 1e-9 else float('nan')
        ctx.log(f'  docs {ks[0]}-{k}: kept_now={kept_now:.3f} kept_doc0={kept_first:.3f} '
                f'retain_lm {retain_lm[0]:.4f}->{retain_lm[-1]:.4f}')
    model.load_state_dict(snapshot)                                              # type: ignore[attr-defined]
    del snapshot

    result = TaskResult()
    by_age: dict[int, list[tuple[float, float]]] = defaultdict(list)           # age -> [(nll3 - nll2, benefit)]
    for k in range(n_docs):
        for j in range(k + 1):
            if not np.isnan(nll2[k, j]):
                by_age[k - j].append((nll3[j] - nll2[k, j], benefit[j]))
    for a in sorted(by_age):
        num = sum(v for v, _ in by_age[a])
        den = sum(b for _, b in by_age[a])
        result.scalars[f'kept_age@{a}'] = num / den if abs(den) > 1e-9 else float('nan')
        result.scalars[f'n_pairs@{a}'] = len(by_age[a])
    final = nll2[n_docs - 1]
    result.items['kept_final'] = [(nll3[j] - final[j]) / benefit[j] if abs(benefit[j]) > 1e-9 else float('nan')
                                  for j in range(n_docs)]
    result.items['lost_final'] = [final[j] - nll1[j] for j in range(n_docs)]
    result.items['nll1'], result.items['nll3'] = list(nll1), list(nll3)
    result.items['benefit'] = benefit.tolist()
    result.scalars['kept_final'] = float((np.array(nll3) - final).sum() / benefit.sum())
    result.scalars['ratio_final'] = 1.0 - result.scalars['kept_final']
    result.scalars['benefit'] = float(benefit.mean())
    result.scalars['retain_delta_final'] = retain_lm[-1] - retain_lm[0]
    result.items['retain_lm'] = retain_lm
    ctx.log('  ' + ' '.join(f'kept_age@{a}={result.scalars[f"kept_age@{a}"]:.3f}' for a in sorted(by_age)
                            if a in set(int(v) for v in ages) or a == n_docs - 1)
            + f' kept_final={result.scalars["kept_final"]:.3f}'
            + (f' retainΔ={result.scalars["retain_delta_final"]:+.4f}' if 'retain_delta_final' in result.scalars else ''))
    result.witness = {
        'n_docs': n_docs, 'x_len': x_len, 'y_len': y_len, 'gap': gap, 'method': method, 'sleep': cfg.asdict(),
        'docs_per_sleep': docs_per_sleep,
        'items': [[store.entries[it.shard]['file'], it.doc] for it in items],
        'nll2': [[None if np.isnan(v) else round(float(v), 6) for v in row] for row in nll2],
        'consolidations': witnesses,
        'store': {'dir': str(store.dir.resolve()), 'source': store.manifest.get('source'),
                  'tokenizer': store.manifest.get('tokenizer', {}).get('id'),
                  'shards': [e['file'] for e in store.entries], 'total_tokens': store.total_tokens},
        'exposure': exposure(ctx.subject, store),
    }
    return result
