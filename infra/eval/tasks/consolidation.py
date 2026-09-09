# The consolidation unit test (see infra/consolidate): documents cut
# into X | Y, Y scored with the memory of X intact (nll1), with no
# memory (nll3) and — when a method is set — after consolidating X into
# the weights and clearing the memory (nll2).
#
# Cells are context lengths x_len (Y is the same text for all of them,
# a shorter X the suffix of a longer one; items.py — and `gap` tokens
# may separate X from Y, see there). Per item and
# length: nll1@L, benefit@L = nll3 - nll1@L, the position buckets of
# both, and with a method nll2@L, lost@L = nll2 - nll1 and the loss
# ratio lost / benefit. The ratio's scalar is the ratio of means (a
# per-item ratio explodes when an item's benefit is ~0).
#
# A consolidation trains the subject's model in place, one item at a
# time, on the lengths in sleep_x_len (default: all of x_len); the
# weights are snapshotted before and restored after, so items are
# independent and later tasks see the original subject. retain_* in
# the witness is the method's own before/after LM loss on fixed store
# windows — the forgetting side of the same procedure.
#
# Lengths a subject cannot stream (max_stream_len < 1 + L + y_len) are
# skipped and listed.

from collections import defaultdict
from typing import Sequence

import numpy as np
import torch

from ...consolidate import (SleepConfig, Sleeper, bucket_means, item_ids, nll_positions,
                            sample_items, score_continuation)
from ..core import EvalCtx, TaskResult, exposure


def run(ctx: EvalCtx, data_dir: str, shards: list[str] | None = None, n_items: int = 32,
        x_len: int | Sequence[int] = (512, 1024, 2048, 4096), y_len: int = 512, gap: int = 0,
        batch_size: int = 2, buckets: Sequence[int] = (64, 256),
        method: str | None = None, sleep: dict | None = None,
        sleep_x_len: Sequence[int] | None = None, n_text_samples: int = 2) -> TaskResult:
    store = ctx.store(data_dir, shards)
    lengths = sorted({int(L) for L in ([x_len] if isinstance(x_len, int) else x_len)})
    x_max = lengths[-1]
    items = sample_items(store, ctx.rng('items'), n_items, x_max, y_len, gap)
    gen = ctx.subject.generator
    limit = gen.model.max_stream_len
    edges = tuple(int(e) for e in buckets)

    def can(L: int) -> bool:
        return limit is None or 1 + L + gap + y_len <= limit

    ran = [L for L in lengths if can(L)]
    skipped = [L for L in lengths if not can(L)]
    per: dict[str, list[float]] = defaultdict(list)
    result = TaskResult()

    # --- nll3 and nll1@L, batched ---------------------------------------
    for b0 in range(0, n_items, batch_size):
        batch = items[b0:b0 + batch_size]
        y = torch.stack([item_ids(store, it)[1] for it in batch])
        nll3 = nll_positions(score_continuation(gen, y), y)
        per['nll3'].extend(nll3.mean(1).tolist())
        for k, v in bucket_means(nll3, edges).items():
            per[f'nll3_{k}'].extend(v.tolist())
        for L in ran:
            x = torch.stack([item_ids(store, it, L)[0] for it in batch])
            nll1 = nll_positions(score_continuation(gen, y, x), y)
            per[f'nll1@{L}'].extend(nll1.mean(1).tolist())
            per[f'benefit@{L}'].extend((nll3.mean(1) - nll1.mean(1)).tolist())
            for k, v in bucket_means(nll1, edges).items():
                per[f'nll1_{k}@{L}'].extend(v.tolist())
            del nll1
        del nll3
    ctx.log('  ' + ' '.join(f'nll3={np.mean(per["nll3"]):.4f}'.split()) + ' ' + ' '.join(
        f'benefit@{L}={np.mean(per[f"benefit@{L}"]):.4f}' for L in ran))

    # --- nll2@L: consolidate, clear, score ---------------------------------
    witnesses: dict[str, list[dict]] = {}
    if method:
        cfg = SleepConfig(method=method, **(sleep or {}))
        sleeper = Sleeper(gen, store, cfg, log=ctx.log)
        sleep_lengths = [int(L) for L in (sleep_x_len or ran)]
        assert all(L in ran for L in sleep_lengths), f'sleep_x_len {sleep_lengths} not all in {ran}'
        model = gen.model
        snapshot = {k: v.detach().clone() for k, v in model.state_dict().items()}   # type: ignore[attr-defined]
        for L in sleep_lengths:
            witnesses[str(L)] = []
            for i, it in enumerate(items):
                x, y = item_ids(store, it, L)
                w = sleeper.consolidate(x)
                nll2 = nll_positions(score_continuation(gen, y[None]), y[None])
                model.load_state_dict(snapshot)                                      # type: ignore[attr-defined]
                per[f'nll2@{L}'].append(float(nll2.mean()))
                per[f'lost@{L}'].append(float(nll2.mean()) - per[f'nll1@{L}'][i])
                for k, v in bucket_means(nll2, edges).items():
                    per[f'nll2_{k}@{L}'].append(float(v[0]))
                if 'retain_before' in w:
                    per[f'retain_delta@{L}'].append(w['retain_after'] - w['retain_before'])
                if 'kl_before' in w:
                    per[f'kl_before@{L}'].append(w['kl_before'])
                    per[f'kl_after@{L}'].append(w['kl_after'])
                witnesses[str(L)].append({k: v for k, v in w.items() if k != 'config'})
                if i < n_text_samples and method == 'replay_kl':
                    m = sleeper.read(x)
                    s, _ = sleeper.replay(m)
                    result.samples.append({'item': i, 'x_len': L, 'context_tail': gen.decode(x[None, -64:])[0],
                                           'replay': sleeper.decode_replay(s, 2)})
                    del m, s
                ctx.log(f'  L={L} item {i}: nll1={per[f"nll1@{L}"][i]:.4f} nll2={float(nll2.mean()):.4f} '
                        f'nll3={per["nll3"][i]:.4f}' + (f' kl {w["kl_before"]:.3f}->{w["kl_after"]:.3f}'
                                                         if 'kl_before' in w else '') +
                        (f' retain {w["retain_before"]:.3f}->{w["retain_after"]:.3f}' if 'retain_before' in w else ''))
        del snapshot
        result.witness['sleep'] = cfg.asdict()
        result.witness['sleep_x_len'] = sleep_lengths

    for k, v in per.items():
        result.items[k] = v
        result.scalars[k] = float(np.mean(v))
    for L in ran:
        if f'lost@{L}' in per:
            b = result.scalars[f'benefit@{L}']
            result.scalars[f'ratio@{L}'] = result.scalars[f'lost@{L}'] / b if abs(b) > 1e-9 else float('nan')
    if method:
        ctx.log('  ' + ' '.join(f'ratio@{L}={result.scalars[f"ratio@{L}"]:.3f}' for L in ran if f'ratio@{L}' in result.scalars))

    result.witness.update({
        'n_items': n_items, 'x_len': ran, 'x_len_skipped': skipped, 'y_len': y_len, 'gap': gap, 'buckets': list(edges),
        'method': method,
        'items': [[store.entries[it.shard]['file'], it.doc] for it in items],
        'store': {'dir': str(store.dir.resolve()), 'source': store.manifest.get('source'),
                  'tokenizer': store.manifest.get('tokenizer', {}).get('id'),
                  'shards': [e['file'] for e in store.entries], 'total_tokens': store.total_tokens},
        'exposure': exposure(ctx.subject, store),
        'consolidations': witnesses,
    })
    return result
