# Repetition in free-running generation. Prompts are document openings
# drawn by seed from the store; each is continued greedily and with
# `seeds` sampled runs, and the document's own continuation at the same
# positions gives the paired real-text floor — the level of n-gram
# repetition natural text has at this length.
#
#   rep{n}   fraction of duplicate n-grams among the n-grams of one
#            continuation (seq-rep-n, Welleck et al. 2019; = 1 -
#            distinct-n). It grows with length and depends on the
#            sampling settings, so both are pinned by the args and
#            written to the witness; compare only at equal settings, and
#            read the absolute value against rep{n}_floor.
#
# Items: rep{n}_greedy and rep{n}_floor per prompt, rep{n}_sampled per
# (seed, prompt) in seed-major order. Samples: every generated
# continuation and the reference, as text, for reading.

from typing import Sequence

import numpy as np
import torch

from ...inference import SamplingConfig
from ..core import EvalCtx, TaskResult, exposure


def rep_n(ids, n: int) -> float:
    grams = [tuple(ids[i:i + n]) for i in range(len(ids) - n + 1)]
    return 1.0 - len(set(grams)) / max(len(grams), 1)


def run(ctx: EvalCtx, data_dir: str, shards: list[str] | None = None, n_prompts: int = 32,
        prompt_tokens: int = 32, new_tokens: int = 96, seeds: int = 5,
        sampling: dict | None = None, greedy: bool = True, rep_ns: Sequence[int] = (4,),
        batch_size: int = 32) -> TaskResult:
    sampling = {'temperature': 0.8, 'top_p': 0.9} | (sampling or {})
    store = ctx.store(data_dir, shards)
    rng = ctx.rng('prompts')
    n_docs = [len(store.doc_offsets(s)) - 1 for s in range(len(store.entries))]
    weights = np.asarray(n_docs, dtype=np.float64)
    weights /= weights.sum()
    need = 1 + prompt_tokens + new_tokens        # doc BOS + prompt + reference continuation
    prompts, refs, picked = [], [], []
    while len(prompts) < n_prompts:
        s = int(rng.choice(len(weights), p=weights))
        i = int(rng.integers(0, n_docs[s]))
        doc = store.doc(s, i)
        if len(doc) < need:
            continue
        prompts.append(doc[1:1 + prompt_tokens].astype(np.int64))   # the stream adds start_id
        refs.append(doc[1 + prompt_tokens:need].astype(np.int64))
        picked.append([store.entries[s]['file'], i])
    P = torch.tensor(np.stack(prompts))
    R = np.stack(refs)

    gen = ctx.subject.generator
    max_len = prompt_tokens + new_tokens + 1

    def generate(cfg: SamplingConfig) -> np.ndarray:
        outs = [gen.generate_ids(P[b:b + batch_size], cfg, max_len=max_len).cpu()
                for b in range(0, n_prompts, batch_size)]
        return torch.cat(outs).numpy()

    ns = tuple(int(n) for n in rep_ns)
    result = TaskResult()
    prompt_text = gen.decode(P)

    def record(kind: str, seed: int | None, cont: np.ndarray):
        texts = gen.decode(torch.tensor(cont))
        for p in range(n_prompts):
            row = {'kind': kind, 'seed': seed, 'prompt_id': p, 'prompt': prompt_text[p], 'text': texts[p]}
            for n in ns:
                r = rep_n(cont[p].tolist(), n)
                result.items.setdefault(f'rep{n}_{kind}', []).append(r)
                row[f'rep{n}'] = r
            result.samples.append(row)

    record('floor', None, R)
    if greedy:
        record('greedy', None, generate(SamplingConfig(max_new_tokens=new_tokens, temperature=0, stop_ids=())))
    for s in range(seeds):
        cfg = SamplingConfig(max_new_tokens=new_tokens, stop_ids=(), seed=ctx.torch_seed(f'sample{s}'), **sampling)
        record('sampled', s, generate(cfg))

    for k, v in result.items.items():
        result.scalars[k] = float(np.mean(v))
    for n in ns:
        if f'rep{n}_sampled' in result.scalars and result.scalars[f'rep{n}_floor'] > 0:
            result.scalars[f'rep{n}_sampled_over_floor'] = (
                result.scalars[f'rep{n}_sampled'] / result.scalars[f'rep{n}_floor'])
    ctx.log('  ' + ' '.join(f'{k}={v:.4f}' for k, v in result.scalars.items()))

    result.witness = {
        'n_prompts': n_prompts, 'prompt_tokens': prompt_tokens, 'new_tokens': new_tokens,
        'seeds': seeds, 'sampling': sampling, 'greedy': greedy, 'rep_ns': list(ns),
        'store': {'dir': str(store.dir.resolve()), 'source': store.manifest.get('source'),
                  'shards': [e['file'] for e in store.entries]},
        'exposure': exposure(ctx.subject, store),
        'prompts': picked,                           # [shard file, doc index]
    }
    return result
