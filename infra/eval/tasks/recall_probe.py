# Associative recall, MQAR-style (Arora et al., Zoology): key/value
# pairs scattered through a context of length L, every key queried at
# the end, the value token scored teacher-forced. No sampling, so the
# probe is deterministic; L and the number of pairs are the two axes
# (cells), and a cell beyond a subject's max_stream_len is skipped and
# listed — the NoPE / linear models run the long cells, a RoPE softmax
# model at its context length does not.
#
# Layout of one sequence (all ids, no re-tokenisation):
#
#   body (L - 4k tokens): noise, with k slots of  key : value \n
#         overwritten at random 4-aligned positions
#   tail (4k tokens):     key : value \n  for every pair in random order
#
# The value position of each tail block is scored: acc = argmax hits the
# value, logprob = log p(value), copy = argmax is *some* in-context value
# (the model is copying, right or wrong; chance for acc given copying is
# 1/k). Keys and values are word-like tokens (a leading word marker,
# alphabetic), disjoint within a sequence; separators are the
# tokenizer's ':' and newline pieces. Noise is real text from the store
# by default (the model stays in distribution; the pairs read as lines
# embedded in a document) or uniform draws from the word pool.
#
# Zero-shot on a pretrained LM this is a copying test, not the trained
# MQAR task; the numbers are comparable across subjects sharing a
# tokenizer, not with the MQAR literature.

from typing import Sequence

import numpy as np
import torch

from ..core import EvalCtx, TaskResult, exposure


def word_pool(tokenizer, vocab_size: int, min_len: int = 5) -> list[int]:
    ids = [i for piece, i in tokenizer.get_vocab().items()
           if i < vocab_size and piece.startswith('▁') and len(piece) > min_len
           and piece[1:].isalpha() and piece[1:].islower()]
    return sorted(ids)


def run(ctx: EvalCtx, lengths: Sequence[int] = (256, 1024, 4096), kv_pairs: Sequence[int] = (8, 32),
        n: int = 64, noise: str = 'text', data_dir: str | None = None,
        shards: list[str] | None = None, batch_size: int = 4,
        pool: list[int] | None = None, seps: list[int] | None = None) -> TaskResult:
    assert noise in ('text', 'random')
    gen = ctx.subject.generator
    vocab_size = int(ctx.subject.meta['model_args']['vocab_size'])
    if pool is None:
        ids = word_pool(gen.tokenizer, vocab_size)
    else:
        lo, hi = pool
        ids = list(range(int(lo), int(hi)))
    if seps is None:
        vocab = gen.tokenizer.get_vocab()
        assert ':' in vocab and '<0x0A>' in vocab, 'tokenizer lacks the separator pieces; pass seps='
        colon, nl = vocab[':'], vocab['<0x0A>']
    else:
        colon, nl = (int(x) for x in seps)
    store = ctx.store(data_dir, shards) if noise == 'text' else None
    weights = None
    if store is not None:
        weights = np.asarray(store.shard_tokens, dtype=np.float64)
        weights /= weights.sum()

    rng = ctx.rng('sequences')
    limit = gen.model.max_stream_len
    off = 0 if gen.start_id is not None else 1      # score_ids drops position 0 without a start token
    result = TaskResult()
    ran, skipped = [], []
    for L in (int(x) for x in lengths):
        for k in (int(x) for x in kv_pairs):
            tail = 4 * k
            body = L - tail
            assert body >= 4 * k, f'L={L} cannot hold {k} pairs'
            cell = f'L{L}_p{k}'
            # draw the cell's sequences before the skip check: the RNG
            # stream, hence every later cell, is the same for every subject
            seqs, answers, values = [], [], []
            for _ in range(n):
                kv = rng.choice(ids, 2 * k, replace=False)
                ks, vs = kv[:k], kv[k:]
                if store is not None:
                    s = int(rng.choice(len(weights), p=weights))
                    st = int(rng.integers(0, store.shard_tokens[s] - body))
                    base = store.read_window(s, st, body).astype(np.int64)
                else:
                    base = rng.choice(ids, body).astype(np.int64)
                slots = np.sort(rng.choice(body // 4, k, replace=False)) * 4
                for j, pos in enumerate(slots):
                    base[pos:pos + 4] = (ks[j], colon, vs[j], nl)
                order = rng.permutation(k)
                tail_ids = np.concatenate([(ks[j], colon, vs[j], nl) for j in order]).astype(np.int64)
                seqs.append(np.concatenate([base, tail_ids]))
                answers.append(vs[order])
                values.append(vs)
            if limit is not None and L > limit:
                skipped.append(cell)
                continue
            pos = torch.tensor([body + 4 * q + 2 - off for q in range(k)])
            acc, lp, copy = [], [], []
            for b0 in range(0, n, batch_size):
                ids_b = torch.tensor(np.stack(seqs[b0:b0 + batch_size]))
                logits = gen.score_ids(ids_b)
                sel = logits[:, pos.to(logits.device)].float().log_softmax(-1)   # (B, k, V)
                del logits
                ans = torch.tensor(np.stack(answers[b0:b0 + batch_size])).to(sel.device)
                vals = torch.tensor(np.stack(values[b0:b0 + batch_size])).to(sel.device)
                pred = sel.argmax(-1)
                acc.extend((pred == ans).float().mean(1).cpu().tolist())
                lp.extend(sel.gather(-1, ans[..., None]).squeeze(-1).mean(1).cpu().tolist())
                copy.extend((pred[..., None] == vals[:, None, :]).any(-1).float().mean(1).cpu().tolist())
            result.items[f'acc_{cell}'] = acc
            result.items[f'logprob_{cell}'] = lp
            result.items[f'copy_{cell}'] = copy
            for name, v in (('acc', acc), ('logprob', lp), ('copy', copy)):
                result.scalars[f'{name}_{cell}'] = float(np.mean(v))
            ran.append(cell)
            ctx.log(f'  {cell}: acc={np.mean(acc):.3f} copy={np.mean(copy):.3f} logprob={np.mean(lp):.3f}')
    if ran:
        result.scalars['acc'] = float(np.mean([result.scalars[f'acc_{c}'] for c in ran]))

    result.witness = {
        'lengths': list(lengths), 'kv_pairs': list(kv_pairs), 'n': n, 'noise': noise,
        'cells': ran, 'cells_skipped': skipped,
        'pool_size': len(ids), 'seps': [colon, nl],
        'chance': {f'p{k}': 1.0 / int(k) for k in kv_pairs},
        'store': None if store is None else {
            'dir': str(store.dir.resolve()), 'source': store.manifest.get('source'),
            'shards': [e['file'] for e in store.entries]},
        'exposure': None if store is None else exposure(ctx.subject, store),
    }
    return result
