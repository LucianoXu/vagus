# X | Y items: one document of the store cut into a context and a
# continuation that depends on it. Same-document text is the first
# source of "Y needs X" — entities, topic, style all carry over — and
# it comes for free from the store's document index.
#
# Layout inside a document of the vagus-tokens-v1 store (BOS-prefixed,
# no EOS): the BOS is dropped (the Generator supplies the start token),
# Y is the y_len tokens that follow the first x_max + gap tokens, and
# the context at length L <= x_max is the L tokens ending x_max. So Y is
# the same text whatever L is — the context lengths are paired on the
# same continuation, and the "no memory" score is shared — and a
# shorter context is a suffix of a longer one, as in a sliding window.
#
# gap: tokens skipped between X and Y. With gap 0 most of what the
# context buys is the local continuation — finishing the sentence and
# paragraph X ends in (the probe on LAX1: +0.9 nat on Y's first 64
# tokens, +0.06 on tokens 256-512). A gap removes that local term and
# leaves what a memory of the document is worth on its own: entities,
# topic, style.

from dataclasses import dataclass

import numpy as np
import torch

from ..dataset.loader import TokenStore


@dataclass(frozen=True)
class Item:
    shard: int
    doc: int
    x_max: int
    y_len: int
    gap: int = 0


def sample_items(store: TokenStore, rng: np.random.Generator, n: int, x_max: int, y_len: int,
                 gap: int = 0, max_tries: int = 100_000) -> list[Item]:
    '''n documents of at least 1 + x_max + gap + y_len tokens, drawn shard-
    weighted by token count and uniformly among the shard's documents
    (rejection on length; long documents are a few percent of
    FineWeb-Edu at 2-5k tokens). Pure function of rng.'''
    need = 1 + x_max + gap + y_len
    weights = np.asarray(store.shard_tokens, dtype=np.float64)
    weights /= weights.sum()
    out: list[Item] = []
    tries = 0
    while len(out) < n:
        tries += 1
        if tries > max_tries:
            raise RuntimeError(f'found {len(out)}/{n} documents of >= {need} tokens in {max_tries} draws')
        s = int(rng.choice(len(weights), p=weights))
        off = store.doc_offsets(s)
        i = int(rng.integers(0, len(off) - 1))
        if int(off[i + 1] - off[i]) >= need:
            out.append(Item(s, i, x_max, y_len, gap))
    return out


def item_ids(store: TokenStore, item: Item, x_len: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    '''(x, y) int64 tensors of shape (x_len,) and (y_len,); x_len
    defaults to x_max. The document's own BOS is not part of x.'''
    L = item.x_max if x_len is None else int(x_len)
    assert 0 <= L <= item.x_max
    d = np.asarray(store.doc(item.shard, item.doc)).astype(np.int64)
    end_x = 1 + item.x_max
    x = torch.from_numpy(d[end_x - L:end_x].copy())
    start_y = end_x + item.gap
    y = torch.from_numpy(d[start_y:start_y + item.y_len].copy())
    assert len(y) == item.y_len
    return x, y
