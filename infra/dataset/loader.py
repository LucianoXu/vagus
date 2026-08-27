# Training-side reading of vagus-tokens-v1 datasets.
#
# Two layers, split on purpose:
#   TokenStore   — what tokens exist: manifest-backed, memmapped, addressable
#                  by flat position and by document. Immutable.
#   WindowLoader — what the model sees when: one sampling policy (uniform
#                  non-overlapping windows, reshuffled per epoch, sharded by
#                  rank, exactly resumable). Doc-aware policies for streaming
#                  block training are meant to become siblings of this class,
#                  built on TokenStore's doc addressing.

import json
from pathlib import Path

import numpy as np
import torch


class TokenStore:
    '''
    Read-only view of a prepared dataset directory. `shards` selects a
    subset by shard file name (e.g. to hold out validation shards); the
    default is every shard in the manifest, in source order.
    '''

    def __init__(self, manifest_dir: str | Path, shards: list[str] | None = None):
        self.dir = Path(manifest_dir)
        self.manifest = json.loads((self.dir / 'manifest.json').read_text())
        assert self.manifest['format'] == 'vagus-tokens-v1'
        assert self.manifest['dtype'] == 'uint16'

        entries = sorted(self.manifest['shards'], key=lambda s: s['source'])
        if shards is not None:
            by_name = {e['file']: e for e in entries}
            entries = [by_name[name] for name in shards]
        self.entries = entries

        self.tokens: list[np.ndarray] = [
            np.load(self.dir / e['file'], mmap_mode='r') for e in entries]
        for arr, e in zip(self.tokens, entries):
            assert arr.dtype == np.uint16 and len(arr) == e['tokens']

        self.shard_tokens = [len(arr) for arr in self.tokens]
        self.total_tokens = sum(self.shard_tokens)
        self.vocab_size = self.manifest['tokenizer']['vocab_size']

        self._idx: list[np.ndarray | None] = [None] * len(entries)

    # document addressing (for doc-aware policies and inspection)

    def doc_offsets(self, shard: int) -> np.ndarray:
        '''uint64 doc start offsets of one shard, docs+1 entries. Lazy: the
        window policy never touches these.'''
        idx = self._idx[shard]
        if idx is None:
            idx = np.load(self.dir / self.entries[shard]['idx'], mmap_mode='r')
            self._idx[shard] = idx
        return idx

    def doc(self, shard: int, i: int) -> np.ndarray:
        off = self.doc_offsets(shard)
        return self.tokens[shard][int(off[i]):int(off[i + 1])]


class WindowLoader:
    '''
    Uniform non-overlapping windows of context_len+1 tokens (input/label
    shift happens here), reshuffled each epoch, rank-sharded, batched.
    Windows never cross shard boundaries; each shard's tail remainder is
    dropped (< context_len+1 tokens per shard, negligible).

    Determinism: the epoch-e order is a pure function of (seed, e), ranks
    take interleaved slices of it, so every batch is a pure function of
    (seed, epoch, rank, batch index). state_dict()/load_state_dict()
    resume mid-epoch exactly; the state carries a fingerprint of the
    order-defining parameters and refuses a mismatched resume, since that
    would silently change which tokens are seen.

    Iteration is infinite (epochs advance automatically); the training
    loop owns the stop condition, in steps or tokens.
    '''

    def __init__(
            self,
            store: TokenStore,
            context_len: int,
            batch_size: int,
            *,
            seed: int = 0,
            rank: int = 0,
            world_size: int = 1,
            shuffle: bool = True,
        ):
        assert context_len > 0 and batch_size > 0
        assert 0 <= rank < world_size

        self.store = store
        self.context_len = context_len
        self.batch_size = batch_size
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.shuffle = shuffle

        # window k of shard s starts at k * context_len and spans
        # context_len + 1 tokens (the +1 provides the shifted label)
        per_shard = [(n - 1) // context_len for n in store.shard_tokens]
        self._starts = np.concatenate([
            base + np.arange(w, dtype=np.int64) * context_len
            for w, base in zip(per_shard, np.cumsum([0] + store.shard_tokens[:-1]))
        ]) if per_shard else np.zeros(0, dtype=np.int64)
        # flat position -> (shard, local) mapping bounds
        self._shard_base = np.cumsum([0] + store.shard_tokens)

        self.window_count = len(self._starts)
        self.batches_per_epoch = self.window_count // (world_size * batch_size)
        assert self.batches_per_epoch > 0, "fewer windows than one global batch"
        self.tokens_per_batch = batch_size * context_len   # per rank

        self.epoch = 0
        self.batch_in_epoch = 0

    def _epoch_order(self, epoch: int) -> np.ndarray:
        if not self.shuffle:
            return self._starts
        perm = np.random.default_rng((self.seed, epoch)).permutation(self.window_count)
        return self._starts[perm]

    def _fingerprint(self) -> tuple:
        return (self.window_count, self.context_len, self.batch_size,
                self.seed, self.world_size, self.shuffle)

    def state_dict(self) -> dict:
        return {'epoch': self.epoch, 'batch_in_epoch': self.batch_in_epoch,
                'fingerprint': self._fingerprint()}

    def load_state_dict(self, state: dict):
        if tuple(state['fingerprint']) != self._fingerprint():
            raise ValueError("loader state was produced under a different "
                             "data order (fingerprint mismatch)")
        self.epoch = state['epoch']
        self.batch_in_epoch = state['batch_in_epoch']

    def _gather(self, flat_starts: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        L = self.context_len
        out = np.empty((len(flat_starts), L + 1), dtype=np.int64)
        for row, flat in enumerate(flat_starts):
            s = int(np.searchsorted(self._shard_base, flat, side='right')) - 1
            local = int(flat - self._shard_base[s])
            out[row] = self.store.tokens[s][local:local + L + 1]
        batch = torch.from_numpy(out)
        return batch[:, :-1], batch[:, 1:]

    def __iter__(self):
        while True:
            order = self._epoch_order(self.epoch)
            B, W, R = self.batch_size, self.world_size, self.rank
            while self.batch_in_epoch < self.batches_per_epoch:
                g = self.batch_in_epoch * W + R   # interleaved rank slices
                self.batch_in_epoch += 1
                yield self._gather(order[g * B:(g + 1) * B])
            self.epoch += 1
            self.batch_in_epoch = 0
