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
#
# I/O path: window bytes come from os.pread, not the mmap view. Random
# 4KB window reads on a store larger than the Lustre OST caches cost
# real seeks (measured 0.4-0.9s per micro-batch on the 140-shard
# FineWeb-Edu store — it throttled the sax1 pilots to 0.65x throughput),
# so WindowLoader prefetches on a background thread; pread releases the
# GIL during the disk wait, whereas a page fault inside numpy's copy
# would hold it and stall the training thread anyway. The mmap views
# stay for doc addressing and init-time validation.

import json
import os
import queue
import threading
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

        # pread path (read_window): raw fds + the npy header size, taken
        # from the memmap so the .npy format stays numpy's problem.
        # Opened eagerly: no lazy-open race with the prefetch thread.
        self._fds = [os.open(str(self.dir / e['file']), os.O_RDONLY)
                     for e in entries]
        self._data_off = [int(arr.offset) for arr in self.tokens]  # type: ignore[attr-defined]

        self._idx: list[np.ndarray | None] = [None] * len(entries)

    def read_window(self, shard: int, start: int, count: int) -> np.ndarray:
        '''`count` tokens at token offset `start` of one shard, read via
        pread — GIL-free during the disk wait (see module header).'''
        buf = os.pread(self._fds[shard], count * 2,
                       self._data_off[shard] + start * 2)
        assert len(buf) == count * 2, \
            f'short read: shard {shard} @ {start}+{count}'
        return np.frombuffer(buf, dtype=np.uint16)

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

    Prefetch: a daemon thread reads `prefetch` batches ahead (0 =
    synchronous), overlapping the window gather I/O with compute. The
    batch sequence is byte-identical either way (one reader running the
    same order), and epoch/batch_in_epoch always track the position the
    CONSUMER has reached — state_dict() taken between training steps
    checkpoints consumed batches, never the thread's read-ahead. Call
    load_state_dict before iter(), not during iteration.

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
            prefetch: int = 4,
        ):
        assert context_len > 0 and batch_size > 0
        assert 0 <= rank < world_size
        assert prefetch >= 0

        self.store = store
        self.context_len = context_len
        self.batch_size = batch_size
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.shuffle = shuffle
        self.prefetch = prefetch

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
            out[row] = self.store.read_window(s, local, L + 1)
        batch = torch.from_numpy(out)
        return batch[:, :-1], batch[:, 1:]

    def _produce(self, epoch: int, batch_in_epoch: int):
        '''Infinite (epoch, index, x, y) stream from a start position.
        Pure w.r.t. loader state — the consumer owns epoch/batch_in_epoch,
        so a read-ahead producer never corrupts what state_dict() saves.'''
        B, W, R = self.batch_size, self.world_size, self.rank
        while True:
            order = self._epoch_order(epoch)
            while batch_in_epoch < self.batches_per_epoch:
                g = batch_in_epoch * W + R   # interleaved rank slices
                x, y = self._gather(order[g * B:(g + 1) * B])
                yield epoch, batch_in_epoch, x, y
                batch_in_epoch += 1
            epoch += 1
            batch_in_epoch = 0

    def __iter__(self):
        src = self._produce(self.epoch, self.batch_in_epoch)
        if self.prefetch == 0:
            for epoch, b, x, y in src:
                # b+1 before the yield, matching the historical states a
                # checkpointed fingerprint may carry
                self.epoch, self.batch_in_epoch = epoch, b + 1
                yield x, y
            return

        q: queue.Queue = queue.Queue(maxsize=self.prefetch)
        stop = threading.Event()

        def reader():
            try:
                for item in src:
                    while not stop.is_set():
                        try:
                            q.put(item, timeout=1.0)
                            break
                        except queue.Full:
                            pass
                    else:
                        return
            except BaseException as e:   # surface in the consumer
                q.put(e)

        threading.Thread(target=reader, daemon=True,
                         name='WindowLoader-prefetch').start()
        try:
            while True:
                item = q.get()
                if isinstance(item, BaseException):
                    raise item
                epoch, b, x, y = item
                self.epoch, self.batch_in_epoch = epoch, b + 1
                yield x, y
        finally:
            stop.set()   # generator closed: let the reader drain and exit
