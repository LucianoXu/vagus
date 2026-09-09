# The nouns of an evaluation run, shared by the runner and the tasks.

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from ..dataset.loader import TokenStore
from ..inference import Generator


@dataclass
class Subject:
    '''One checkpoint under evaluation. `meta` is load_model's record of
    it (model_name/args, tokenizer, step, tokens_seen, format);
    `run_meta` the training run's meta.json when it sits beside the
    checkpoint (the witnessed data identity lives there).'''
    label: str
    path: Path
    sha256: str
    meta: dict
    generator: Generator
    run_meta: dict | None = None

    @property
    def model(self):
        return self.generator.model

    @property
    def param_count(self) -> int:
        return sum(p.numel() for p in self.model.parameters())  # type: ignore[attr-defined]

    def identity(self) -> dict:
        '''What meta.json records about the subject.'''
        return {
            'label': self.label,
            'path': str(self.path),
            'sha256': self.sha256,
            'checkpoint': {k: self.meta.get(k) for k in
                           ('model_name', 'model_args', 'tokenizer', 'step', 'tokens_seen', 'format')},
            'param_count': self.param_count,
            'max_stream_len': self.model.max_stream_len,
            'train_data': (self.run_meta or {}).get('data'),
        }


@dataclass
class TaskResult:
    '''scalars: the summary numbers. items: per-item arrays in an order
    that depends only on (seed, task args) — the runner pairs subjects on
    them. witness: what the task actually did (item ids, effective
    settings, floors), for audit. samples: rows the runner writes to
    samples/<task>/<label>.jsonl (generated text and the like).'''
    scalars: dict[str, float] = field(default_factory=dict)
    items: dict[str, list[float]] = field(default_factory=dict)
    witness: dict[str, Any] = field(default_factory=dict)
    samples: list[dict] = field(default_factory=list)


_STORES: dict[tuple, TokenStore] = {}


def open_store(data_dir: str | Path, shards: list[str] | None = None) -> TokenStore:
    key = (str(Path(data_dir).resolve()), tuple(shards) if shards else None)
    if key not in _STORES:
        _STORES[key] = TokenStore(data_dir, shards=list(shards) if shards else None)
    return _STORES[key]


class EvalCtx:
    '''The contract between the runner and a task.'''

    def __init__(self, subject: Subject, task: str, seed: int, device: torch.device,
                 dtype: torch.dtype, log: Callable[[str], None]):
        self.subject = subject
        self.task = task
        self.seed = seed
        self.device = device
        self.dtype = dtype
        self.log = log

    def _digest(self, salt: str) -> bytes:
        return hashlib.sha256(f'{self.seed}/{self.task}/{salt}'.encode()).digest()

    def rng(self, salt: str = '') -> np.random.Generator:
        '''A numpy RNG that is a pure function of (seed, task, salt):
        every subject draws the same items from it.'''
        return np.random.default_rng(int.from_bytes(self._digest(salt)[:8], 'little'))

    def torch_seed(self, salt: str = '') -> int:
        '''A seed for SamplingConfig.seed, derived the same way.'''
        return int.from_bytes(self._digest(salt)[8:12], 'little') % (2 ** 31)

    def store(self, data_dir: str | Path, shards: list[str] | None = None) -> TokenStore:
        store = open_store(data_dir, shards)
        check_tokenizer(self.subject, store)
        return store


def check_tokenizer(subject: Subject, store: TokenStore) -> None:
    '''A store's tokens mean nothing to a model trained on another
    tokenizer; refuse when both sides record a hash and they differ.'''
    mine = (subject.meta.get('tokenizer') or {}).get('sha256')
    theirs = store.manifest.get('tokenizer', {}).get('sha256')
    if mine and theirs and mine != theirs:
        raise ValueError(f'{subject.label}: checkpoint tokenizer {mine[:8]} != store tokenizer '
                         f'{theirs[:8]} ({store.dir})')


def exposure(subject: Subject, store: TokenStore) -> dict:
    '''How often the subject can have seen a token of this store during
    training, on the uniform-window assumption: tokens_seen / store
    tokens. `same_store` compares the store's identity with the training
    run's witnessed data record when one is available.'''
    seen = subject.meta.get('tokens_seen')
    total = store.total_tokens
    train = (subject.run_meta or {}).get('data')
    same = None
    if train:
        same = (train.get('source') == store.manifest.get('source') and
                (train.get('tokenizer') or {}).get('sha256') == store.manifest['tokenizer'].get('sha256'))
    return {
        'tokens_seen': seen,
        'store_tokens': total,
        'expected_views': (seen / total) if seen else None,
        'same_store': same,
        'train_total_tokens': train.get('total_tokens') if train else None,
    }


def sha256_file(path: str | Path, chunk: int = 1 << 23) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()
