# Tokenizer registry: one vendored package per tokenizer id, each with a
# META.json (provenance, hash, conventions). Checkpoints record the id;
# Generator.from_checkpoint resolves it here.

import importlib
import json
from pathlib import Path

_DIR = Path(__file__).resolve().parent

TOKENIZERS = ('mistral32k',)


def meta(tokenizer_id: str) -> dict:
    if tokenizer_id not in TOKENIZERS:
        raise KeyError(f'unknown tokenizer {tokenizer_id!r}; registered: {TOKENIZERS}')
    return json.loads((_DIR / tokenizer_id / 'META.json').read_text())


def load(tokenizer_id: str):
    '''The HF `tokenizers.Tokenizer` for a registered id (imports the
    `tokenizers` package lazily; it is an optional dependency).'''
    meta(tokenizer_id)   # validates the id
    return importlib.import_module(f'{__name__}.{tokenizer_id}').load()


def stream_conventions(tokenizer_id: str) -> tuple[int | None, tuple[int, ...]]:
    '''(start_id, stop_ids) from META["stream"]: how a document starts
    and ends in this tokenizer's packed training streams.'''
    s = meta(tokenizer_id).get('stream', {})
    return s.get('start'), tuple(s.get('stop', ()))
