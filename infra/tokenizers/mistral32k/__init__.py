from tokenizers import Tokenizer

import hashlib
import json
from pathlib import Path

_DIR = Path(__file__).resolve().parent
_FILE = _DIR / "tokenizer.json"

UNK_ID, BOS_ID, EOS_ID = 0, 1, 2


def load() -> Tokenizer:
    '''
    The Mistral-7B-v0.1-lineage 32k tokenizer, loaded from the vendored
    tokenizer.json — the exact file the flame/fla benchmark line uses
    (provenance and hash in META.json).

    encode() prepends BOS (`<s>`) via the post_processor baked into the file
    and never appends EOS, so in packed streams BOS is the document
    separator. Pass add_special_tokens=False for raw pieces. vocab_size
    32000 fits uint16 shards.
    '''
    return Tokenizer.from_file(str(_FILE))


def fingerprint() -> str:
    '''sha256 of the vendored tokenizer.json, for pinning in data manifests.'''
    return hashlib.sha256(_FILE.read_bytes()).hexdigest()


def meta() -> dict:
    '''The provenance record (source repo/revision, hash, conventions).'''
    return json.loads((_DIR / "META.json").read_text())
