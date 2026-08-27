# Tokenization stage: local source parquet -> uint16 token shards + manifest.
#
# Format "vagus-tokens-v1". One source parquet maps to one output shard, so
# the whole stage is an idempotent per-file map: resume after a crash by
# re-running, prepare a prefix of the corpus and extend it later. Doc order
# is preserved 1:1, which keeps the traceability chain dereferenceable:
# token position -> (.idx) -> doc ordinal -> source parquet row -> url/score.
#
# Per shard NNN.npy (uint16 token stream: per-doc BOS prefix, no EOS, docs
# concatenated in source order) and NNN.idx.npy (uint64 doc start offsets,
# docs+1 entries, last = shard token count). manifest.json records the
# identity triple the stream is a pure function of: ordered source list,
# tokenizer fingerprint, format version — plus per-shard sha256 so any
# copy of the data can be verified after transfer.

import hashlib
import importlib
import json
import os
import subprocess
import time
from array import array
from datetime import datetime
from pathlib import Path

import numpy as np

FORMAT = "vagus-tokens-v1"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()


def _atomic_write(path: Path, write_fn):
    '''Write via tmp + rename, keeping "file present == file complete".'''
    tmp = path.with_name(path.name + '.tmp')
    with open(tmp, 'wb') as f:
        write_fn(f)
    os.replace(tmp, path)


def _git_state() -> dict:
    root = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.run(
            ['git', 'rev-parse', 'HEAD'], cwd=root,
            capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(
            ['git', 'status', '--porcelain'], cwd=root,
            capture_output=True, text=True, check=True).stdout.strip())
        return {'commit': commit, 'dirty': dirty}
    except Exception:
        return {'commit': None, 'dirty': None}


def _tokenize_file(tok, src: Path, batch_rows: int):
    '''One source parquet -> (uint16 token array, uint64 doc offsets).'''
    # prep-only dep, lazy: keeps infra.dataset importable in the bare
    # training env (the training side never runs this stage)
    import pyarrow.parquet as pq

    buf = array('H')            # native uint16; overflows loudly on id >= 2**16
    offsets = array('Q', [0])
    for batch in pq.ParquetFile(src).iter_batches(batch_size=batch_rows,
                                                  columns=['text']):
        for enc in tok.encode_batch(batch.column('text').to_pylist()):
            buf.extend(enc.ids)
            offsets.append(len(buf))
    return np.frombuffer(buf, dtype=np.uint16), np.frombuffer(offsets, dtype=np.uint64)


def prepare(
        sources: list[Path],
        provenance: dict,
        out_dir: str | Path,
        tokenizer_id: str = "mistral32k",
        batch_rows: int = 1000,
    ) -> Path:
    '''
    Tokenize `sources` (order matters: it defines the token stream) into
    out_dir and return the manifest path. `provenance` is the source
    identity block recorded verbatim in the manifest (for fineweb_edu:
    repo_id, revision, sample — shard entries then carry file names only).

    Idempotent: shards already listed in the manifest and present on disk
    are skipped, so re-running resumes; calling again with more sources
    extends the dataset. The identity blocks (format, tokenizer,
    provenance) must match the existing manifest — anything else is a
    different dataset and belongs in a different out_dir.
    '''
    tok_mod = importlib.import_module(f'infra.tokenizers.{tokenizer_id}')
    tok = tok_mod.load()
    assert tok.get_vocab_size() <= 1 << 16, "uint16 shard format requires vocab <= 65536"

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / 'manifest.json'

    tokenizer_block = {
        'id': tokenizer_id,
        'sha256': tok_mod.fingerprint(),
        'vocab_size': tok.get_vocab_size(),
        'convention': 'per-doc BOS prefix, no EOS, docs concatenated in source order',
    }

    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        for key, expect in [('format', FORMAT), ('source', provenance),
                            ('tokenizer', tokenizer_block)]:
            if manifest[key] != expect:
                raise ValueError(f"out_dir holds a different dataset: {key} mismatch")
    else:
        manifest = {
            'format': FORMAT,
            'status': 'building',
            'created': datetime.now().astimezone().isoformat(timespec='seconds'),
            'source': provenance,
            'tokenizer': tokenizer_block,
            'dtype': 'uint16',
            'files': [],
            'shards': [],
            'total_tokens': 0,
            'total_docs': 0,
            'code': _git_state(),
        }

    done = {s['source'] for s in manifest['shards']
            if (out_dir / s['file']).exists() and (out_dir / s['idx']).exists()}
    manifest['files'] = [src.name for src in sources]

    for src in sources:
        if src.name in done:
            continue

        t0 = time.perf_counter()
        tokens, offsets = _tokenize_file(tok, src, batch_rows)

        shard = out_dir / f'{src.stem}.npy'
        idx = out_dir / f'{src.stem}.idx.npy'
        _atomic_write(shard, lambda f: np.save(f, tokens))
        _atomic_write(idx, lambda f: np.save(f, offsets))

        entry = {
            'file': shard.name,
            'idx': idx.name,
            'source': src.name,
            'tokens': len(tokens),
            'docs': len(offsets) - 1,
            'sha256': _sha256(shard),
        }
        manifest['shards'] = [s for s in manifest['shards']
                              if s['source'] != src.name] + [entry]
        manifest['shards'].sort(key=lambda s: s['source'])
        _update_totals(manifest, sources)
        _atomic_write(manifest_path,
                      lambda f: f.write(json.dumps(manifest, indent=2).encode()))

        dt = time.perf_counter() - t0
        print(f"{src.name}: {entry['tokens']:,} tokens, {entry['docs']:,} docs "
              f"in {dt:.0f}s ({entry['tokens']/dt/1e6:.1f} Mtok/s)")

    _update_totals(manifest, sources)
    _atomic_write(manifest_path,
                  lambda f: f.write(json.dumps(manifest, indent=2).encode()))
    return manifest_path


def _update_totals(manifest: dict, sources: list[Path]):
    manifest['total_tokens'] = sum(s['tokens'] for s in manifest['shards'])
    manifest['total_docs'] = sum(s['docs'] for s in manifest['shards'])
    have = {s['source'] for s in manifest['shards']}
    manifest['status'] = ('complete'
                          if all(src.name in have for src in sources)
                          else 'building')
