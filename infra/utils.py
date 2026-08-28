
import os
import subprocess
from pathlib import Path
from typing import Callable, Iterator

import torch
from torch import nn


def atomic_write(path: str | Path, write_fn: Callable) -> None:
    '''Write via tmp + os.replace, keeping the invariant both the data
    pipeline and checkpointing rely on: a file at `path` is either absent
    or complete, never partial. write_fn receives the open binary handle.'''
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    with open(tmp, 'wb') as f:
        write_fn(f)
    os.replace(tmp, path)


def git_state() -> dict:
    '''Commit + dirty flag of the vagus repo, for pinning code identity
    into data manifests and run metadata.'''
    root = Path(__file__).resolve().parents[1]
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


def _iter_tensors(obj) -> Iterator[torch.Tensor]:
    if isinstance(obj, torch.Tensor):
        yield obj
    elif isinstance(obj, nn.Module):
        yield from obj.parameters()
        yield from obj.buffers()
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _iter_tensors(v)
    elif isinstance(obj, (list, tuple, set)):
        for v in obj:
            yield from _iter_tensors(v)


def infer_device(*objs) -> torch.device:
    for t in _iter_tensors(objs):
        return t.device
    return torch.device('cpu')


def infer_dtype(*objs) -> torch.dtype:
    # integer tensors (token ids, masks) don't determine the working dtype,
    # so prefer the first floating-point tensor
    fallback = None
    for t in _iter_tensors(objs):
        if t.is_floating_point() or t.is_complex():
            return t.dtype
        if fallback is None:
            fallback = t.dtype
    return fallback if fallback is not None else torch.get_default_dtype()
