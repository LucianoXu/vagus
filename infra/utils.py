
from typing import Iterator

import torch
from torch import nn


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
