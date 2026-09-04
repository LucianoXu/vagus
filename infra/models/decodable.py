# The streaming-inference protocol every registered model implements.
#
# Generation (infra/inference) is written once against this protocol; the
# architecture-specific part — what "state" is — stays inside the model:
# a KV cache for softmax attention, a fixed-size recurrent state for
# linear attention, one per layer for hybrids. The generator never looks
# inside the state; it only moves it around (export/load) and feeds it.
#
# Stream semantics: decode_step(tokens) consumes the *next block* of one
# continuous token stream, whose prefix the state already holds. Prefill
# is just the first (large) block after reset_cache; single-token decode
# is L == 1. Blocks may have any length, including L == 0 (a no-op).

from typing import Protocol, runtime_checkable

import torch
from torch import nn


@runtime_checkable
class Decodable(Protocol):

    # hard upper bound on the stream length this model was built for
    # (context_len for softmax attention: RoPE table + trained length);
    # None for models with no positional limit (pure linear attention)
    max_stream_len: int | None

    def reset_cache(self, batch_size: int, max_cache_len: int) -> None:
        '''Allocate an empty state for `batch_size` streams. max_cache_len
        bounds the number of tokens the state can hold; models with a
        fixed-size state accept and ignore it.'''
        ...

    def decode_step(self, tokens: torch.Tensor, return_logits: bool = True) -> torch.Tensor | None:
        '''Consume the next block, (B, L) int64, advancing the state.
        Returns logits (B, L, vocab), or None when return_logits is False
        — the caller only advances the state (prefill), so the model skips
        whatever the output side costs (the vocab projection, mainly).'''
        ...

    def export_cache(self) -> dict:
        '''A compact, self-contained copy of the state (tensors cloned).'''
        ...

    def load_cache(self, cache: dict, max_cache_len: int) -> None:
        '''Restore a state exported by export_cache, allocating for
        max_cache_len tokens (must be >= the exported prefix length).'''
        ...


def missing_decodable(obj) -> list[str]:
    '''Names of protocol members `obj` lacks (empty = conforms).

    Models declare conformance by inheriting Decodable explicitly
    (class TransformerPP(nn.Module, Decodable)), which makes isinstance()
    hold nominally. This structural check exists for the duck-typed case:
    since 3.12 isinstance() against a runtime_checkable Protocol uses
    static attribute lookup and misses members nn.Module
    resolves dynamically (submodules, buffers).'''
    return [a for a in sorted(Decodable.__protocol_attrs__) if not hasattr(obj, a)]
