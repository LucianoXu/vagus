# The streaming-inference protocol every registered model implements:
# WithCache (components/cache.py — the state handling shared with blocks
# and mixers) plus the model-level decode step and its length limit.
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

from ..components.cache import WithCache, missing_members


@runtime_checkable
class Decodable(WithCache, Protocol):

    @property
    def max_stream_len(self) -> int | None:
        '''Hard upper bound on the stream length this model was built for
        (context_len for softmax attention: RoPE table + trained length);
        None for models with no positional limit (pure linear attention).'''
        ...

    def decode_step(self, tokens: torch.Tensor, return_logits: bool = True) -> torch.Tensor | None:
        '''Consume the next block, (B, L) int64, advancing the state.
        Returns logits (B, L, vocab), or None when return_logits is False
        — the caller only advances the state (prefill), so the model skips
        whatever the output side costs (the vocab projection, mainly).'''
        ...


def missing_decodable(obj) -> list[str]:
    '''Names of Decodable members `obj` lacks (empty = conforms). Models
    also inherit Decodable explicitly (class TransformerPP(nn.Module,
    Decodable)) for the nominal isinstance(); see
    components.cache.missing_members for why both checks exist.'''
    return missing_members(Decodable, obj)
