# The token-mixer protocol: what a Block needs from the module it wraps.
# WithCache (the state handling) plus the two compute paths. The
# layer-level counterpart of models/decodable.Decodable.
#
# A mixer maps (B, L, dim) -> (B, L, dim), statelessly in forward() and
# against its own state in decode_step(). What the state is (KV cache,
# recurrent matrix, conv tail) stays inside the mixer; the block and the
# model only move it around.
#
# Stream semantics match Decodable: decode_step(x) consumes the next
# block of a stream whose prefix the state already holds; L == 1 is
# single-token decode, a large first block is prefill, L == 0 a no-op.

from typing import Protocol, runtime_checkable

import torch

from .cache import WithCache, missing_members


@runtime_checkable
class Mixer(WithCache, Protocol):

    def forward(self, x: torch.Tensor, is_causal: bool = True) -> torch.Tensor:
        '''Stateless training / full-sequence path, (B, L, dim) -> (B, L, dim).'''
        ...

    def __call__(self, x: torch.Tensor, is_causal: bool = True) -> torch.Tensor:
        '''forward through nn.Module's hook machinery — what a Block
        actually invokes. Declared so a Mixer-typed attribute is callable
        for the type checker; nn.Module satisfies it structurally.'''
        ...

    def decode_step(self, x: torch.Tensor) -> torch.Tensor:
        '''Consume the next block (B, L, dim) of the stream, advancing
        the state; returns the mixer output for those L positions.'''
        ...


def missing_mixer(obj) -> list[str]:
    '''Names of Mixer members `obj` lacks (empty = conforms).'''
    return missing_members(Mixer, obj)
