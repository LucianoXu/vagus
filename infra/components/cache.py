# The state interface shared by everything that streams: a mixer's KV
# cache or recurrent matrix, a block (which forwards to its mixer), a
# model (which forwards to its blocks). Mixer (components/mixer.py) and
# Decodable (models/decodable.py) both extend it with their own
# decode_step — the signatures differ (tensors in/out vs tokens in,
# logits out), so only the state handling is common.

from typing import Protocol, runtime_checkable


@runtime_checkable
class WithCache(Protocol):

    def reset_cache(self, batch_size: int, max_cache_len: int) -> None:
        '''Allocate an empty state for `batch_size` streams. max_cache_len
        bounds the tokens a growing state (KV cache) can hold; a
        fixed-size state accepts and ignores it.'''
        ...

    def export_cache(self) -> dict:
        '''A compact, self-contained copy of the state (tensors cloned).'''
        ...

    def load_cache(self, cache: dict, max_cache_len: int) -> None:
        '''Restore a state exported by export_cache, allocating for
        max_cache_len tokens (must be >= the exported prefix length).'''
        ...


def missing_members(protocol, obj) -> list[str]:
    '''Names of `protocol`'s members that `obj` lacks (empty = conforms),
    inherited members included. The structural check: since 3.12
    isinstance() against a runtime_checkable Protocol uses static
    attribute lookup and misses members nn.Module resolves dynamically
    (submodules, buffers), so classes here also inherit their protocol
    explicitly for the nominal check, and this covers the duck-typed case.'''
    return [a for a in sorted(protocol.__protocol_attrs__) if not hasattr(obj, a)]
