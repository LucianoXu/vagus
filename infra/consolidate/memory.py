# The memory as an object: read it out of a Generator state, degrade it
# to a level between full (1) and blank (0), and hand it to the
# differentiable forward. This is the "how to remove information from
# m" axis of the multi-hop curriculum (sleep.py): a hop m -> m' is one
# of these degradations at a lower level.
#
# Supported memory: the GDN matrix states of a GDNLM whose blocks are
# all GatedDeltaNet mixers (LAX1). The short-conv buffers of the state
# are left as they are (decode path) or absent (forward path); they hold
# the last K-1 tokens, not the document.
#
#   scale            S <- level * S on every layer. Uniform fading.
#   layers_topdown   the top (1 - level) fraction of layers blank, the
#                    lower layers intact: the memory is removed from the
#                    output side first.
#   layers_bottomup  the same from the input side.
#   svd              per head, keep the top ceil(level * rank) singular
#                    components of S: the memory loses its weakest
#                    associations first.
#
# Every degradation returns a new state dict; the input is not touched.

import copy
import math

import torch

DEGRADATIONS = ('scale', 'layers_topdown', 'layers_bottomup', 'svd')


def block_states(state: dict) -> list[torch.Tensor]:
    '''The per-block matrix states (B, H, dk, dv) of an exported
    Generator state. Refuses anything but the all-GDN layout.'''
    out = []
    for i, blk in enumerate(state['cache']['blocks']):
        att = blk['att']
        if 'state' not in att or att['state'].dim() != 4:
            raise ValueError(f'block {i}: not a GatedDeltaNet state (keys {sorted(att)})')
        out.append(att['state'])
    return out


def with_block_states(state: dict, states: list[torch.Tensor]) -> dict:
    '''A copy of `state` with the matrix states replaced.'''
    new = copy.deepcopy(state)
    for blk, S in zip(new['cache']['blocks'], states, strict=True):
        blk['att']['state'] = S.clone()
    return new


def degrade(state: dict, level: float, how: str = 'scale') -> dict:
    '''`state` at memory level `level` in [0, 1] (1 = unchanged, 0 = blank).'''
    assert how in DEGRADATIONS, f'{how!r} not in {DEGRADATIONS}'
    assert 0.0 <= level <= 1.0, level
    Ss = block_states(state)
    n = len(Ss)
    if how == 'scale':
        new = [S * level for S in Ss]
    elif how in ('layers_topdown', 'layers_bottomup'):
        keep = int(round(level * n))                    # layers that keep their memory
        if how == 'layers_topdown':
            new = [S if i < keep else torch.zeros_like(S) for i, S in enumerate(Ss)]
        else:
            new = [S if i >= n - keep else torch.zeros_like(S) for i, S in enumerate(Ss)]
    else:
        new = [svd_truncate(S, level) for S in Ss]
    return with_block_states(state, new)


def svd_truncate(S: torch.Tensor, level: float) -> torch.Tensor:
    '''Per head, keep the top ceil(level * min(dk, dv)) singular
    components (0 at level 0).'''
    B, H, dk, dv = S.shape
    r = min(dk, dv)
    k = int(math.ceil(level * r)) if level > 0 else 0
    if k >= r:
        return S.clone()
    if k == 0:
        return torch.zeros_like(S)
    U, sv, Vh = torch.linalg.svd(S.float(), full_matrices=False)
    sv = sv.clone()
    sv[..., k:] = 0
    return ((U * sv[..., None, :]) @ Vh).to(S.dtype)
