# Intra-layer (parallel) hybrid mixer: a softmax-attention branch and a
# linear-attention branch read the same normalised input side by side,
# each output is normalised on its own, the two are concatenated and one
# shared output projection maps them back to the residual width. The
# HAX series' hybrid block (vs the inter-layer alternative of whole
# softmax layers, which GDNLM's 'softmax' kind still provides).
#
#     y = wo( [ norm_a(att(x)) ; norm_l(la(x)) ] )
#
# Design, after Hymba (arXiv:2411.13676) and the Meta hybrid study
# (arXiv:2510.04800): per-branch normalisation is the piece that matters
# (the recurrent branch's output magnitude systematically exceeds the
# attention branch's; without it one branch dominates at init), fusion
# by concatenation is as good as any, and the learnable per-branch
# scales Hymba adds are redundant once the branches are normalised. Both
# branches are half-width (1:1 split, the study's recommended ratio):
# with q/k/v/o at d x d/2 per branch plus the d x d output projection
# the block costs 4 d^2 like MHA, so a parallel layer keeps the param
# coordinate of the layer it replaces.
#
# The branches are built without their own wo (out_proj=False) and
# expose out_width; this module only owns the two norms and wo. State
# handling composes: each branch keeps its own cache (KV cache /
# recurrent matrix), exported and reloaded under 'att' / 'la'.

import math

import torch
from torch import nn

from .attention import SoftmaxAttention
from .linear_attention import GatedDeltaNet
from .mixer import Mixer
from .norm_layer import RMSNorm


class ParallelMixer(nn.Module, Mixer):

    def __init__(
            self,
            dim: int,
            *,
            att: SoftmaxAttention,
            la: GatedDeltaNet,
            rmsnorm_eps: float = 1e-6,
            init_std: float = 0.02,
            layer_count: int | None = None,
        ):
        super().__init__()
        assert not att.out_proj and not la.out_proj, \
            'ParallelMixer owns the output projection: build branches with out_proj=False'
        assert att.in_dim == dim and la.dim == dim, 'both branches read the residual width'

        self.dim = dim
        self.att = att
        self.la = la
        self.att_norm = RMSNorm(att.out_width, rmsnorm_eps)
        self.la_norm = RMSNorm(la.out_width, rmsnorm_eps)
        self.wo = nn.Linear(att.out_width + la.out_width, dim, bias=False)
        wo_std = init_std / math.sqrt(2 * layer_count) if layer_count else init_std
        nn.init.normal_(self.wo.weight, std=wo_std)

    @property
    def gate_projections(self) -> list[nn.Parameter]:
        '''The linear branch's skinny gate/beta matrices (see
        GatedDeltaNet.gate_projections): a model's param_groups may
        route them to AdamW.'''
        return self.la.gate_projections

    def _fuse(self, a, b):
        return self.wo(torch.cat([self.att_norm(a), self.la_norm(b)], dim=-1))

    def forward(self, x, is_causal: bool = True):
        return self._fuse(self.att(x, is_causal), self.la(x, is_causal))

    # --- streaming ----------------------------------------------------

    def reset_cache(self, batch_size: int, max_cache_len: int):
        self.att.reset_cache(batch_size, max_cache_len)
        self.la.reset_cache(batch_size, max_cache_len)

    def load_cache(self, cache: dict, max_cache_len: int):
        self.att.load_cache(cache['att'], max_cache_len=max_cache_len)
        self.la.load_cache(cache['la'], max_cache_len=max_cache_len)

    def export_cache(self) -> dict:
        return {'att': self.att.export_cache(), 'la': self.la.export_cache()}

    @torch.no_grad()
    def decode_step(self, x):
        return self._fuse(self.att.decode_step(x), self.la.decode_step(x))
