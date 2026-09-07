# The pre-norm residual block shared by every model: norm -> token mixer
# -> residual, norm -> SwiGLU FFN -> residual. The mixer is injected
# (SoftmaxAttention, GatedDeltaNet, ...) and carries the architecture;
# the block only owns the norms and the FFN, so a model is a choice of
# mixer per layer plus the embedding/head. What the block needs from a
# mixer is the Mixer protocol (components/mixer.py), checked here at
# construction so a non-conforming module fails before training starts.

import math
import torch
from torch import nn

from .norm_layer import RMSNorm
from .ffn import FFN
from .cache import WithCache
from .mixer import Mixer, missing_mixer


class Block(nn.Module, WithCache):
    '''
    Pre-norm residual block around a token mixer and a SwiGLU FFN. The
    mixer is injected (SoftmaxAttention, GatedDeltaNet, ...) and must
    satisfy the Mixer protocol; the block owns the norms and the FFN.
    Kept under the attribute name `att` (metric probes address it).
    '''
    def __init__(
            self,
            dim: int,
            ffn_hidden_dim: int | None = None,
            rmsnorm_eps: float = 1e-6,
            *,
            mixer: Mixer,
            layer_count: int | None
        ):
        super().__init__()

        missing = missing_mixer(mixer)
        if missing:
            raise TypeError(f'{type(mixer).__name__} is not a Mixer: missing {missing}')

        # Llama's SwiGLU sizing: 8/3 * dim, rounded up to a multiple of 256
        ffn_hidden_dim = ffn_hidden_dim or math.ceil(int(8 * dim / 3) / 256) * 256

        self.rmsnorm1 = RMSNorm(dim, rmsnorm_eps)
        self.att: Mixer = mixer
        self.rmsnorm2 = RMSNorm(dim, rmsnorm_eps)
        self.ffn = FFN(
            dim=dim,
            hidden_dim=ffn_hidden_dim,
            init_std=0.02,
            layer_count=layer_count
        )

    def forward(self, x, is_causal: bool = True):
        dx = self.rmsnorm1(x)
        dx = self.att(dx, is_causal)
        x = x + dx

        dx = self.rmsnorm2(x)
        dx = self.ffn(dx)
        x = x + dx
        
        return x

    def reset_cache(self, batch_size: int, max_cache_len: int):
        self.att.reset_cache(batch_size, max_cache_len)

    def load_cache(self, cache: dict, max_cache_len: int):
        self.att.load_cache(
            cache['att'],
            max_cache_len=max_cache_len
        )

    def export_cache(self) -> dict:

        return {
            'att': self.att.export_cache()
        }


    @torch.no_grad()
    def decode_step(self, x):
        dx = self.rmsnorm1(x)
        dx = self.att.decode_step(dx)
        x = x + dx

        dx = self.rmsnorm2(x)
        dx = self.ffn(dx)
        x = x + dx
        
        return x
