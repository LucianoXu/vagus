# The recipe of Transformer++

from typing import Any

import torch
from torch import nn

from ..components.pos_embed import RoPE
from ..components.norm_layer import RMSNorm
from ..components.attention import SoftmaxAttention
from ..components.block import Block
from .decodable import Decodable

class TransformerPP(nn.Module, Decodable):
    def __init__(self,
            vocab_size: int,
            dim: int,
            head_dim: int,
            context_len: int,
            layer_count: int,
            ffn_hidden_dim: int | None = None,
            kv_head_count: int | None = None,
            rmsnorm_eps: float = 1e-6,
            rope_base: float = 10000,
            tie_embedding: bool = True,
            qk_norm: bool = False,
        ):
        super().__init__()

        self.config: dict[str, Any] = dict(
            vocab_size=vocab_size,
            dim=dim,
            head_dim=head_dim,
            context_len=context_len,
            layer_count=layer_count,
            ffn_hidden_dim=ffn_hidden_dim,
            kv_head_count=kv_head_count,
            rmsnorm_eps=rmsnorm_eps,
            rope_base=rope_base,
            tie_embedding=tie_embedding,
            qk_norm=qk_norm,
        )

        # constructing the pipeline
        self.embedding = nn.Embedding(vocab_size, dim)
        self.rope = RoPE(dim, head_dim, context_len, base=rope_base)
        assert dim % head_dim == 0
        head_count = dim // head_dim
        self.blocks = nn.ModuleList([
            Block(
                dim=dim,
                ffn_hidden_dim=ffn_hidden_dim,
                rmsnorm_eps=rmsnorm_eps,
                mixer=SoftmaxAttention(
                    dim=dim,
                    head_count=head_count,
                    kv_head_count=kv_head_count,
                    v_dim_mult=1,
                    short_conv_size=None,
                    qk_norm=qk_norm,
                    rope=self.rope,
                    init_std=0.02,
                    layer_count=layer_count   # total depth for the 1/sqrt(2L)
                ),                            # residual scaling, same for every layer
                layer_count=layer_count
            )
            for _ in range(layer_count)
        ])
        self.rms_head = RMSNorm(dim, rmsnorm_eps)
        self.head = nn.Linear(
            in_features=dim,
            out_features=vocab_size,
            bias=False
        )

        nn.init.normal_(self.embedding.weight, std=0.02)
        if tie_embedding:
            self.head.weight = self.embedding.weight
        else:
            nn.init.normal_(self.head.weight, std=0.02)

    @classmethod
    def from_config(cls, config: dict) -> "TransformerPP":
        '''Rebuild from the dict stored in self.config
        (e.g. saved alongside a state_dict in a checkpoint).'''
        return cls(**config)

    def compile_blocks(self):
        '''Regional compilation, a post-construction runtime action.
        Compiles each block's forward; decode_step stays eager (see the
        SoftmaxAttention docstring for what compiling it would take).'''
        for blk in self.blocks:
            blk.compile()

    def forward(self, tokens, is_causal: bool = True, return_hidden: bool = False):
        # tokens : (B, L) int64
        x = self.embedding(tokens)
        for blk in self.blocks:
            x = blk(x, is_causal)
        x = self.rms_head(x)
        if return_hidden:
            # pre-head hidden for fused-CE losses (the head projection is
            # folded into the loss; self.head.weight is read by the caller).
            # Must stay reachable through the DDP-wrapped forward, so this
            # is a flag rather than a separate method.
            return x
        return self.head(x)   # (B, L, vocab_size) logits

    # streaming inference

    def reset_cache(self, batch_size: int, max_cache_len: int):
        for blk in self.blocks:
            blk.reset_cache(batch_size, max_cache_len)  # type: ignore

    def load_cache(self, cache: dict, max_cache_len: int):
        for blk, c in zip(self.blocks, cache['blocks'], strict=True):
            blk.load_cache(c, max_cache_len=max_cache_len)  # type: ignore

    def export_cache(self) -> dict:
        return {'blocks': [blk.export_cache() for blk in self.blocks]}  # type: ignore

    @property
    def max_stream_len(self) -> int:
        # the trained window; reset_cache with a larger max_cache_len
        # extends the RoPE table (extrapolation, not a supported regime)
        return int(self.config['context_len'])

    @torch.no_grad()
    def decode_step(self, tokens, return_logits: bool = True):
        # tokens: the next block of the stream, (B, L) int64; L == 0 is a
        # no-op. return_logits=False advances the state only (prefill):
        # no (B, L, vocab) projection is ever materialised for a prompt.
        x = self.embedding(tokens)
        for blk in self.blocks:
            x = blk.decode_step(x)  # type: ignore
        if not return_logits:
            return None
        return self.head(self.rms_head(x))

    # training

    def param_groups(self) -> dict[str, list[nn.Parameter]]:
        '''Three-way split for the Muon + AdamW pattern.
        For plain AdamW, use muon + adamw_decay as the decay group.'''
        muon = [p for p in self.blocks.parameters() if p.requires_grad and p.dim() == 2]
        muon_ids = {id(p) for p in muon}   # `p in list` would call Tensor.__eq__
        adamw_decay = [p for p in self.parameters()
                       if p.requires_grad and id(p) not in muon_ids and p.dim() >= 2]
        adamw_no_decay = [p for p in self.parameters() if p.requires_grad and p.dim() < 2]
        return {
            'muon': muon,
            'adamw_decay': adamw_decay,
            'adamw_no_decay': adamw_no_decay,
        }

    def attn_flops_per_token(self, context_len: int) -> float:
        '''Attention matmul FLOPs per token (fwd + bwd), for the MFU
        estimate: QK^T and PV are 2 * d * L each forward, x3 with the
        backward, per layer — the PaLM 12 * layers * d * L term.'''
        return float(12 * int(self.config['layer_count']) * int(self.config['dim']) * context_len)

    def metric_hooks(self) -> dict:
        return {'slow': [self._metric_max_attn_logit]}

    @torch.no_grad()
    def _metric_max_attn_logit(self, ctx) -> dict:
        '''Global max pre-softmax attention logit, probed on one sequence
        of the trainer's most recent micro-batch (ctx.last_batch). Calls
        submodules directly rather than Block.forward, so the probe's
        (B=1, no_grad) shapes never touch the compiled graphs.'''
        tokens = ctx.last_batch
        if tokens is None:
            return {}
        x = self.embedding(tokens[:1])
        worst = None
        for blk in self.blocks:
            assert isinstance(blk, Block) and isinstance(blk.att, SoftmaxAttention)
            h = blk.rmsnorm1(x)
            m = blk.att.max_attn_logit(h)
            worst = m if worst is None else torch.maximum(worst, m)
            x = x + blk.att(h)
            x = x + blk.ffn(blk.rmsnorm2(x))
        return {} if worst is None else {'attn_logit_max': float(worst)}
