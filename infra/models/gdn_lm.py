# Gated delta-rule language model — the LAX series' architecture: the
# Transformer++ skeleton (pre-norm blocks, SwiGLU FFN, tied embedding,
# RMSNorm head) with the GatedDeltaNet mixer family as the token mixer:
# gate='vector' is Kimi Delta Attention (LAX1), 'scalar' Gated DeltaNet,
# 'none' DeltaNet. layer_pattern admits softmax layers for hybrids
# ('gdn,gdn,gdn,softmax' cycles over depth); a pure pattern has no
# positional encoding and no length limit.

from typing import Any

import torch
from torch import nn

from ..components.attention import SoftmaxAttention
from ..components.block import Block
from ..components.linear_attention import GatedDeltaNet
from ..components.mixer import Mixer
from ..components.norm_layer import RMSNorm
from ..components.pos_embed import RoPE
from .decodable import Decodable


class GDNLM(nn.Module, Decodable):
    def __init__(self,
            vocab_size: int,
            dim: int,
            layer_count: int,
            head_count: int,
            key_head_dim: int,
            value_head_dim: int,
            ffn_hidden_dim: int | None = None,
            short_conv_size: int | None = 4,
            gate: str | bool = 'vector',
            delta: bool = True,
            gate_rank: int = 64,
            chunk_size: int = 64,
            la_impl: str = 'auto',
            layer_pattern: str = 'gdn',
            softmax_head_dim: int = 64,
            context_len: int = 2048,
            rope_base: float = 10000,
            rmsnorm_eps: float = 1e-6,
            tie_embedding: bool = True,
            gate_proj_optimizer: str = 'adamw',
        ):
        super().__init__()
        assert gate_proj_optimizer in ('adamw', 'muon')

        self.config: dict[str, Any] = dict(
            vocab_size=vocab_size, dim=dim, layer_count=layer_count,
            head_count=head_count, key_head_dim=key_head_dim, value_head_dim=value_head_dim,
            ffn_hidden_dim=ffn_hidden_dim, short_conv_size=short_conv_size,
            gate=gate, delta=delta, gate_rank=gate_rank, chunk_size=chunk_size, la_impl=la_impl,
            layer_pattern=layer_pattern, softmax_head_dim=softmax_head_dim,
            context_len=context_len, rope_base=rope_base, rmsnorm_eps=rmsnorm_eps,
            tie_embedding=tie_embedding, gate_proj_optimizer=gate_proj_optimizer,
        )

        kinds = [t.strip() for t in layer_pattern.split(',')]
        assert kinds and all(k in ('gdn', 'softmax') for k in kinds), layer_pattern
        self.kinds = [kinds[i % len(kinds)] for i in range(layer_count)]

        self.embedding = nn.Embedding(vocab_size, dim)
        self.rope = None
        if 'softmax' in self.kinds:
            assert dim % softmax_head_dim == 0
            self.rope = RoPE(dim, softmax_head_dim, context_len, base=rope_base)

        def mixer(kind: str) -> Mixer:
            if kind == 'gdn':
                return GatedDeltaNet(
                    dim=dim, head_count=head_count, key_head_dim=key_head_dim,
                    value_head_dim=value_head_dim, short_conv_size=short_conv_size,
                    gate=gate, delta=delta, gate_rank=gate_rank, chunk_size=chunk_size, impl=la_impl,
                    init_std=0.02, layer_count=layer_count)
            assert self.rope is not None
            return SoftmaxAttention(
                dim=dim, head_count=dim // softmax_head_dim, kv_head_count=None,
                v_dim_mult=1, short_conv_size=None, qk_norm=True, rope=self.rope,
                init_std=0.02, layer_count=layer_count)

        self.blocks = nn.ModuleList([
            Block(dim=dim, ffn_hidden_dim=ffn_hidden_dim, rmsnorm_eps=rmsnorm_eps,
                  mixer=mixer(kind), layer_count=layer_count)
            for kind in self.kinds
        ])
        self.rms_head = RMSNorm(dim, rmsnorm_eps)
        self.head = nn.Linear(dim, vocab_size, bias=False)

        nn.init.normal_(self.embedding.weight, std=0.02)
        if tie_embedding:
            self.head.weight = self.embedding.weight
        else:
            nn.init.normal_(self.head.weight, std=0.02)

    @classmethod
    def from_config(cls, config: dict) -> 'GDNLM':
        return cls(**config)

    def compile_blocks(self):
        for blk in self.blocks:
            blk.compile()

    def forward(self, tokens, is_causal: bool = True, return_hidden: bool = False):
        x = self.embedding(tokens)
        for blk in self.blocks:
            x = blk(x, is_causal)
        x = self.rms_head(x)
        if return_hidden:
            return x
        return self.head(x)

    # streaming inference

    @property
    def max_stream_len(self) -> int | None:
        # fixed-size state everywhere: no limit. A hybrid inherits the
        # softmax layers' RoPE window.
        return int(self.config['context_len']) if self.rope is not None else None

    def reset_cache(self, batch_size: int, max_cache_len: int):
        for blk in self.blocks:
            blk.reset_cache(batch_size, max_cache_len)  # type: ignore

    def load_cache(self, cache: dict, max_cache_len: int):
        for blk, c in zip(self.blocks, cache['blocks'], strict=True):
            blk.load_cache(c, max_cache_len=max_cache_len)  # type: ignore

    def export_cache(self) -> dict:
        return {'blocks': [blk.export_cache() for blk in self.blocks]}  # type: ignore

    @torch.no_grad()
    def decode_step(self, tokens, return_logits: bool = True):
        x = self.embedding(tokens)
        for blk in self.blocks:
            x = blk.decode_step(x)  # type: ignore
        if not return_logits:
            return None
        return self.head(self.rms_head(x))

    # training

    def param_groups(self) -> dict[str, list[nn.Parameter]]:
        '''Muon on the block matrices, AdamW elsewhere. The d x H gate
        projections (wa, wb) go to AdamW by default (gate_proj_optimizer):
        Muon's orthogonalisation + rms scaling is untested at that aspect
        ratio, and the fla/GDN reference recipes train them with Adam.
        1-D params (A_log, dt_bias, norm gammas) are never decayed.'''
        skip = set()
        if self.config['gate_proj_optimizer'] == 'adamw':
            for blk in self.blocks:
                skip.update(id(p) for p in getattr(blk.att, 'gate_projections', []))
        muon = [p for p in self.blocks.parameters()
                if p.requires_grad and p.dim() == 2 and id(p) not in skip]
        muon_ids = {id(p) for p in muon}
        adamw_decay = [p for p in self.parameters()
                       if p.requires_grad and id(p) not in muon_ids and p.dim() >= 2]
        adamw_no_decay = [p for p in self.parameters() if p.requires_grad and p.dim() < 2]
        return {'muon': muon, 'adamw_decay': adamw_decay, 'adamw_no_decay': adamw_no_decay}

    def attn_flops_per_token(self, context_len: int) -> float:
        '''Mixer matmul FLOPs per token (fwd + bwd), for the MFU estimate.
        GDN: state write, delta read-back and query read-out are each
        2 * H * dk * dv per token; softmax layers use the 12 * d * L term.'''
        c = self.config
        H, dk, dv, dim = (int(c[k]) for k in ('head_count', 'key_head_dim', 'value_head_dim', 'dim'))
        gdn = (18 if c['delta'] else 12) * H * dk * dv
        sm = 12 * dim * context_len
        return float(sum(gdn if k == 'gdn' else sm for k in self.kinds))

    def metric_hooks(self) -> dict:
        return {'slow': [self._metric_gates]}

    @torch.no_grad()
    def _metric_gates(self, ctx) -> dict:
        '''Gate health on one sequence of the trainer's last micro-batch:
        mean decay a (all gdn layers), the median / max effective memory
        length 1/(1 - a) over heads (over channels for a vector gate),
        mean write strength b, and the worst end-of-sequence state RMS
        over layers. Runs the fp32 torch path
        on submodules directly, outside the compiled block graphs.'''
        tokens = ctx.last_batch
        if tokens is None:
            return {}
        x = self.embedding(tokens[:1])
        alpha, mem, beta, srms = [], [], [], []
        for blk in self.blocks:
            assert isinstance(blk, Block)
            h = blk.rmsnorm1(x)
            if isinstance(blk.att, GatedDeltaNet):
                st = blk.att.gate_stats(h)
                srms.append(float(st['state_rms']))
                if 'alpha' in st:
                    alpha.append(st['alpha']); mem.append(st['mem_len'])
                if 'beta' in st:
                    beta.append(st['beta'])
            x = x + blk.att(h)
            x = x + blk.ffn(blk.rmsnorm2(x))
        out = {}
        if srms:
            out['gdn/state_rms_max'] = max(srms)
        if alpha:
            a = torch.cat(alpha).float(); m = torch.cat(mem).float()
            out.update({'gdn/alpha_mean': float(a.mean()),
                        'gdn/mem_len_median': float(m.median()),
                        'gdn/mem_len_max': float(m.max())})
        if beta:
            out['gdn/beta_mean'] = float(torch.cat(beta).float().mean())
        return out
