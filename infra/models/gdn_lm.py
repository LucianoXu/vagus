# Gated delta-rule language model — the LAX / HAX series' architecture:
# the Transformer++ skeleton (pre-norm blocks, SwiGLU FFN, tied
# embedding, RMSNorm head) with the GatedDeltaNet mixer family as the
# token mixer: gate='vector' is Kimi Delta Attention (LAX1), 'scalar'
# Gated DeltaNet, 'none' DeltaNet.
#
# Layer kinds (per layer, from layer_kinds or the cyclic layer_pattern):
#     gdn       the linear mixer alone (a pure model has no positional
#               encoding and no length limit)
#     softmax   a whole softmax-attention layer (inter-layer hybrid, the
#               Kimi Linear / Qwen3-Next 3:1 layout)
#     parallel  ParallelMixer: half-width softmax attention and half-width
#               linear mixer side by side in one layer (intra-layer
#               hybrid, the HAX series' block)
# softmax_rope=False makes every softmax branch NoPE (Kimi Linear's
# choice for its global layers: the recurrent mixer is the position-
# aware operator, and an unrotated global layer has no context window).

from typing import Any

import torch
from torch import nn

from ..components.attention import SoftmaxAttention
from ..components.block import Block
from ..components.linear_attention import GatedDeltaNet, chunk_scan_flops
from ..components.mixer import Mixer
from ..components.norm_layer import RMSNorm
from ..components.parallel_mixer import ParallelMixer
from ..components.pos_embed import RoPE
from .decodable import Decodable

KINDS = ('gdn', 'softmax', 'parallel')


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
            la_fused: bool | str | list = False,
            la_conv_impl: str = 'conv1d',
            la_disable_recompute: bool = False,
            gate_lower_bound: float | None = None,
            layer_pattern: str = 'gdn',
            layer_kinds: list[str] | None = None,
            softmax_head_dim: int = 64,
            softmax_rope: bool = True,
            softmax_out_gate: bool = False,
            parallel_width: int | None = None,
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
            la_fused=la_fused, la_conv_impl=la_conv_impl,
            la_disable_recompute=la_disable_recompute,
            gate_lower_bound=gate_lower_bound,
            layer_pattern=layer_pattern, layer_kinds=layer_kinds,
            softmax_head_dim=softmax_head_dim, softmax_rope=softmax_rope,
            softmax_out_gate=softmax_out_gate, parallel_width=parallel_width,
            context_len=context_len, rope_base=rope_base, rmsnorm_eps=rmsnorm_eps,
            tie_embedding=tie_embedding, gate_proj_optimizer=gate_proj_optimizer,
        )

        if layer_kinds is not None:
            kinds = [k.strip() for k in layer_kinds]
            assert len(kinds) == layer_count, \
                f'layer_kinds has {len(kinds)} entries for layer_count {layer_count}'
        else:
            cycle = [t.strip() for t in layer_pattern.split(',')]
            assert cycle, layer_pattern
            kinds = [cycle[i % len(cycle)] for i in range(layer_count)]
        assert all(k in KINDS for k in kinds), kinds
        self.kinds = kinds

        has_softmax = any(k != 'gdn' for k in kinds)
        pw = parallel_width or dim // 2
        if 'parallel' in kinds:
            assert pw % softmax_head_dim == 0 and pw % value_head_dim == 0, \
                f'parallel_width {pw} must hold whole softmax ({softmax_head_dim}) and value ({value_head_dim}) heads'
        self.parallel_width = pw

        self.embedding = nn.Embedding(vocab_size, dim)
        self.rope = None
        if has_softmax and softmax_rope:
            assert dim % softmax_head_dim == 0
            self.rope = RoPE(dim, softmax_head_dim, context_len, base=rope_base)

        def linear(width: int, out_proj: bool) -> GatedDeltaNet:
            assert width % value_head_dim == 0
            return GatedDeltaNet(
                dim=dim, head_count=width // value_head_dim, key_head_dim=key_head_dim,
                value_head_dim=value_head_dim, short_conv_size=short_conv_size,
                gate=gate, delta=delta, gate_rank=gate_rank, chunk_size=chunk_size, impl=la_impl,
                fused=la_fused, conv_impl=la_conv_impl,
                disable_recompute=la_disable_recompute,
                gate_lower_bound=gate_lower_bound,
                init_std=0.02, layer_count=layer_count, out_proj=out_proj)

        def softmax(width: int, out_proj: bool) -> SoftmaxAttention:
            assert width % softmax_head_dim == 0
            return SoftmaxAttention(
                dim=width, head_count=width // softmax_head_dim, kv_head_count=None,
                v_dim_mult=1, short_conv_size=None, qk_norm=True, rope=self.rope,
                init_std=0.02, layer_count=layer_count, in_dim=dim, out_proj=out_proj,
                out_gate=softmax_out_gate)

        def mixer(kind: str) -> Mixer:
            if kind == 'gdn':
                return linear(head_count * value_head_dim, True)
            if kind == 'softmax':
                return softmax(dim, True)
            return ParallelMixer(dim, att=softmax(pw, False), la=linear(pw, False),
                                 rmsnorm_eps=rmsnorm_eps, init_std=0.02, layer_count=layer_count)

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

    def forward(self, tokens, is_causal: bool = True, return_hidden: bool = False,
                caches: list | None = None):
        '''caches: export_cache()['blocks'] (or a list with None for
        blocks started blank) taken as constant entry points — the
        differentiable forward of a stream whose prefix is already in
        the memory. See GatedDeltaNet.forward.'''
        x = self.embedding(tokens)
        if caches is None:
            for blk in self.blocks:
                x = blk(x, is_causal)
        else:
            assert len(caches) == len(self.blocks), (len(caches), len(self.blocks))
            for blk, c in zip(self.blocks, caches):
                x = blk(x, is_causal, cache=None if c is None else c['att'])
        x = self.rms_head(x)
        if return_hidden:
            return x
        return self.head(x)

    @torch.no_grad()
    def surprise(self, tokens) -> torch.Tensor:
        '''(B, L, layers, H) delta-rule residual ratios of every
        GatedDeltaNet layer along a fresh stream (see
        GatedDeltaNet.residuals): the memory's surprise at each token,
        the signal a prioritised replay samples from. All-GDN layouts
        only.'''
        x = self.embedding(tokens)
        out = []
        for blk in self.blocks:
            att = self._linear_of(blk.att)
            assert att is not None and att is blk.att, 'surprise: all-GDN layouts only'
            out.append(att.residuals(blk.rmsnorm1(x)))
            x = blk(x)
        return torch.stack(out, dim=2)

    # --- the mixers by role -----------------------------------------------

    @staticmethod
    def _linear_of(mixer) -> GatedDeltaNet | None:
        if isinstance(mixer, GatedDeltaNet):
            return mixer
        if isinstance(mixer, ParallelMixer):
            return mixer.la
        return None

    @staticmethod
    def _softmax_of(mixer) -> SoftmaxAttention | None:
        if isinstance(mixer, SoftmaxAttention):
            return mixer
        if isinstance(mixer, ParallelMixer):
            return mixer.att
        return None

    # streaming inference

    @property
    def max_stream_len(self) -> int | None:
        # fixed-size state everywhere, or NoPE softmax: no positional
        # limit (a KV cache still needs max_len from the caller). RoPE
        # softmax layers bound the stream to their trained window.
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
        Linear: the chunkwise algorithm's cost per head
        (chunk_scan_flops: recurrence + intra-chunk products at the
        configured chunk size, the kernels' 64); softmax: the PaLM
        12 * width * L term. A parallel layer is the sum of its
        half-width branches.'''
        c = self.config
        dk, dv, dim = (int(c[k]) for k in ('key_head_dim', 'value_head_dim', 'dim'))
        per_head = chunk_scan_flops(dk, dv, int(c['chunk_size']), bool(c['delta']))

        def flops(kind: str) -> float:
            if kind == 'gdn':
                return per_head * int(c['head_count'])
            if kind == 'softmax':
                return 12 * dim * context_len
            return per_head * (self.parallel_width // dv) + 12 * self.parallel_width * context_len

        return float(sum(flops(k) for k in self.kinds))

    def metric_hooks(self) -> dict:
        return {'slow': [self._metric_probe]}

    @torch.no_grad()
    def _metric_probe(self, ctx) -> dict:
        '''Health probe on one sequence of the trainer's last micro-batch,
        calling submodules directly (outside the compiled block graphs).
        Gates, over every linear mixer (pure layers and parallel
        branches): mean decay a, the median / max effective memory length
        1/(1 - a) over heads (over channels for a vector gate), mean write
        strength b, and the worst end-of-sequence state RMS. Softmax: the
        global max pre-softmax logit (the quantity qk-norm / z-loss bound).'''
        tokens = ctx.last_batch
        if tokens is None:
            return {}
        x = self.embedding(tokens[:1])
        alpha, mem, beta, srms = [], [], [], []
        worst = None
        for blk in self.blocks:
            assert isinstance(blk, Block)
            h = blk.rmsnorm1(x)
            la = self._linear_of(blk.att)
            if la is not None:
                st = la.gate_stats(h)
                srms.append(float(st['state_rms']))
                if 'alpha' in st:
                    alpha.append(st['alpha']); mem.append(st['mem_len'])
                if 'beta' in st:
                    beta.append(st['beta'])
            sm = self._softmax_of(blk.att)
            if sm is not None:
                m = sm.max_attn_logit(h)
                worst = m if worst is None else torch.maximum(worst, m)
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
        if worst is not None:
            out['attn_logit_max'] = float(worst)
        return out
