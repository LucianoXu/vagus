
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention.bias import causal_lower_right

import math

from ..utils import infer_device, infer_dtype
from .norm_layer import RMSNorm
from .opt import ShortConv
from .pos_embed import RoPE


class SoftmaxAttention(nn.Module):
    '''
    Multi-head softmax attention with RoPE, grouped-query attention
    (kv_head_count < head_count shares each K/V head across a group of
    query heads; None or == head_count is plain MHA), optionally widened
    V heads (v_dim_mult) and a depthwise short conv on the q/k/v
    projections. GQA shrinks the KV cache and its decode read traffic by
    head_count / kv_head_count. qk_norm applies a per-head RMSNorm to q and
    k before RoPE (Qwen/Gemma style) for logit stability at large lr.

    Init is Transformer++ style, hardcoded: N(0, init_std) on all
    projections, wo shrunk by 1/sqrt(2 * layer_count) when layer_count is
    given (pass it in any multi-layer model; None leaves wo at init_std,
    for standalone use). The short convs keep their own default init. RoPE
    base is set on the shared RoPE module (default 10000).

    forward() is the stateless training/prefill path; decode_step() streams
    blocks against a preallocated KV cache (reset_cache/load_cache first).

    torch.compile: forward compiles as-is. For decode_step, cache_len is a
    python int attribute that changes every step, and dynamo specializes
    module ints — a plain torch.compile(attn.decode_step) recompiles per
    step until the recompile limit, then falls back to eager. Mark the
    attribute dynamic first (torch >= 2.7):

        import torch.compiler.config as ccfg
        ccfg.dynamic_sources = ','.join(
            filter(None, [ccfg.dynamic_sources, '.*cache_len']))

    Matching is a regex fullmatch on the internal source name, so use the
    suffix pattern — exact spellings like "L['self'].cache_len" do not
    match. On older torch fall back to
    torch._dynamo.config.allow_unspec_int_on_nn_module = True. Keep decode
    block sizes fixed (e.g. L=1) so each path compiles one static shape.
    '''

    def __init__(
            self, 
            dim: int,
            head_count: int,
            kv_head_count: int | None,
            v_dim_mult: int,
            short_conv_size: int | None = None,
            qk_norm: bool = False,
            *,
            rope: RoPE,
            init_std: float = 0.02,
            layer_count: int | None = None,
        ):
        super().__init__()

        if kv_head_count is None:
            kv_head_count = head_count

        assert dim % head_count == 0
        assert head_count % kv_head_count == 0
        assert v_dim_mult > 0
        assert rope.head_dim == dim // head_count

        self.dim = dim
        self.v_dim_mult = v_dim_mult
        self.head_count = head_count
        self.kv_head_count = kv_head_count
        self.short_conv_size = short_conv_size
        self.qk_norm = qk_norm

        self.rope = rope

        Dh = self.dim // self.head_count


        self.wq = nn.Linear(
            in_features=self.dim,
            out_features=self.dim,
            bias = False
        )

        self.wk = nn.Linear(
            in_features=self.dim,
            out_features=Dh * self.kv_head_count,
            bias = False
        )

        self.wv = nn.Linear(
            in_features=self.dim,
            out_features=Dh * self.kv_head_count * self.v_dim_mult,
            bias = False
        )

        self.wo = nn.Linear(
            in_features=self.dim * self.v_dim_mult,
            out_features=self.dim,
            bias = False
        )

        if self.short_conv_size is not None:
            self.conv_q = ShortConv(self.dim, self.short_conv_size)
            self.conv_k = ShortConv(Dh * self.kv_head_count, self.short_conv_size)
            self.conv_v = ShortConv(Dh * self.kv_head_count * self.v_dim_mult, self.short_conv_size)

        # per-head RMSNorm on q/k, applied before RoPE (Qwen/Gemma style)
        if self.qk_norm:
            self.q_norm = RMSNorm(Dh)
            self.k_norm = RMSNorm(Dh)

        # Transformer++ init: N(0, init_std) everywhere, with the residual
        # output projection shrunk by 1/sqrt(2 * layer_count) so the residual
        # stream keeps O(1) variance at init regardless of depth
        for lin in (self.wq, self.wk, self.wv):
            nn.init.normal_(lin.weight, std=init_std)
        wo_std = init_std / math.sqrt(2 * layer_count) if layer_count else init_std
        nn.init.normal_(self.wo.weight, std=wo_std)


        # KV cache in (B, H, T, Dh) layout so slices feed SDPA without copies.
        # buffers: never saved, moved by .to();
        # allocate with reset_cache/load_cache after the module is placed.
        self.cache_len = 0
        self.register_buffer("k_cache", torch.zeros(0, 0, 0, 0), persistent=False)
        self.register_buffer("v_cache", torch.zeros(0, 0, 0, 0), persistent=False)

        if self.short_conv_size is not None:
            self.register_buffer("qp_cache", torch.zeros(0, self.short_conv_size, self.dim), persistent=False)
            self.register_buffer("kp_cache", torch.zeros(0, self.short_conv_size, Dh * self.kv_head_count), persistent=False)
            self.register_buffer("vp_cache", torch.zeros(0, self.short_conv_size, Dh * self.kv_head_count * self.v_dim_mult), persistent=False)


    def forward(self, x, is_causal: bool = True):
        '''
        Standard block forward. No caching.
        '''

        # x : (b, l, d)

        B, L = x.shape[0], x.shape[1]
        H, H_kv, Dh = self.head_count, self.kv_head_count, self.dim // self.head_count
        Dvh = self.dim * self.v_dim_mult // self.head_count   # widened value head dim

        qp = self.wq(x)
        kp = self.wk(x)
        vp = self.wv(x)

        if self.short_conv_size is not None:
            qp, kp, vp = self.conv_q(qp), self.conv_k(kp), self.conv_v(vp)

        q = qp.reshape(B, L, H, Dh).transpose(1, 2)                         # (B, H, L, Dh)
        k = kp.reshape(B, L, H_kv, Dh).transpose(1, 2)                      # (B, H_kv, L, Dh)
        if self.qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)
        q = self.rope(q)
        k = self.rope(k)
        v = vp.reshape(B, L, H_kv, Dvh).transpose(1, 2)                     # (B, H_kv, L, Dvh)

        # fused scaled-dot-product attention (Flash-style): never materialises
        # the (B, H, L, L) score matrix, scales by 1/sqrt(Dh). V may be widened.
        out = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal, enable_gqa=H_kv!=H)

        out = out.transpose(1, 2).reshape(B, L, -1)   # (B, H, L, Dvh) -> (B, L, dim*v_dim_mult)

        x = self.wo(out)   # (B, L, dim*v_dim_mult) -> (B, L, dim)

        return x


    @torch.no_grad()
    def max_attn_logit(self, x) -> torch.Tensor:
        '''Max causal pre-softmax logit over the input (the health signal
        z-loss/qk_norm exist to bound; marin dashboard item). Mirrors
        forward()'s q/k path exactly, then materialises the (B, H, L, L)
        score matrix — probe-sized inputs only (a sequence or two), not
        the training batch. fp32 for an exact max.'''
        B, L = x.shape[0], x.shape[1]
        H, H_kv, Dh = self.head_count, self.kv_head_count, self.dim // self.head_count

        qp, kp = self.wq(x), self.wk(x)
        if self.short_conv_size is not None:
            qp, kp = self.conv_q(qp), self.conv_k(kp)
        q = qp.reshape(B, L, H, Dh).transpose(1, 2)
        k = kp.reshape(B, L, H_kv, Dh).transpose(1, 2)
        if self.qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)
        q = self.rope(q)
        k = self.rope(k)
        if H_kv != H:
            k = k.repeat_interleave(H // H_kv, dim=1)

        scores = q.float() @ k.float().transpose(-1, -2) / math.sqrt(Dh)
        causal = torch.ones(L, L, dtype=torch.bool, device=x.device).tril()
        return scores.masked_fill(~causal, float('-inf')).amax()

    # for reasoning
    def reset_cache(self, batch_size: int, max_cache_len: int):
        device = infer_device(self)
        dtype = infer_dtype(self)

        H_kv, Dh = self.kv_head_count, self.dim // self.head_count
        Dvh = self.dim * self.v_dim_mult // self.head_count

        self.cache_len = 0
        self.k_cache = torch.zeros(batch_size, H_kv, max_cache_len, Dh, device=device, dtype=dtype)
        self.v_cache = torch.zeros(batch_size, H_kv, max_cache_len, Dvh, device=device, dtype=dtype)

        if self.short_conv_size is not None:
            self.qp_cache = torch.zeros(batch_size, self.short_conv_size, self.dim, device=device, dtype=dtype)
            self.kp_cache = torch.zeros(batch_size, self.short_conv_size, Dh * H_kv, device=device, dtype=dtype)
            self.vp_cache = torch.zeros(batch_size, self.short_conv_size, Dvh * H_kv, device=device, dtype=dtype)

        self.rope.prepare_m(max_cache_len)


    def load_cache(self, cache: dict, max_cache_len: int):
        prefix_len = cache['k_cache'].shape[2]
        if prefix_len > max_cache_len:
            raise ValueError("The cache length already exceeds the given max_cache_len.")

        self.reset_cache(cache['k_cache'].shape[0], max_cache_len)
        self.cache_len = prefix_len

        self.k_cache[:, :, :prefix_len].copy_(cache['k_cache'])
        self.v_cache[:, :, :prefix_len].copy_(cache['v_cache'])

        if self.short_conv_size is not None:
            self.qp_cache.copy_(cache['qp_cache'])
            self.kp_cache.copy_(cache['kp_cache'])
            self.vp_cache.copy_(cache['vp_cache'])

    def export_cache(self) -> dict:
        # only the valid prefix, so the export is compact and can be
        # re-loaded with any max_cache_len >= cache_len
        cache = {
            'k_cache' : self.k_cache[:, :, :self.cache_len].clone(),
            'v_cache' : self.v_cache[:, :, :self.cache_len].clone(),
        }

        if self.short_conv_size is not None:
            cache.update(
                {
                    'qp_cache' : self.qp_cache.clone(),
                    'kp_cache' : self.kp_cache.clone(),
                    'vp_cache' : self.vp_cache.clone(),
                }
            )

        return cache
        

    @torch.no_grad()
    def decode_step(self, x):
        '''
        Streaming continuation: x is the next block of the sequence, the
        cache holds everything before it. Prefill is just the first
        (large) block after reset_cache; single-token decode is L == 1.
        '''

        # x : (b, l, d)

        B, L = x.shape[0], x.shape[1]

        assert x.dtype == self.k_cache.dtype and x.device == self.k_cache.device
        assert self.k_cache.shape[2] > 0, "No cache allocated. Call reset_cache or load_cache first."
        assert B == self.k_cache.shape[0], "The input batch_size does not match with the cache. Reset or load cache first."

        offset = self.cache_len
        T = offset + L
        assert T <= self.k_cache.shape[2], "Cache overflow. Allocate with a larger max_cache_len."

        H, H_kv, Dh = self.head_count, self.kv_head_count, self.dim // self.head_count
        Dvh = self.dim * self.v_dim_mult // self.head_count   # widened value head dim

        qp, kp, vp = self.wq(x), self.wk(x), self.wv(x)

        if self.short_conv_size is not None:

            # prepend the last short_conv_size raw projections, so
            # direct_conv (no padding) yields exactly the outputs for the
            # L new positions, each seeing its kernel-1 predecessors
            qp = torch.concat([self.qp_cache, qp], dim=1)
            kp = torch.concat([self.kp_cache, kp], dim=1)
            vp = torch.concat([self.vp_cache, vp], dim=1)

            # update the conv state in place
            self.qp_cache.copy_(qp[:, -self.short_conv_size:, :])
            self.kp_cache.copy_(kp[:, -self.short_conv_size:, :])
            self.vp_cache.copy_(vp[:, -self.short_conv_size:, :])

            qp = self.conv_q.direct_conv(qp)[:, -L:, :]
            kp = self.conv_k.direct_conv(kp)[:, -L:, :]
            vp = self.conv_v.direct_conv(vp)[:, -L:, :]

        q = qp.reshape(B, L, H, Dh).transpose(1, 2)                              # (B, H, L, Dh)
        k = kp.reshape(B, L, H_kv, Dh).transpose(1, 2)                           # (B, H_kv, L, Dh)
        if self.qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)
        q = self.rope(q, offset)
        k = self.rope(k, offset)
        v = vp.reshape(B, L, H_kv, Dvh).transpose(1, 2)                          # (B, H_kv, L, Dvh)

        # keys are cached post-RoPE: each key's rotation is fixed by its
        # absolute position, so it never needs to be recomputed
        self.k_cache[:, :, offset:T].copy_(k)
        self.v_cache[:, :, offset:T].copy_(v)
        self.cache_len = T

        k_all = self.k_cache[:, :, :T]   # views into the preallocated cache,
        v_all = self.v_cache[:, :, :T]   # no copy

        if L == 1:
            # a single query attends to the whole prefix: no mask needed
            out = F.scaled_dot_product_attention(q, k_all, v_all,
                                                 enable_gqa=H_kv != H)
        else:
            # block continuation needs a LOWER-RIGHT-aligned causal mask
            # (query i sits at global position offset+i); is_causal=True
            # causal_lower_right keeps the fused kernels.
            out = F.scaled_dot_product_attention(q, k_all, v_all,
                                                 attn_mask=causal_lower_right(L, T),
                                                 enable_gqa=H_kv != H)

        out = out.transpose(1, 2).reshape(B, L, -1)   # (B, H, L, Dvh) -> (B, L, dim*v_dim_mult)

        x = self.wo(out)   # (B, L, dim*v_dim_mult) -> (B, L, dim)

        return x

