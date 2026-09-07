# The gated delta-rule family — Kimi Delta Attention (KDA, Kimi Linear,
# arXiv:2510.26692), Gated DeltaNet (arXiv:2412.06464) and their
# ungated / non-delta reductions — as one token mixer with the
# SoftmaxAttention interface and a fixed-size recurrent state instead of
# a KV cache.
#
# Per head, with S in R^{d_k x d_v}, forget gate a_t and write strength
# b_t in (0, 1):
#
#     S_t = (I - b_t k_t k_t^T) Diag(a_t) S_{t-1} + b_t k_t v_t^T
#     o_t = scale * S_t^T q_t
#
# i.e. decay the state's key rows by a_t, then a delta-rule write (the
# write first erases what the state currently returns for k_t, so
# identical keys overwrite instead of accumulating). The gate's
# granularity and the write rule are the two knobs:
#     gate='vector'  a_t in (0,1)^{d_k}, per key channel  -> KDA
#     gate='scalar'  a_t a scalar per head                -> Gated DeltaNet
#     gate='none'    a_t = 1                              -> DeltaNet
#     delta=False    drops the (I - b k k^T) erase        -> gated linear
#                    attention (GLA-like with a vector gate, SSD-like
#                    with a scalar one); both off is plain linear attention.
#
# Three execution paths, one semantics, all verified against each other
# (tests/test_linear_attention.py):
#     'fla'     fla.ops chunk kernels (Triton, CUDA, bf16) — training path:
#               chunk_kda / chunk_gated_delta_rule / chunk_simple_gla by
#               flags. `pip install fla-core` on the cluster; the laptop
#               uv env does not carry it (see pyproject).
#     'torch'   chunkwise WY-form references in fp32 (chunk_scan for
#               scalar gates, chunk_scan_vec for vector gates): the
#               fallback where fla is absent, the decode path for
#               multi-token blocks, and the correctness oracle for the
#               kernels.
#     recurrent token loop in fp32 (recurrent_scan): single-token decode
#               and the oracle for chunk_scan itself.
#
# No positional encoding: the causal short conv and the decay carry
# position, so a pure-GDN model has no length limit (max_stream_len None).

import math

import torch
from torch import nn
from torch.nn import functional as F

from ..utils import infer_device, infer_dtype
from .mixer import Mixer
from .norm_layer import RMSNorm
from .opt import ShortConv

try:
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule  # type: ignore
    from fla.ops.kda import chunk_kda  # type: ignore
    from fla.ops.simple_gla import chunk_simple_gla  # type: ignore
    HAS_FLA = True
except ImportError:  # CUDA/Triton-only package; import-guarded like liger
    chunk_gated_delta_rule = chunk_kda = chunk_simple_gla = None  # type: ignore[assignment]
    HAS_FLA = False

GATES = ('none', 'scalar', 'vector')


# ---------------------------------------------------------------------------
# Scans: pure functions over (B, L, H, d) tensors, fp32 inside.
#   q, k: (B, L, H, dk)   v: (B, L, H, dv)
#   g:    (B, L, H) log a_t <= 0 (scalar gate), (B, L, H, dk) (vector
#         gate, per key channel), or None (a_t = 1)
#   beta: (B, L, H) b_t in (0, 1); ignored when delta=False
#   S0:   (B, H, dk, dv) entry state, or None (zeros)
# return o: (B, L, H, dv) in q's dtype, S: (B, H, dk, dv) fp32
# (fp64 in and out when q is fp64 — the tests' exactness oracle).
# ---------------------------------------------------------------------------

def _compute_dtype(q) -> torch.dtype:
    return torch.float64 if q.dtype == torch.float64 else torch.float32


def recurrent_scan(q, k, v, g, beta, S0, *, scale: float, delta: bool):
    '''Token-by-token recurrence. The definition; everything else must
    match it.'''
    B, L, H, dk = q.shape
    dv = v.shape[-1]
    dt = _compute_dtype(q)
    q, k, v = q.to(dt) * scale, k.to(dt), v.to(dt)
    S = torch.zeros(B, H, dk, dv, device=q.device, dtype=dt) if S0 is None else S0.to(dt)
    outs = []
    for t in range(L):
        if g is not None:
            a = g[:, t].to(dt).exp()
            S = S * (a[..., None] if a.dim() == 3 else a[..., None, None])   # per-channel rows / scalar
        kt, vt = k[:, t], v[:, t]
        if delta:
            assert beta is not None
            kS = torch.einsum('bhk,bhkv->bhv', kt, S)            # what S returns for k_t
            w = kt * beta[:, t].to(dt)[..., None]
            S = S + torch.einsum('bhk,bhv->bhkv', w, vt - kS)
        else:
            S = S + torch.einsum('bhk,bhv->bhkv', kt, vt)
        outs.append(torch.einsum('bhk,bhkv->bhv', q[:, t], S))
    o = torch.stack(outs, dim=1) if L else q.new_zeros(B, 0, H, dv)
    return o.to(q.dtype), S


def chunk_scan(q, k, v, g, beta, S0, *, scale: float, delta: bool, chunk_size: int = 64):
    '''Chunkwise form of recurrent_scan (WY representation of the
    intra-chunk delta-rule product), O(L) sequential steps of size C.
    Scalar (or no) gate; a (B, L, H, dk) gate dispatches to chunk_scan_vec.

    Inside a chunk with entry state S_0, cumulative decay G_i = prod a_j
    and A = Q K^T (causal):
        w_i = b_i (k_i - sum_{j<i} (k_i.k_j) w_j)                  # (I+B) W = diag(b) K
        u_i = b_i (v_i - sum_{j<i} (G_i/G_j)(k_i.k_j) u_j)        # (I+B*D) U = diag(b) V
        u~  = U - diag(G) W S_0
        o   = diag(G) Q S_0 + (A * D) u~            D_ij = G_i / G_j
        S_C = G_C S_0 + K^T diag(G_C / G_j) u~
    The scalar decay commutes with the projections, so W is decay-free
    while U carries the G_i/G_j factors. delta=False sets W = 0, u~ = V.'''
    if g is not None and g.dim() == 4:
        return chunk_scan_vec(q, k, v, g, beta, S0, scale=scale, delta=delta, chunk_size=chunk_size)
    B, L, H, dk = q.shape
    dv = v.shape[-1]
    C = chunk_size
    dt = _compute_dtype(q)
    dev = q.device
    out_dtype = q.dtype
    q, k, v = q.to(dt) * scale, k.to(dt), v.to(dt)

    pad = (-L) % C
    if pad:
        # padded positions: q = k = v = 0, a = 1, b = 0 -> inert
        q, k, v = (F.pad(t, (0, 0, 0, 0, 0, pad)) for t in (q, k, v))
        g = None if g is None else F.pad(g, (0, 0, 0, pad))
        beta = None if beta is None else F.pad(beta, (0, 0, 0, pad))
    N = (L + pad) // C

    def to_chunks(t, last):   # (B, N*C, H, d) -> (B, H, N, C, d)
        return t.view(B, N, C, H, last).permute(0, 3, 1, 2, 4)

    q, k, v = to_chunks(q, dk), to_chunks(k, dk), to_chunks(v, dv)
    if g is None:
        gamma = torch.zeros(B, H, N, C, device=dev, dtype=dt)
    else:
        gamma = g.to(dt).view(B, N, C, H).permute(0, 3, 1, 2).cumsum(-1)   # log G_i
    tril = torch.ones(C, C, device=dev, dtype=torch.bool).tril()
    strict = tril.tril(-1)
    # D_ij = G_i / G_j on i >= j; mask BEFORE exp (the upper triangle
    # holds positive exponents that would overflow)
    D = (gamma[..., :, None] - gamma[..., None, :]).masked_fill(~tril, float('-inf')).exp()
    A = (q @ k.transpose(-1, -2)) * D            # (A * D), causal

    if delta:
        assert beta is not None
        beta = beta.to(dt).view(B, N, C, H).permute(0, 3, 1, 2)             # (B, H, N, C)
        Bm = (beta[..., :, None] * (k @ k.transpose(-1, -2))) * strict      # b_i k_i.k_j, j < i
        eye = torch.eye(C, device=dev, dtype=dt)
        W = torch.linalg.solve_triangular(eye + Bm, beta[..., None] * k,
                                          upper=False, unitriangular=True)
        U = torch.linalg.solve_triangular(eye + Bm * D, beta[..., None] * v,
                                          upper=False, unitriangular=True)
    else:
        W, U = None, v

    G = gamma.exp()                                  # G_i          (B, H, N, C)
    G_end = (gamma[..., -1:] - gamma).exp()          # G_C / G_j    (B, H, N, C)
    S = torch.zeros(B, H, dk, dv, device=dev, dtype=dt) if S0 is None else S0.to(dt)
    outs = []
    for n in range(N):
        Gn = G[:, :, n, :, None]
        Ut = U[:, :, n]
        if W is not None:                            # delta: u~ = U - diag(G) W S_0
            Ut = Ut - Gn * (W[:, :, n] @ S)
        outs.append(Gn * (q[:, :, n] @ S) + A[:, :, n] @ Ut)
        S = G[:, :, n, -1, None, None] * S + k[:, :, n].transpose(-1, -2) @ (G_end[:, :, n, :, None] * Ut)
    o = torch.stack(outs, dim=2)                     # (B, H, N, C, dv)
    o = o.permute(0, 2, 3, 1, 4).reshape(B, N * C, H, dv)[:, :L]
    return o.to(out_dtype), S


def chunk_scan_vec(q, k, v, g, beta, S0, *, scale: float, delta: bool, chunk_size: int = 64):
    '''Chunkwise form for a per-channel gate g: (B, L, H, dk) (KDA).

    With Diag decays the scalar trick (pull G out of the Householder
    product) is gone; instead conjugate by the cumulative decay
    G_i = Diag(exp(gamma_i)): the transition (I - b k k^T) D_i becomes
    G_i (I - b k~_i k^_i^T) with k^_i = G_i k_i, k~_i = G_i^{-1} k_i, and
    every inner product turns into a gated one,
        <x_i, y_j>_g = sum_c x_i[c] y_j[c] exp(gamma_i[c] - gamma_j[c])   (i >= j),
    whose exponents are <= 0 (so k~ is never formed). Then, per chunk:
        (I + M) W = diag(b) K^,  (I + M) U = diag(b) V,   M_ij = b_i <k_i, k_j>_g  (j < i)
        u~  = U - W S_0
        o   = (Q * G) S_0 + A u~,                      A_ij = <q_i, k_j>_g  (j <= i)
        S_C = G_C S_0 + (K * G_C/G_j)^T u~
    The scalar chunk_scan is this with gamma broadcast over channels.
    The pairwise gated products cost C * C * dk per chunk, so the chunk
    loop builds them one chunk at a time.'''
    B, L, H, dk = q.shape
    dv = v.shape[-1]
    C = chunk_size
    dt = _compute_dtype(q)
    dev = q.device
    out_dtype = q.dtype
    q, k, v, g = q.to(dt) * scale, k.to(dt), v.to(dt), g.to(dt)

    pad = (-L) % C
    if pad:
        q, k, v, g = (F.pad(t, (0, 0, 0, 0, 0, pad)) for t in (q, k, v, g))
        beta = None if beta is None else F.pad(beta, (0, 0, 0, pad))
    N = (L + pad) // C

    def to_chunks(t, last):   # (B, N*C, H, d) -> (B, H, N, C, d)
        return t.view(B, N, C, H, last).permute(0, 3, 1, 2, 4)

    q, k, v = to_chunks(q, dk), to_chunks(k, dk), to_chunks(v, dv)
    gamma = to_chunks(g, dk).cumsum(3)                                     # log G_i, (B, H, N, C, dk)
    tril = torch.ones(C, C, device=dev, dtype=torch.bool).tril()
    strict = tril.tril(-1)
    eye = torch.eye(C, device=dev, dtype=dt)
    if delta:
        assert beta is not None
        beta = beta.to(dt).view(B, N, C, H).permute(0, 3, 1, 2)          # (B, H, N, C)

    S = torch.zeros(B, H, dk, dv, device=dev, dtype=dt) if S0 is None else S0.to(dt)
    outs = []
    for n in range(N):
        qn, kn, vn, gn = q[:, :, n], k[:, :, n], v[:, :, n], gamma[:, :, n]    # (B, H, C, .)
        # E_ijc = exp(gamma_i[c] - gamma_j[c]) on i >= j, 0 above (masked before exp)
        E = (gn[:, :, :, None, :] - gn[:, :, None, :, :]).masked_fill(
            ~tril[None, None, :, :, None], float('-inf')).exp()             # (B, H, C, C, dk)
        Aqk = torch.einsum('bhijd,bhjd->bhij', qn[:, :, :, None, :] * E, kn)
        Gn = gn.exp()                                                       # G_i        (B, H, C, dk)
        Gend = (gn[:, :, -1:, :] - gn).exp()                                # G_C / G_j  (B, H, C, dk)
        if beta is not None:                                                # delta
            Akk = torch.einsum('bhijd,bhjd->bhij', kn[:, :, :, None, :] * E, kn)
            bn = beta[:, :, n]                                              # (B, H, C)
            M = (bn[..., None] * Akk) * strict
            W = torch.linalg.solve_triangular(eye + M, bn[..., None] * kn * Gn,
                                              upper=False, unitriangular=True)
            U = torch.linalg.solve_triangular(eye + M, bn[..., None] * vn,
                                              upper=False, unitriangular=True)
            Ut = U - W @ S
        else:
            Ut = vn
        outs.append((qn * Gn) @ S + Aqk @ Ut)
        S = Gn[:, :, -1, :, None] * S + (kn * Gend).transpose(-1, -2) @ Ut
    o = torch.stack(outs, dim=2)                     # (B, H, N, C, dv)
    o = o.permute(0, 2, 3, 1, 4).reshape(B, N * C, H, dv)[:, :L]
    return o.to(out_dtype), S


def fla_scan(q, k, v, g, beta, S0, *, scale: float, delta: bool):
    '''fla chunk kernels. q/k/v must be bf16/fp16 on CUDA; g/beta fp32.
    Semantics identical to recurrent_scan (fla's fused_recurrent
    reference is the same recurrence: decay, then delta write).'''
    if not HAS_FLA:
        raise RuntimeError('fla is not installed (pip install fla-core)')
    S0 = None if S0 is None else S0.to(torch.float32).contiguous()
    assert chunk_gated_delta_rule is not None and chunk_simple_gla is not None
    assert chunk_kda is not None
    if g is not None and g.dim() == 4:
        # per-channel gate: KDA kernel (delta) or GLA-style gated linear
        # attention (no delta: fla's gla kernel takes the same g layout)
        if not delta:
            raise NotImplementedError('vector gate without delta has no fla path here')
        assert beta is not None
        o, S = chunk_kda(q, k, v, g.to(torch.float32), beta.to(q.dtype), scale=scale,
                         initial_state=S0, output_final_state=True)
        return o, S
    if g is None:
        g = torch.zeros(q.shape[:3], device=q.device, dtype=torch.float32)
    g = g.to(torch.float32)
    if delta:
        assert beta is not None
        o, S = chunk_gated_delta_rule(q, k, v, g, beta.to(q.dtype), scale=scale,
                                      initial_state=S0, output_final_state=True)
    else:
        o, S = chunk_simple_gla(q, k, v, g, scale=scale,
                                initial_state=S0, output_final_state=True)
    return o, S


# ---------------------------------------------------------------------------
# The mixer
# ---------------------------------------------------------------------------

class GatedDeltaNet(nn.Module, Mixer):
    '''
    Layout per layer (fla's GatedDeltaNet / KimiDeltaAttention, minus
    the bells we do not use):
        q, k = SiLU(conv(W x)), L2-normalised per head; v = SiLU(conv(W x))
        a_t  = exp(-exp(A_log) * softplus(f(x) + dt_bias))   [gate]
               f = W_a x (d -> H) for the scalar gate; for the vector
               gate a low-rank W_a2 W_a1 x (d -> gate_rank -> H dk), with
               A_log per head and dt_bias per channel (fla's KDA gate)
        b_t  = sigmoid(W_b x)                                  [delta]
        o    = RMSNorm_head(scan(q, k, v)) * SiLU(W_g x); out = W_o o
    The a_t parameterisation is Mamba2's: A_log ~ log U(1, 16), dt_bias
    the inverse-softplus of dt ~ logU(1e-3, 1e-1), so initial forgetting
    rates spread over ~3 decades in log space (uniform sigmoid gates start
    every head at the same rate and train slowly out of it).

    Init: N(0, init_std) on all projections, wo shrunk by 1/sqrt(2 *
    layer_count) (Transformer++ convention, same as SoftmaxAttention);
    the conv keeps its default init.

    impl: 'auto' picks fla when installed and the input is on CUDA in
    bf16/fp16, else the torch chunk reference. 'fla' / 'torch' force.

    Parameter sizing at dim d: q, k are d x (H dk); v, gate, o are
    d x (H dv). With H dk = d/2 and H dv = d (the GLA layout) the mixer
    has 4 d^2 params, matching MHA at the same dim.
    '''

    def __init__(
            self,
            dim: int,
            head_count: int,
            key_head_dim: int,
            value_head_dim: int,
            short_conv_size: int | None = 4,
            gate: str | bool = 'vector',
            delta: bool = True,
            gate_rank: int = 64,
            chunk_size: int = 64,
            impl: str = 'auto',
            *,
            init_std: float = 0.02,
            layer_count: int | None = None,
            A_init_range: tuple[float, float] = (1.0, 16.0),
            dt_init_range: tuple[float, float] = (1e-3, 1e-1),
        ):
        super().__init__()
        if isinstance(gate, bool):          # legacy spelling: True = scalar (GDN)
            gate = 'scalar' if gate else 'none'
        assert gate in GATES, gate
        assert dim > 0 and head_count > 0 and key_head_dim > 0 and value_head_dim > 0
        assert impl in ('auto', 'fla', 'torch')
        if impl == 'fla' and not HAS_FLA:
            raise RuntimeError("impl='fla' but fla is not installed")

        self.dim = dim
        self.head_count = head_count
        self.key_head_dim = key_head_dim
        self.value_head_dim = value_head_dim
        self.short_conv_size = short_conv_size
        self.gate = gate
        self.delta = delta
        self.gate_rank = gate_rank
        self.chunk_size = chunk_size
        self.impl = impl
        self.scale = key_head_dim ** -0.5

        H, dk, dv = head_count, key_head_dim, value_head_dim
        self.wq = nn.Linear(dim, H * dk, bias=False)
        self.wk = nn.Linear(dim, H * dk, bias=False)
        self.wv = nn.Linear(dim, H * dv, bias=False)
        self.wg = nn.Linear(dim, H * dv, bias=False)      # output gate
        self.wo = nn.Linear(H * dv, dim, bias=False)
        self.o_norm = RMSNorm(dv)                         # per-head, gamma shared

        if short_conv_size is not None:
            self.conv_q = ShortConv(H * dk, short_conv_size)
            self.conv_k = ShortConv(H * dk, short_conv_size)
            self.conv_v = ShortConv(H * dv, short_conv_size)

        if gate != 'none':
            n_dt = H * dk if gate == 'vector' else H        # one dt per channel / per head
            if gate == 'vector':
                self.wa1 = nn.Linear(dim, gate_rank, bias=False)
                self.wa2 = nn.Linear(gate_rank, H * dk, bias=False)
            else:
                self.wa = nn.Linear(dim, H, bias=False)
            A = torch.empty(H).uniform_(*A_init_range)
            self.A_log = nn.Parameter(A.log())
            lo, hi = math.log(dt_init_range[0]), math.log(dt_init_range[1])
            dt = torch.exp(torch.rand(n_dt) * (hi - lo) + lo).clamp(min=1e-4)
            self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))   # softplus^-1
        if delta:
            self.wb = nn.Linear(dim, H, bias=False)

        for lin in (self.wq, self.wk, self.wv, self.wg):
            nn.init.normal_(lin.weight, std=init_std)
        for name in ('wa', 'wa1', 'wa2', 'wb'):
            if hasattr(self, name):
                nn.init.normal_(getattr(self, name).weight, std=init_std)
        wo_std = init_std / math.sqrt(2 * layer_count) if layer_count else init_std
        nn.init.normal_(self.wo.weight, std=wo_std)

        # recurrent state (fp32) + conv tails; allocated by reset_cache
        self.cache_len = 0
        self.register_buffer('state', torch.zeros(0, 0, 0, 0), persistent=False)
        if short_conv_size is not None:
            self.register_buffer('qp_cache', torch.zeros(0, short_conv_size, H * dk), persistent=False)
            self.register_buffer('kp_cache', torch.zeros(0, short_conv_size, H * dk), persistent=False)
            self.register_buffer('vp_cache', torch.zeros(0, short_conv_size, H * dv), persistent=False)

    # --- pieces -----------------------------------------------------

    @property
    def gate_projections(self) -> list[nn.Parameter]:
        '''The gate/beta matrices (d x H, or the low-rank pair). Skinny
        shapes: a model's param_groups may route them to AdamW instead
        of Muon.'''
        return [getattr(self, n).weight for n in ('wa', 'wa1', 'wa2', 'wb') if hasattr(self, n)]

    @staticmethod
    def _hi(x):
        '''At least fp32 (bf16 -> fp32, fp64 stays fp64).'''
        return x.to(torch.promote_types(x.dtype, torch.float32))

    @classmethod
    def _l2norm(cls, x, eps: float = 1e-6):
        h = cls._hi(x)
        return (h * torch.rsqrt(h.pow(2).sum(-1, keepdim=True) + eps)).to(x.dtype)

    def _gates(self, x):
        '''(g, beta) in fp32 (fp64 for fp64 models): log a_t (or None)
        and b_t (or None).'''
        g = beta = None
        if self.gate == 'scalar':
            g = -self._hi(self.A_log).exp() * F.softplus(self._hi(self.wa(x)) + self._hi(self.dt_bias))
        elif self.gate == 'vector':
            B, L = x.shape[0], x.shape[1]
            f = self._hi(self.wa2(self.wa1(x))) + self._hi(self.dt_bias)            # (B, L, H*dk)
            g = -self._hi(self.A_log).exp()[:, None] * F.softplus(f).view(B, L, self.head_count, self.key_head_dim)
        if self.delta:
            beta = torch.sigmoid(self._hi(self.wb(x)))
        return g, beta

    def _heads(self, qp, kp, vp):
        B, L = qp.shape[0], qp.shape[1]
        H, dk, dv = self.head_count, self.key_head_dim, self.value_head_dim
        q = self._l2norm(F.silu(qp).view(B, L, H, dk))
        k = self._l2norm(F.silu(kp).view(B, L, H, dk))
        v = F.silu(vp).view(B, L, H, dv)
        return q, k, v

    def _pick_impl(self, q) -> str:
        if self.impl != 'auto':
            return self.impl
        ok = HAS_FLA and q.is_cuda and q.dtype in (torch.bfloat16, torch.float16)
        return 'fla' if ok else 'torch'

    def _scan(self, q, k, v, g, beta, S0):
        if self._pick_impl(q) == 'fla':
            return fla_scan(q, k, v, g, beta, S0, scale=self.scale, delta=self.delta)
        return chunk_scan(q, k, v, g, beta, S0, scale=self.scale, delta=self.delta,
                          chunk_size=self.chunk_size)

    def _output(self, o, x):
        B, L = x.shape[0], x.shape[1]
        gate = F.silu(self.wg(x)).view(B, L, self.head_count, self.value_head_dim)
        o = self.o_norm(o.to(gate.dtype)) * gate
        return self.wo(o.reshape(B, L, -1))

    # --- training / stateless -----------------------------------------

    def forward(self, x, is_causal: bool = True):
        assert is_causal, 'linear attention is causal by construction'
        qp, kp, vp = self.wq(x), self.wk(x), self.wv(x)
        if self.short_conv_size is not None:
            qp, kp, vp = self.conv_q(qp), self.conv_k(kp), self.conv_v(vp)
        q, k, v = self._heads(qp, kp, vp)
        g, beta = self._gates(x)
        o, _ = self._scan(q, k, v, g, beta, None)
        return self._output(o, x)

    @torch.no_grad()
    def gate_stats(self, x) -> dict:
        '''Health probe on a small input (a sequence or two): per-head mean
        decay a, effective memory length 1/(1-a), mean write strength b,
        and the RMS of the state after the sequence. fp32 torch path.'''
        qp, kp, vp = self.wq(x), self.wk(x), self.wv(x)
        if self.short_conv_size is not None:
            qp, kp, vp = self.conv_q(qp), self.conv_k(kp), self.conv_v(vp)
        q, k, v = self._heads(qp, kp, vp)
        g, beta = self._gates(x)
        _, S = chunk_scan(q, k, v, g, beta, None, scale=self.scale, delta=self.delta,
                          chunk_size=self.chunk_size)
        out = {'state_rms': S.pow(2).mean().sqrt()}
        if g is not None:
            a = g.exp().mean(dim=(0, 1)).flatten()            # per head, or per (head, channel)
            out['alpha'] = a
            out['mem_len'] = 1.0 / (1.0 - a).clamp(min=1e-6)
        if beta is not None:
            out['beta'] = beta.mean(dim=(0, 1))
        return out

    # --- streaming ----------------------------------------------------

    def reset_cache(self, batch_size: int, max_cache_len: int | None = None):
        '''max_cache_len is accepted for interface parity and ignored:
        the state is fixed-size.'''
        device, dtype = infer_device(self), infer_dtype(self)
        H, dk, dv = self.head_count, self.key_head_dim, self.value_head_dim
        self.cache_len = 0
        sdt = torch.float64 if dtype == torch.float64 else torch.float32
        self.state = torch.zeros(batch_size, H, dk, dv, device=device, dtype=sdt)
        if self.short_conv_size is not None:
            K = self.short_conv_size
            self.qp_cache = torch.zeros(batch_size, K, H * dk, device=device, dtype=dtype)
            self.kp_cache = torch.zeros(batch_size, K, H * dk, device=device, dtype=dtype)
            self.vp_cache = torch.zeros(batch_size, K, H * dv, device=device, dtype=dtype)

    def load_cache(self, cache: dict, max_cache_len: int | None = None):
        self.reset_cache(cache['state'].shape[0], max_cache_len)
        self.cache_len = int(cache['cache_len'])
        self.state.copy_(cache['state'])
        if self.short_conv_size is not None:
            self.qp_cache.copy_(cache['qp_cache'])
            self.kp_cache.copy_(cache['kp_cache'])
            self.vp_cache.copy_(cache['vp_cache'])

    def export_cache(self) -> dict:
        cache = {'state': self.state.clone(), 'cache_len': self.cache_len}
        if self.short_conv_size is not None:
            cache.update(qp_cache=self.qp_cache.clone(), kp_cache=self.kp_cache.clone(),
                         vp_cache=self.vp_cache.clone())
        return cache

    @torch.no_grad()
    def decode_step(self, x):
        '''Consume the next block (B, L, dim), advancing the state. L == 1
        runs one recurrent step; longer blocks run the chunk scan from the
        current state (prefill is a big first block).'''
        B, L = x.shape[0], x.shape[1]
        assert self.state.shape[0] > 0, 'No state allocated. Call reset_cache or load_cache first.'
        assert B == self.state.shape[0], 'Batch size does not match the state. Reset or load cache first.'
        if L == 0:
            return x

        qp, kp, vp = self.wq(x), self.wk(x), self.wv(x)
        if self.short_conv_size is not None:
            K = self.short_conv_size
            qp = torch.cat([self.qp_cache, qp], dim=1)
            kp = torch.cat([self.kp_cache, kp], dim=1)
            vp = torch.cat([self.vp_cache, vp], dim=1)
            self.qp_cache.copy_(qp[:, -K:])
            self.kp_cache.copy_(kp[:, -K:])
            self.vp_cache.copy_(vp[:, -K:])
            qp = self.conv_q.direct_conv(qp)[:, -L:]
            kp = self.conv_k.direct_conv(kp)[:, -L:]
            vp = self.conv_v.direct_conv(vp)[:, -L:]
        q, k, v = self._heads(qp, kp, vp)
        g, beta = self._gates(x)

        if L == 1:
            o, S = recurrent_scan(q, k, v, g, beta, self.state, scale=self.scale, delta=self.delta)
        else:
            o, S = self._scan(q, k, v, g, beta, self.state)
        self.state.copy_(S)
        self.cache_len += L
        return self._output(o, x)
