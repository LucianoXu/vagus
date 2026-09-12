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

try:
    from fla.modules.fused_norm_gate import rms_norm_gated  # type: ignore
    HAS_FLA_NORM_GATE = True
except ImportError:
    rms_norm_gated = None  # type: ignore[assignment]
    HAS_FLA_NORM_GATE = False

GATES = ('none', 'scalar', 'vector')

# The places a GatedDeltaNet layer can hand work to an fla kernel
# instead of doing it in PyTorch. They are separate flags because under
# torch.compile they do not have the same sign, and the difference is
# not which is 'more fused' — it is what inductor would otherwise have
# done with that work:
#     'conv'      fla's Triton causal_conv1d in place of the cuDNN
#                 depthwise call. Worth 1.29x on the layer in eager and
#                 a reproducible 6% LOSS end to end (150.6k tok/s
#                 against 160.1k, five interleaved runs): the custom
#                 autograd Function breaks the graph where the cuDNN
#                 call did not, and the regions either side stop fusing.
#                 Do not turn it on. The conv is worth attacking — it is
#                 the largest non-GEMM item in a compiled step — but
#                 from the other end: see ShortConv's 'shift'.
#     'gate'      the gate activation reaches chunk_kda as the raw
#                 pre-activation, so the activation and its chunk cumsum
#                 happen in one pass. Removes an fp32 (B, L, H, dk)
#                 tensor and the separate chunk_local_cumsum over it —
#                 work inductor cannot remove, because the tensor exists
#                 only to be handed to an opaque kernel. The one stage
#                 that pays: -1.34 GB (5.7%) of peak at no cost in
#                 speed, twice over.
#     'beta'      the sigmoid moves into the same kernel. Small, and
#                 measured neutral (160.0k tokens/s against 160-161k).
#     'l2norm'    fla's use_qk_l2norm_in_kernel is a *separate*
#                 l2norm_fwd launch, not a fusion into the chunk kernel.
#                 Inductor fuses our l2norm into the SiLU that precedes
#                 it, so this trades a fused kernel for a standalone one:
#                 neutral (160.1k).
#     'norm_gate' likewise: RMSNorm x SiLU is one inductor kernel
#                 already. Neutral (161.2k).
# Combinations are worse than their parts — every spec containing 'conv'
# lands at 142-152k against 160-161k without it, and the full set at
# 135.8k — because each custom autograd Function is another break in a
# graph inductor was fusing across.
# 'scan' is the shorthand for the three the scan kernel accepts.
FUSIONS = ('conv', 'l2norm', 'gate', 'beta', 'norm_gate')
_ALIASES = {'scan': ('l2norm', 'gate', 'beta'), 'all': FUSIONS}


def parse_fusions(spec) -> frozenset:
    '''`fused=` accepts True (all), False/None (none), or a subset —
    'gate', 'conv,gate', 'scan' (= l2norm,gate,beta), ['conv', 'gate'].'''
    if spec is True:
        return frozenset(FUSIONS)
    if not spec:
        return frozenset()
    names = [s.strip() for s in spec.split(',')] if isinstance(spec, str) else list(spec)
    out: set[str] = set()
    for n in names:
        if not n:
            continue
        out.update(_ALIASES.get(n, (n,)))
    bad = out - set(FUSIONS)
    assert not bad, f'unknown fusion {sorted(bad)}; pick from {FUSIONS} or {sorted(_ALIASES)}'
    return frozenset(out)


def chunk_scan_flops(dk: int, dv: int, chunk: int, delta: bool) -> float:
    '''Matmul FLOPs per token per head, forward + backward (x3), of the
    chunkwise algorithm the kernels actually run — the honest MFU
    numerator for a linear-attention layer. The recurrence alone (state
    write, delta read-back, query read-out: 2 dk dv each) is what a
    naive count credits; the chunked form adds, per token, the
    intra-chunk products over a C-token chunk: Q K^T and the masked
    attention-times-U read (2C dk + 2C dv), and with the delta rule the
    K K^T gram and the WY solve of w = T (beta K), u = T (beta V)
    (2C dk + 2C dk + 2C dv). At dk 128, dv 256, C 64 that is 1.58x the
    naive count. Softmax layers keep the PaLM 12 d L convention (full
    matrix, not the causal half) so MFU stays comparable with published
    numbers and the SAX runs.'''
    state = (6 if delta else 4) * dk * dv
    intra = 2 * chunk * ((3 * dk + 2 * dv) if delta else (dk + dv))
    return 3.0 * (state + intra)


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


def recurrent_scan(q, k, v, g, beta, S0, *, scale: float, delta: bool, w=None):
    '''Token-by-token recurrence. The definition; everything else must
    match it.

    w: (B, L, H, dk) write vectors, used in place of the delta rule's
    beta_t k_t. The recurrence S <- (I - w_t k_t^T) S + w_t v_t^T is the
    *asymmetric* delta rule (w no longer parallel to k), which Diagonal
    KDN's Kalman gain produces; beta is ignored when w is given. The
    chunkwise forms below do NOT accept it — the compact-WY factorisation
    they use assumes the symmetric write — so an asymmetric write runs
    this scan, at this scan's cost.'''
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
            kS = torch.einsum('bhk,bhkv->bhv', kt, S)            # what S returns for k_t
            if w is None:
                assert beta is not None
                wt = kt * beta[:, t].to(dt)[..., None]
            else:
                wt = w[:, t].to(dt)
            S = S + torch.einsum('bhk,bhv->bhkv', wt, vt - kS)
        else:
            S = S + torch.einsum('bhk,bhv->bhkv', kt, vt)
        outs.append(torch.einsum('bhk,bhkv->bhv', q[:, t], S))
    o = torch.stack(outs, dim=1) if L else q.new_zeros(B, 0, H, dv)
    return o.to(q.dtype), S


def residual_scan(k, v, g, beta, S0, *, delta: bool, w=None):
    '''The delta rule's per-token residual, |v_t - S_{t-1}^T k_t| /
    |v_t| with S_{t-1} already decayed — what the memory did not
    predict about the value it is about to store (the write is beta_t
    times this residual). (B, L, H) fp32. Without the delta rule the
    residual is v_t itself (ratio 1). Same recurrence as recurrent_scan,
    and it takes the same optional explicit write vector w.

    This is the diagnostic behind the capacity reading: run it along a
    stream from a blank state and the residual falls while the memory
    still has room, then plateaus once it is full. On LAX1-340M it falls
    0.92 -> 0.715 over positions 1-128 = d_k and is flat to 2048.'''
    B, L, H, dk = k.shape
    dv = v.shape[-1]
    dt = _compute_dtype(k)
    k, v = k.to(dt), v.to(dt)
    S = torch.zeros(B, H, dk, dv, device=k.device, dtype=dt) if S0 is None else S0.to(dt)
    out = []
    for t in range(L):
        if g is not None:
            a = g[:, t].to(dt).exp()
            S = S * (a[..., None] if a.dim() == 3 else a[..., None, None])
        kt, vt = k[:, t], v[:, t]
        if delta:
            kS = torch.einsum('bhk,bhkv->bhv', kt, S)
            r = vt - kS
            out.append(r.norm(dim=-1) / vt.norm(dim=-1).clamp(min=1e-6))
            if w is None:
                assert beta is not None
                wt = kt * beta[:, t].to(dt)[..., None]
            else:
                wt = w[:, t].to(dt)
            S = S + torch.einsum('bhk,bhv->bhkv', wt, r)
        else:
            out.append(torch.ones(B, H, device=k.device, dtype=dt))
            S = S + torch.einsum('bhk,bhv->bhkv', kt, vt)
    return torch.stack(out, dim=1).float() if L else k.new_zeros(B, 0, H, dtype=torch.float32)


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
    else:
        beta = None                                                       # no erase, no W

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


def fla_scan(q, k, v, g, beta, S0, *, scale: float, delta: bool, chunk_size: int = 64,
             l2norm_in_kernel: bool = False, gate_in_kernel: bool = False,
             beta_in_kernel: bool = False, A_log=None, dt_bias=None,
             disable_recompute: bool = False, lower_bound: float | None = None):
    '''fla chunk kernels. q/k/v must be bf16/fp16 on CUDA; g/beta fp32.
    Semantics identical to recurrent_scan (fla's fused_recurrent
    reference is the same recurrence: decay, then delta write).

    The *_in_kernel flags move work that this module would otherwise do
    in PyTorch into the Triton kernels, which is where it belongs: each
    one is an elementwise pass over a (B, L, H, d) tensor whose only
    cost is memory traffic, and the kernel already has the operand in
    registers.
        l2norm_in_kernel  q, k arrive un-normalised; the kernel L2-norms
                          them (and differentiates through it).
        gate_in_kernel    g arrives as the raw pre-activation f; the
                          kernel computes -exp(A_log) softplus(f +
                          dt_bias) *and its chunk cumsum* in one pass
                          (KDA only; A_log (H,), dt_bias (H*dk,)).
        beta_in_kernel    beta arrives as logits; the kernel takes the
                          sigmoid.
    disable_recompute keeps the intra-chunk activations from the forward
    instead of recomputing them in the backward: memory for time, worth
    it at small model sizes.'''
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
        if gate_in_kernel:
            assert A_log is not None and dt_bias is not None
            kw: dict = dict(use_gate_in_kernel=True, A_log=A_log, dt_bias=dt_bias)
            if lower_bound is not None:
                # bounded decay: the kernel takes the tensor-core path for
                # the intra-chunk diagonal blocks, which the unbounded
                # exponents of the softplus gate cannot use safely
                kw.update(safe_gate=True, lower_bound=lower_bound)
        else:
            g = g.to(torch.float32)
            kw = {}
        o, S = chunk_kda(q, k, v, g, beta if beta_in_kernel else beta.to(q.dtype),
                         scale=scale, initial_state=S0, output_final_state=True,
                         use_qk_l2norm_in_kernel=l2norm_in_kernel,
                         use_beta_sigmoid_in_kernel=beta_in_kernel,
                         disable_recompute=disable_recompute,
                         chunk_size=chunk_size, **kw)
        return o, S
    assert not gate_in_kernel, 'the fused gate activation is a KDA (vector-gate) kernel'
    if g is None:
        g = torch.zeros(q.shape[:3], device=q.device, dtype=torch.float32)
    g = g.to(torch.float32)
    if delta:
        assert beta is not None
        o, S = chunk_gated_delta_rule(q, k, v, g, beta if beta_in_kernel else beta.to(q.dtype),
                                      scale=scale, initial_state=S0, output_final_state=True,
                                      use_qk_l2norm_in_kernel=l2norm_in_kernel,
                                      use_beta_sigmoid_in_kernel=beta_in_kernel,
                                      chunk_size=chunk_size)
    else:
        assert not l2norm_in_kernel and not beta_in_kernel, \
            'simple_gla takes no fused l2norm / beta'
        o, S = chunk_simple_gla(q, k, v, g, scale=scale,
                                initial_state=S0, output_final_state=True,
                                chunk_size=chunk_size)
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

    fused: which of FUSIONS (see the note above them) to hand to an fla
    kernel instead of doing it in PyTorch — True for all, False for
    none, or a subset: 'gate', 'conv,gate', 'scan'. Every one of them is
    the same computation at the same precision (under autocast the
    pre-activations were already bf16 and each kernel does its
    arithmetic in fp32; only the round trip through memory changes), so
    the choice is purely about speed, and it has to be measured against
    torch.compile rather than assumed. Off by default; the torch path
    ignores it.

    conv_impl: how the short conv is written — see ShortConv. 'conv1d'
    is the cuDNN depthwise call, 'shift' the same convolution as K
    shifted multiply-adds, which inductor fuses with the SiLU and L2
    norm around it. Ignored when the 'conv' fusion selects fla's kernel.

    disable_recompute: keep the intra-chunk activations from the forward
    rather than recomputing them in the backward. Memory for time, and
    at 340M there is memory to spend.

    gate_lower_bound: switch the decay to Kimi's bounded form
    lower_bound * sigmoid(exp(A_log) (f + dt_bias)) in [lower_bound, 0)
    instead of -exp(A_log) softplus(...). Vector gate only. With the
    'gate' fusion it also puts the intra-chunk diagonal blocks on the
    tensor cores (fla's safe_gate), which the unbounded exponents of the
    softplus gate cannot use — but that is worth +1.9% end to end
    (184.5k / 185.1k tokens/s against 180.4k / 182.1k, interleaved), not
    the ~5% the eager layer bench suggests: the KDA kernels are only
    ~20% of a compiled step and safe_gate touches part of that. It is a
    different gate family, so it costs a pilot and it breaks the HAX/LAX/
    SAX batch pairing. At 1.9% that trade does not look worth making.
    -5 is Kimi's recommended value. See the dt_bias init: each form
    needs its own inverse.

    Parameter sizing at dim d: q, k are d x (H dk); v, gate, o are
    d x (H dv). With H dk = d/2 and H dv = d (the GLA layout) the mixer
    has 4 d^2 params, matching MHA at the same dim.

    out_proj=False drops wo: forward returns the gated, normalised head
    concat of width out_width = H dv, for a caller that owns the output
    projection (a branch of ParallelMixer).
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
            fused: bool | str | list = False,
            conv_impl: str = 'conv1d',
            disable_recompute: bool = False,
            gate_lower_bound: float | None = None,
            *,
            init_std: float = 0.02,
            layer_count: int | None = None,
            A_init_range: tuple[float, float] = (1.0, 16.0),
            dt_init_range: tuple[float, float] = (1e-3, 1e-1),
            out_proj: bool = True,
        ):
        super().__init__()
        if isinstance(gate, bool):          # legacy spelling: True = scalar (GDN)
            gate = 'scalar' if gate else 'none'
        assert gate in GATES, gate
        assert dim > 0 and head_count > 0 and key_head_dim > 0 and value_head_dim > 0
        assert impl in ('auto', 'fla', 'torch')
        if impl == 'fla' and not HAS_FLA:
            raise RuntimeError("impl='fla' but fla is not installed")
        assert gate_lower_bound is None or gate == 'vector', \
            'the bounded decay is a KDA (vector-gate) parameterisation'
        assert gate_lower_bound is None or -5 <= gate_lower_bound < 0, \
            f'gate_lower_bound must be in [-5, 0), got {gate_lower_bound}'

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
        self.fusions = parse_fusions(fused)
        self.disable_recompute = disable_recompute
        self.gate_lower_bound = gate_lower_bound
        self.scale = key_head_dim ** -0.5
        self.out_proj = out_proj
        self.out_width = head_count * value_head_dim

        H, dk, dv = head_count, key_head_dim, value_head_dim
        self.wq = nn.Linear(dim, H * dk, bias=False)
        self.wk = nn.Linear(dim, H * dk, bias=False)
        self.wv = nn.Linear(dim, H * dv, bias=False)
        self.wg = nn.Linear(dim, H * dv, bias=False)      # output gate
        if out_proj:
            self.wo = nn.Linear(H * dv, dim, bias=False)
        self.o_norm = RMSNorm(dv)                         # per-head, gamma shared

        if short_conv_size is not None:
            # the 'conv' fusion is the fla kernel; conv_impl chooses among
            # the rest ('conv1d' as written, 'shift' as elementwise work)
            ci = 'fla' if 'conv' in self.fusions else conv_impl
            self.conv_q = ShortConv(H * dk, short_conv_size, impl=ci)
            self.conv_k = ShortConv(H * dk, short_conv_size, impl=ci)
            self.conv_v = ShortConv(H * dv, short_conv_size, impl=ci)

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
            # dt_bias is whatever makes the gate's own activation return
            # -dt at f = 0, so both parameterisations start from the same
            # spread of forgetting rates. The inverse differs with the
            # activation, and using the softplus one for the bounded gate
            # is not a small error: logit's argument would be dt/|lb|
            # ~ 1e-3, so sigmoid saturates at 0 and every head starts
            # with no forgetting at all (measured: decay a = 1.0 across
            # every quantile, against a median memory length of 9.7).
            if gate_lower_bound is None:
                bias = dt + torch.log(-torch.expm1(-dt))                  # softplus^-1
            else:
                # match the softplus form at f = 0, where it gives a decay
                # of exp(A_log) * dt (A below is already exp(A_log)):
                #   -lb * sigmoid(A b) = A dt  =>  b = logit(A dt / -lb) / A
                a = A.repeat_interleave(dk)
                p = (a * dt / -gate_lower_bound).clamp(1e-6, 1 - 1e-6)
                bias = torch.logit(p) / a
            self.dt_bias = nn.Parameter(bias)
        if delta:
            self.wb = nn.Linear(dim, H, bias=False)

        for lin in (self.wq, self.wk, self.wv, self.wg):
            nn.init.normal_(lin.weight, std=init_std)
        for name in ('wa', 'wa1', 'wa2', 'wb'):
            if hasattr(self, name):
                nn.init.normal_(getattr(self, name).weight, std=init_std)
        if out_proj:
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

    def _gates(self, x, *, raw_gate: bool = False, raw_beta: bool = False):
        '''(g, beta) in fp32 (fp64 for fp64 models): log a_t (or None)
        and b_t (or None).

        raw_gate / raw_beta return the KDA kernel's inputs instead — the
        gate pre-activation f, shaped (B, L, H, dk), and the beta
        logits, both in x's dtype. The kernel then computes
        -exp(A_log) softplus(f + dt_bias) (fusing the chunk cumsum that
        follows it) and sigmoid(beta) itself, so neither the fp32
        pre-activation nor the fp32 decay is ever written to memory.'''
        g = beta = None
        if self.gate == 'scalar':
            assert not raw_gate, 'the fused gate activation is KDA-only (vector gate)'
            g = -self._hi(self.A_log).exp() * F.softplus(self._hi(self.wa(x)) + self._hi(self.dt_bias))
        elif self.gate == 'vector':
            B, L = x.shape[0], x.shape[1]
            f = self.wa2(self.wa1(x))                                               # (B, L, H*dk)
            if raw_gate:
                g = f.view(B, L, self.head_count, self.key_head_dim)
            elif self.gate_lower_bound is not None:
                f = (self._hi(f) + self._hi(self.dt_bias)).view(
                    B, L, self.head_count, self.key_head_dim)
                g = self.gate_lower_bound * torch.sigmoid(self._hi(self.A_log).exp()[:, None] * f)
            else:
                f = self._hi(f) + self._hi(self.dt_bias)
                g = -self._hi(self.A_log).exp()[:, None] * F.softplus(f).view(
                    B, L, self.head_count, self.key_head_dim)
        if self.delta and hasattr(self, 'wb'):
            beta = self.wb(x) if raw_beta else torch.sigmoid(self._hi(self.wb(x)))
        return g, beta

    def _heads(self, qp, kp, vp, *, l2norm: bool = True, conv: bool = True):
        '''conv -> SiLU -> per-head view, and (unless the kernel will do
        it) the L2 norm on q and k. The activation is handed to the conv
        rather than applied after it so a fused conv kernel can absorb
        it.'''
        B, L = qp.shape[0], qp.shape[1]
        H, dk, dv = self.head_count, self.key_head_dim, self.value_head_dim
        if conv and self.short_conv_size is not None:
            qp, kp, vp = self.conv_q(qp, 'silu'), self.conv_k(kp, 'silu'), self.conv_v(vp, 'silu')
        else:                              # already convolved (decode), or no conv
            qp, kp, vp = F.silu(qp), F.silu(kp), F.silu(vp)
        q, k, v = qp.view(B, L, H, dk), kp.view(B, L, H, dk), vp.view(B, L, H, dv)
        if l2norm:
            q, k = self._l2norm(q), self._l2norm(k)
        return q, k, v

    def _pick_impl(self, q) -> str:
        if self.impl != 'auto':
            return self.impl
        ok = HAS_FLA and q.is_cuda and q.dtype in (torch.bfloat16, torch.float16)
        return 'fla' if ok else 'torch'

    def _active_fusions(self, qp) -> frozenset:
        '''The fusions this call actually runs: the configured set minus
        the ones this layer's kernel cannot take. Decided on a
        projection output, the tensor whose device and dtype the kernels
        will actually see — under FSDP's mixed precision the residual
        stream reaching forward() may still be fp32. Off the fla path
        (the torch reference, cpu/fp32/fp64) nothing is fused.

        The one filter that is not about the device: without the delta
        rule the scan is chunk_simple_gla, which has no beta and takes
        no fused L2 norm, and only the vector-gate kernel (chunk_kda)
        computes the gate activation itself.'''
        if self._pick_impl(qp) != 'fla':
            return frozenset()
        on = self.fusions
        if not self.delta:
            on = on - {'l2norm', 'beta'}
        if self.gate != 'vector':
            on = on - {'gate'}
        return on

    def _write(self, x, k, g, beta):
        '''(beta, w) — the delta rule's write for this block. The family's
        own write is the symmetric one, beta_t k_t, so w is None and the
        chunkwise kernels apply; KalmanDeltaNet overrides this to put a
        Kalman gain in beta's place (Isotropic: still a scalar per head,
        so nothing downstream changes) or to return an explicit
        asymmetric w (Diagonal).'''
        return beta, None

    def _scan(self, q, k, v, g, beta, S0, *, fusions=frozenset(), w=None):
        if w is not None:
            # asymmetric write: no chunkwise form (see recurrent_scan)
            assert self.delta, 'an explicit write vector needs the delta rule'
            return recurrent_scan(q, k, v, g, beta, S0, scale=self.scale,
                                  delta=True, w=w)
        if self._pick_impl(q) == 'fla':
            return fla_scan(q, k, v, g, beta, S0, scale=self.scale, delta=self.delta,
                            chunk_size=self.chunk_size,
                            l2norm_in_kernel='l2norm' in fusions,
                            gate_in_kernel='gate' in fusions,
                            beta_in_kernel='beta' in fusions,
                            A_log=self.A_log if self.gate != 'none' else None,
                            dt_bias=self.dt_bias if self.gate != 'none' else None,
                            disable_recompute=self.disable_recompute,
                            lower_bound=self.gate_lower_bound)
        return chunk_scan(q, k, v, g, beta, S0, scale=self.scale, delta=self.delta,
                          chunk_size=self.chunk_size)

    def _output(self, o, x):
        B, L = x.shape[0], x.shape[1]
        gate = self.wg(x).view(B, L, self.head_count, self.value_head_dim)
        if 'norm_gate' in self.fusions and HAS_FLA_NORM_GATE and o.is_cuda and \
                o.dtype in (torch.bfloat16, torch.float16):
            # one kernel for RMSNorm(o) * SiLU(gate) over the head dim
            assert rms_norm_gated is not None
            o = rms_norm_gated(o.to(gate.dtype), gate, self.o_norm.gamma, None,
                               activation='swish', eps=self.o_norm.eps).reshape(B, L, -1)
        else:
            o = (self.o_norm(o.to(gate.dtype)) * F.silu(gate)).reshape(B, L, -1)
        return self.wo(o) if self.out_proj else o

    # --- training / stateless -----------------------------------------

    def forward(self, x, is_causal: bool = True):
        assert is_causal, 'linear attention is causal by construction'
        qp, kp, vp = self.wq(x), self.wk(x), self.wv(x)
        on = self._active_fusions(qp)
        q, k, v = self._heads(qp, kp, vp, l2norm='l2norm' not in on)
        g, beta = self._gates(x, raw_gate='gate' in on, raw_beta='beta' in on)
        beta, w = self._write(x, k, g, beta)
        o, _ = self._scan(q, k, v, g, beta, None, fusions=on, w=w)
        return self._output(o, x)

    @torch.no_grad()
    def residuals(self, x) -> torch.Tensor:
        '''(B, L, H) delta-rule residual ratios along a fresh stream x
        (B, L, dim) — the layer's surprise at each token. Torch path;
        no gradient.'''
        qp, kp, vp = self.wq(x), self.wk(x), self.wv(x)
        _, k, v = self._heads(qp, kp, vp)
        g, beta = self._gates(x)
        beta, w = self._write(x, k, g, beta)
        return residual_scan(k, v, g, beta, None, delta=self.delta, w=w)

    @torch.no_grad()
    def gate_stats(self, x) -> dict:
        '''Health probe on a small input (a sequence or two): per-head mean
        decay a, effective memory length 1/(1-a), mean write strength b,
        and the RMS of the state after the sequence. fp32 torch path.

        Goes through _write, like forward and decode_step: a subclass may
        put something other than sigmoid(w_beta . x) in beta's place, and
        for KalmanDeltaNet _gates alone returns beta=None (the Kalman gain
        needs the keys, so it is formed in _write). Reading the gates
        without the hook asserted inside chunk_scan_vec and killed the LAX2
        smoke at its first slow-metric step (job 30199344).'''
        q, k, v = self._heads(self.wq(x), self.wk(x), self.wv(x))
        g, beta = self._gates(x)
        beta, w = self._write(x, k, g, beta)
        if w is None:
            _, S = chunk_scan(q, k, v, g, beta, None, scale=self.scale, delta=self.delta,
                              chunk_size=self.chunk_size)
        else:
            _, S = recurrent_scan(q, k, v, g, beta, None, scale=self.scale,
                                  delta=self.delta, w=w)
        out = {'state_rms': S.pow(2).mean().sqrt()}
        if g is not None:
            a = g.exp().mean(dim=(0, 1)).flatten()            # per head, or per (head, channel)
            out['alpha'] = a
            out['mem_len'] = 1.0 / (1.0 - a).clamp(min=1e-6)
        # write strength is |w_t| per head. For the symmetric write
        # w = beta k with |k| = 1, so this IS beta and the metric keeps its
        # meaning across the family; an asymmetric Kalman gain has no beta
        # but the same quantity is defined.
        if w is not None:
            out['beta'] = w.norm(dim=-1).mean(dim=(0, 1))
        elif beta is not None:
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
        q, k, v = self._heads(qp, kp, vp, conv=False)
        g, beta = self._gates(x)
        beta, w = self._write(x, k, g, beta)

        if L == 1 or w is not None:
            o, S = recurrent_scan(q, k, v, g, beta, self.state, scale=self.scale,
                                  delta=self.delta, w=w)
        else:
            o, S = self._scan(q, k, v, g, beta, self.state)
        self.state.copy_(S)
        self.cache_len += L
        return self._output(o, x)
