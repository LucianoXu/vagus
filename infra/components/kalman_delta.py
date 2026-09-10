# Kalman Delta Networks (KDN, arXiv:2609.07816) — the delta rule's write
# strength replaced by a Kalman gain.
#
# Why: in DeltaNet / Gated DeltaNet / KDA the write strength is
# beta_t = sigmoid(w_beta . x_t) — a function of the current token alone.
# It cannot see the state, so the layer has no way to tell a large
# residual that should revise a tentative association from one that
# should be discounted because the association has been confirmed many
# times. Measured on our own trained LAX1-340M (2026-09-10): beta is
# near-saturated (mean 0.80) and carries no information about the
# residual — corr(beta, |v - S^T k|/|v|) = -0.031 over 46k tokens, and
# the top 10% of tokens by surprise take 15.0% of the write mass against
# a uniform 10%. The write gate is doing nothing selective.
#
# KDN's fix is to read the memory as a linear-Gaussian state-space model
# and use the Kalman gain, which weights each write by accumulated
# evidence. The trick that keeps it parallel is that the gain does not
# need S: it needs only an uncertainty summary with its own cheap
# recurrence, computable *before* the memory scan. The memory update
# then stays an input-only affine scan.
#
#   Isotropic KDN — one precision scalar c per head:
#       a_t     = mean_i(alpha_ti^2)              trace-projected decay
#       bhat_t  = a_t / c_{t-1} + omega_t         predictive variance
#       beta_t  = bhat_t / (r_t + bhat_t |k_t|^2)
#       c_t     = 1 / bhat_t + s |k_t|^2 / (dk r_t)
#     beta_t is still ONE SCALAR PER HEAD, so the memory recurrence is
#     bit-for-bit the delta rule we already run — the chunkwise kernels,
#     fla included, apply unchanged. This is the variant we can train.
#
#   Diagonal KDN — one covariance scalar per key channel:
#       phat_t  = alpha_t^2 * p_{t-1} + omega_t
#       kappa_t = phat_t * k_t / (r_t + sum_i phat_ti k_ti^2)
#       p_t     = phat_t / (1 + s k_t^2 / r_t * phat_t)
#     kappa_t is a VECTOR and is not parallel to k_t, so the write
#     (I - kappa k^T) S + kappa v^T is the asymmetric delta rule. The
#     compact-WY factorisation the chunk kernels use assumes the
#     symmetric write, so on our path Diagonal KDN runs the token
#     recurrence: correct, and fast enough for eval and tests, but not
#     for pretraining until an asymmetric chunk kernel exists. Upstream
#     ships a Triton one (github.com/ngocbh/kalman-delta-networks,
#     kdn_ops/diag_kdn_chunk.py); porting it is the open work.
#
# Both uncertainty recurrences are Mobius maps x -> (A x + B) / (C x + D)
# with A, B, C, D > 0, so a prefix product of 2x2 matrices computes them
# at logarithmic depth with no cancellation (mobius_scan below).
#
# The decay alpha is KDA's exactly (per-key-channel vector gate), so a
# KDN layer is a KDA layer with w_beta swapped for the noise projections.
# Parameterisation follows the reference implementation:
#     omega_t = omega_min + softplus(...)   process noise
#     r_t     = r_min     + softplus(...)   observation noise, per head
#     c_0     = 0.1 + softplus(param)       per head, init to 1.0
#     s (info_scale) defaults to dk
# Isotropic puts omega on a per-head projection; Diagonal puts it on a
# low-rank per-channel pair whose second factor starts at zero, so every
# channel starts at omega = omega_min + softplus(0).

import math

import torch
import torch.nn.functional as F
from torch import nn

from .linear_attention import GatedDeltaNet

KALMAN = ('iso', 'diag')

OMEGA_MIN = 0.05
R_MIN = 0.05
PRECISION_FLOOR = 0.1
PRECISION_INIT = 1.0


def mobius_scan(A, B, C, D, x0, *, impl: str = 'parallel'):
    '''x_t = (A_t x_{t-1} + B_t) / (C_t x_{t-1} + D_t) along dim 1.

    A..D: (B, L, *rest) non-negative, x0: (B, *rest); returns (B, L, *rest).
    A Mobius map is the action of [[A, B], [C, D]] on the homogeneous pair
    (n, d) with x = n / d, so composing them is a matrix product and the
    scan is a prefix product — associative, log depth. Scaling a matrix
    leaves the map alone, which is what makes the renormalisation below
    free; with every entry non-negative there is no cancellation to lose.
    'recurrent' is the definition, kept as the test oracle.'''
    L = A.shape[1]
    if impl == 'recurrent':
        x, out = x0, []
        for t in range(L):
            x = (A[:, t] * x + B[:, t]) / (C[:, t] * x + D[:, t])
            out.append(x)
        return torch.stack(out, dim=1)
    assert impl == 'parallel', impl
    M = torch.stack([torch.stack([A, B], -1), torch.stack([C, D], -1)], -2)
    stride = 1
    while stride < L:
        M = torch.cat([M[:, :stride], M[:, stride:] @ M[:, :-stride]], dim=1)
        M = M / M.abs().amax(dim=(-2, -1), keepdim=True).clamp(min=1e-30)
        stride *= 2
    x0 = x0[:, None]
    return (M[..., 0, 0] * x0 + M[..., 0, 1]) / (M[..., 1, 0] * x0 + M[..., 1, 1])


def kalman_gain_iso(k, alpha, omega, r, c0, *, info_scale: float, impl: str = 'parallel'):
    '''Isotropic KDN. k: (B, L, H, dk), alpha (the decay a_t, not its log):
    (B, L, H, dk), omega / r: (B, L, H), c0: (B, H) entry precision.
    Returns (beta (B, L, H), c_final (B, H)).'''
    dk = k.shape[-1]
    a = alpha.square().mean(-1)                                  # trace projection
    kn2 = k.square().sum(-1)
    m = (info_scale / dk) * kn2 / r                              # posterior information
    c = mobius_scan(1.0 + m * omega, m * a, omega, a, c0, impl=impl)
    c_prev = torch.cat([c0[:, None], c[:, :-1]], dim=1)          # bhat uses c_{t-1}
    bhat = a / c_prev + omega
    return bhat / (r + bhat * kn2), c[:, -1]


def kalman_gain_diag(k, alpha, omega, r, p0, *, info_scale: float, impl: str = 'parallel'):
    '''Diagonal KDN. omega: (B, L, H, dk), r: (B, L, H), p0: (B, H, dk)
    entry covariance. Returns (kappa (B, L, H, dk), p_final (B, H, dk)).'''
    a2 = alpha.square()
    w = info_scale * k.square() / r[..., None]
    p = mobius_scan(a2, omega, w * a2, 1.0 + w * omega, p0, impl=impl)
    p_prev = torch.cat([p0[:, None], p[:, :-1]], dim=1)
    phat = a2 * p_prev + omega
    kappa = phat * k / (r[..., None] + (phat * k.square()).sum(-1, keepdim=True))
    return kappa, p[:, -1]


class KalmanDeltaNet(GatedDeltaNet):
    '''KDA with the Kalman gain in beta's place. Everything else — the
    short conv, the L2-normalised q/k, the per-channel decay, the gated
    RMSNorm output, the recurrent state and its cache — is the parent's.

    The extra recurrent state is the uncertainty: a precision scalar per
    head (iso) or a covariance vector per key channel (diag). It is
    carried in the cache alongside the memory, so streaming decode and a
    stateless forward agree.'''

    def __init__(self, *args, kalman: str = 'iso', omega_min: float = OMEGA_MIN,
                 r_min: float = R_MIN, info_scale: float | None = None,
                 gain_impl: str = 'parallel', **kw):
        assert kalman in KALMAN, f'kalman must be one of {KALMAN}, got {kalman!r}'
        kw.setdefault('gate', 'vector')
        assert kw['gate'] == 'vector', 'KDN needs KDA\'s per-channel decay'
        assert kw.get('delta', True), 'KDN is a delta-rule write'
        kw['delta'] = True
        super().__init__(*args, **kw)

        self.kalman = kalman
        self.omega_min = omega_min
        self.r_min = r_min
        self.gain_impl = gain_impl
        H, dk = self.head_count, self.key_head_dim
        self.info_scale = float(dk if info_scale is None else info_scale)

        del self.wb                                   # beta is not a projection any more
        self.wr = nn.Linear(self.dim, H, bias=True)
        if kalman == 'iso':
            self.womega = nn.Linear(self.dim, H, bias=True)
            nn.init.normal_(self.womega.weight, std=0.02)
        else:
            # low-rank per-channel process noise; the second factor starts
            # at zero so every channel begins at omega_min + softplus(0)
            self.womega1 = nn.Linear(self.dim, self.gate_rank, bias=False)
            self.womega2 = nn.Linear(self.gate_rank, H * dk, bias=True)
            nn.init.normal_(self.womega1.weight, std=0.02)
            nn.init.zeros_(self.womega2.weight)
            nn.init.zeros_(self.womega2.bias)
        nn.init.normal_(self.wr.weight, std=0.02)
        self.c0_param = nn.Parameter(torch.full(
            (H,), math.log(math.expm1(PRECISION_INIT - PRECISION_FLOOR))))

        self._entry_gain = None                       # threaded by decode_step only
        self.register_buffer('gain_state', torch.zeros(0, 0), persistent=False)

    @property
    def gate_projections(self) -> list[nn.Parameter]:
        '''The skinny gate / noise matrices, for a model that routes them
        to AdamW instead of Muon (as GatedDeltaNet does for wb).'''
        names = ('wa', 'wa1', 'wa2', 'wr', 'womega', 'womega1', 'womega2')
        return [getattr(self, n).weight for n in names if hasattr(self, n)]

    # --- the gain --------------------------------------------------------

    def _entry_precision(self, B: int, device, dtype) -> torch.Tensor:
        '''c_0 per head, broadcast over the batch (iso), or its reciprocal
        as the entry covariance p_0 (diag).'''
        c0 = (PRECISION_FLOOR + F.softplus(self.c0_param.float())).to(device=device, dtype=dtype)
        c0 = c0[None].expand(B, -1)
        if self.kalman == 'iso':
            return c0
        return c0[..., None].expand(-1, -1, self.key_head_dim).reciprocal()

    def _gates(self, x, *, raw_gate: bool = False, raw_beta: bool = False):
        '''Only the decay. There is no beta projection to evaluate — the
        write strength is the Kalman gain, and _write computes it once the
        keys exist (it needs |k_t|).'''
        assert not raw_beta, 'the beta fusion is dropped for KDN (see _active_fusions)'
        g, _ = super()._gates(x, raw_gate=raw_gate, raw_beta=False)
        return g, None

    def _noise(self, x, B: int, L: int):
        r = self.r_min + F.softplus(self._hi(self.wr(x)))
        if self.kalman == 'iso':
            omega = self.omega_min + F.softplus(self._hi(self.womega(x)))
        else:
            raw = self.womega2(self.womega1(x)).view(B, L, self.head_count, self.key_head_dim)
            omega = self.omega_min + F.softplus(self._hi(raw))
        return omega, r

    def _write(self, x, k, g, beta):
        '''Kalman gain in place of the parent's sigmoid beta. g is log a_t;
        the gains need a_t itself, and fp32 throughout (the uncertainty
        recurrence is a ratio of growing products).'''
        assert g is not None, 'KDN needs the decay; the raw-gate fusion is not supported'
        B, L = x.shape[0], x.shape[1]
        alpha = self._hi(g).exp()
        omega, r = self._noise(x, B, L)
        kf = self._hi(k)
        entry = self._entry_gain
        if entry is None:
            entry = self._entry_precision(B, x.device, alpha.dtype)
        if self.kalman == 'iso':
            beta, final = kalman_gain_iso(kf, alpha, omega, r, entry,
                                          info_scale=self.info_scale, impl=self.gain_impl)
            w = None
        else:
            w, final = kalman_gain_diag(kf, alpha, omega, r, entry,
                                        info_scale=self.info_scale, impl=self.gain_impl)
            beta = None
        if self._entry_gain is not None:
            self.gain_state.copy_(final)
        return beta, w

    def _active_fusions(self, qp) -> frozenset:
        '''The beta fusion computes sigmoid(beta_logits) inside the kernel;
        our beta is a Kalman gain, already in (0, 1) for L2-normalised keys
        but not a sigmoid of anything. Drop it; the rest still apply.'''
        return super()._active_fusions(qp) - {'beta'}

    # --- streaming: the uncertainty rides along with the memory ----------

    def reset_cache(self, batch_size: int, max_cache_len: int | None = None):
        super().reset_cache(batch_size, max_cache_len)
        dev = self.state.device
        shape = (batch_size, self.head_count)
        if self.kalman == 'diag':
            shape = shape + (self.key_head_dim,)
        self.gain_state = torch.zeros(*shape, device=dev, dtype=torch.float32)
        self.gain_state.copy_(self._entry_precision(batch_size, dev, torch.float32))

    def load_cache(self, cache: dict, max_cache_len: int | None = None):
        super().load_cache(cache, max_cache_len)
        self.gain_state.copy_(cache['gain_state'])

    def export_cache(self) -> dict:
        return {**super().export_cache(), 'gain_state': self.gain_state.clone()}

    @torch.no_grad()
    def decode_step(self, x):
        '''The parent's step, with the uncertainty state threaded into
        _write and back out. _entry_gain is the only channel for it: the
        hook's signature is shared with the plain delta rule, which has no
        such state.'''
        if x.shape[1] == 0:
            return x
        self._entry_gain = self.gain_state.clone()
        try:
            return super().decode_step(x)
        finally:
            self._entry_gain = None
