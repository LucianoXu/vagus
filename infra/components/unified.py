# TRIAGE-family streaming block (naming registered 2026-09-01, engram
# PLANNING.md): 'triage' = the dynamic management algorithm; this module
# implements the evict+demote sub-family; 'full-triage' adds merge (MRC
# is the merge sub-policy's ancestor name). Module path stays
# `unified.py` so in-flight jobs' imports never break.
#
# Unified-block streaming: atom table + moment pools (P=0, P=1), managed
# by the distortion calculus (engram theory §4/§5′/§6(a′)/§7′c′),
# runnable frozen (inference) or differentiably (management-aware
# training).
#
# v2 (2026-09-01, theory §6(a′) + §5′-7 + §5′-8) — the two derived
# changes over v1:
#
#   A. Demotion pricing uses the POSITION-FACTORIZED ensemble instead of
#      the near-window transported ring. Content distribution: window
#      queries un-rotated to the 0-frame -> per-band mean m_b and
#      band-isotropic variance s_b^2. Offset distribution: exponential
#      discount pi(u') ~ exp(-lam u') from the decision time — the
#      position integral is arithmetic: per-band coherence
#      g(w) = lam / (lam + i w), |g| = lam/sqrt(lam^2 + w^2), and each
#      atom's logit mean/variance under the joint measure is a
#      trigonometric closed form (small fixed 32x32 band matrices).
#      The eviction score is UNCHANGED (v1 transported-linear MC over
#      the ring; E1-validated) — variable isolation.
#
#   B. Pool writes are the L^2(Q) PROJECTION (Pi_1 e^x =
#      e^{mu+sigma^2/2}(1 + (x - mu)), Stein), not the Taylor jet:
#      mass factor e^{sigma^2/2} makes the pooled partition-function
#      contribution mean-exact under Q (the v1 harm was mean-level Z
#      pollution), and the stored slope is the per-band coherence-damped
#      key gamma_b * k_b. A P=0 tier (only t0/T0; phi = c e^{mu +
#      sigma^2/2} constant — always positive, zero decoherence) is the
#      exit for atoms whose damped slope is degenerate.
#
# Exit rule per atom: argmin over {evict, demote} in one log-score
# currency; a demoted atom lands in P=1 when its damped slope still
# captures variance (sigma_cap^2 >= slope_eps), else P=0 (5′-8: sinks'
# high-frequency keys damp to near-zero slope — P=0 makes it explicit
# and removes denominator risk). Implementation choice, recorded: with
# the literal residual formulas the P=1 residual is never above P=0
# (an optimal slope cannot hurt in L^2), so the P0/P1 branch follows
# 5′-8's slope-degeneracy argument rather than a score comparison.
#
# Residual formulas (per atom, centered; log-domain in code):
#   P=1 (offset-averaged per-offset projection residual):
#       e^{2 mu + 2V + sc^2} (e^{sc^2} - 1 - sc^2)
#   P=0 (full-measure Var(e^delta)):
#       e^{2 mu + st^2} (e^{st^2} - 1),   st^2 = sc^2 + V
#   with sc^2 = content variance (band-isotropic, offset-invariant),
#   V = variance of the offset-oscillating mean (decoherence), both
#   from the closed forms above. Scores multiply the 5′-3 structure
#   (c/Z̄)^2 ||v - f(q)||^2 and compare to the eviction MC score on a
#   common log axis.
#
# Everything else (explicit-denominator readout, signed-pool guardrail,
# hard selection with differentiable state, per-cell checkpointed BPTT)
# is v1 unchanged; scope: SAX subset (MHA, v_dim_mult=1, no short conv).

import math
from dataclasses import dataclass

import torch
from torch.utils.checkpoint import checkpoint

from .attention import SoftmaxAttention
from .pos_embed import RoPE


@dataclass
class ManageCfg:
    block_len: int = 256      # decisions between blocks of this many tokens
    budget: int = 512         # max alive atoms per (batch, kv-head)
    ring_window: int = 32     # recent queries kept for the score ensemble
    demote: bool = True       # enable the type-change exits (P=0/P=1 pool)
    lam: float = 1.0 / 1024   # offset-discount rate of the position measure
    slope_eps: float = 1e-3   # damped-slope variance below this -> P=0
    # Pool aging (v3, theory (c″)(iv)): 'static' = v2, slope damped once
    # at write by gamma_b; 'stepped' = slope written undamped, then t1/T1
    # decay per streaming block by exp(-lam_b * block), lam_b =
    # lam * ln(1/gamma_b) — cumulative decay matches the static damping
    # at the mean query horizon 1/lam. t0/T0 (and the P=0 tier) never
    # decay: constants carry no phase. Mutually exclusive switch; the
    # static branch is the untouched v2 path.
    pool_gate: str = 'static'
    # v4 (op-point delta verdict, 2026-09-01): pool numerator/denominator
    # SLOPES read through the damped-key Gram's whitener,
    # (t1,T1) -> tau (G + eps*tau*I)^-1 (t1,T1), tau = tr(G)/d — the
    # DeltaNet/RLS fixed point on the pooled subtable; isotropic G
    # recovers Hebbian exactly. Mass column (t0/T0) and the guardrail
    # untouched. 'hebbian' = v2/v3 behavior.
    pool_write: str = 'hebbian'   # 'hebbian' | 'delta'
    eps_z: float = 1e-4       # denominator guardrail threshold
    mu_clamp: float = 25.0    # cap on the centering logit inside exp()


@dataclass
class LayerState:
    '''Per-layer streaming state. Functional: block steps return new
    instances; tensors are never mutated in place (autograd safety).'''
    k: torch.Tensor        # (B, H, T, Dh) post-RoPE keys, absolute positions
    v: torch.Tensor        # (B, H, T, Dv)
    logc: torch.Tensor     # (B, H, T) fp32 log-mass
    alive: torch.Tensor    # (B, H, T) bool
    pos: torch.Tensor      # (T,) int64 absolute position of each atom
    t0: torch.Tensor       # (B, H)        pool constants (P=0 and P=1)
    t1: torch.Tensor       # (B, H, Dh)    pool: damped-key first moment
    T0: torch.Tensor       # (B, H, Dv)
    T1: torch.Tensor       # (B, H, Dh, Dv)
    Gk: torch.Tensor       # (B, H, Dh, Dh) damped-key Gram (delta mode)
    ring_q: torch.Tensor   # (B, H, W, Dh) fp32 post-RoPE recent queries
    ring_pos: torch.Tensor # (W,) int64 absolute positions of ring queries

    def tensors(self) -> tuple:
        return (self.k, self.v, self.logc, self.alive, self.pos, self.t0,
                self.t1, self.T0, self.T1, self.Gk, self.ring_q,
                self.ring_pos)

    @staticmethod
    def from_tensors(ts) -> 'LayerState':
        return LayerState(*ts)


def init_state(att: SoftmaxAttention, batch: int, device, dtype,
               mcfg: ManageCfg) -> LayerState:
    H = att.kv_head_count
    Dh = att.dim // att.head_count
    Dv = Dh * att.v_dim_mult
    z = lambda *s, dt=dtype: torch.zeros(*s, device=device, dtype=dt)
    return LayerState(
        k=z(batch, H, 0, Dh), v=z(batch, H, 0, Dv),
        logc=z(batch, H, 0, dt=torch.float32),
        alive=torch.zeros(batch, H, 0, device=device, dtype=torch.bool),
        pos=torch.zeros(0, device=device, dtype=torch.int64),
        t0=z(batch, H, dt=torch.float32), t1=z(batch, H, Dh, dt=torch.float32),
        T0=z(batch, H, Dv, dt=torch.float32),
        T1=z(batch, H, Dh, Dv, dt=torch.float32),
        Gk=z(batch, H, Dh, Dh, dt=torch.float32),
        ring_q=z(batch, H, 0, Dh, dt=torch.float32),
        ring_pos=torch.zeros(0, device=device, dtype=torch.int64),
    )


class Health:
    '''Accumulates stream-health counters across blocks/layers.'''
    def __init__(self):
        self.z_fallback = 0.0   # fraction of (query, head) readouts that
        self.z_events = 0       # tripped the denominator guardrail
        self.mu_clamped = 0.0   # fraction of demoted atoms with clamped mu
        self.mu_events = 0
        self.demoted_p0 = 0     # atoms demoted into the P=0 tier
        self.demoted_p1 = 0     # atoms demoted into the P=1 tier
        self.evicted = 0
        self.pool_w_max = 0.0   # largest pooled write weight seen (forward
        self.pool_t1_max = 0.0  # signature of the v1 amplifier; cheap
                                # stand-in for a pool-path gnorm split)

    def as_dict(self) -> dict:
        d = {}
        if self.z_events:
            d['unified_z_fallback'] = self.z_fallback / self.z_events
        if self.mu_events:
            d['unified_mu_clamped'] = self.mu_clamped / self.mu_events
        d['unified_demoted_p0'] = float(self.demoted_p0)
        d['unified_demoted_p1'] = float(self.demoted_p1)
        d['unified_evicted'] = float(self.evicted)
        if self.pool_w_max:
            d['unified_pool_w_max'] = self.pool_w_max
            d['unified_pool_t1_max'] = self.pool_t1_max
        return d


# --------------------------------------------------------------------------
# Band algebra (RoPE frequencies, complex views, coherence factors)
# --------------------------------------------------------------------------

def _band_freqs(rope: RoPE, device) -> torch.Tensor:
    '''omega_b = base^(-b/arm_dim), radians per token, (arm_dim,).'''
    idx = torch.arange(rope.arm_dim, device=device, dtype=torch.float64)
    return torch.pow(torch.tensor(float(rope.base), device=device,
                                  dtype=torch.float64), -idx / rope.arm_dim)


def _as_complex(x: torch.Tensor) -> torch.Tensor:
    '''Interleaved pairs (..., Dh) -> complex (..., Dh/2), matching the
    RoPE convention (pair (x0, x1) rotates as (x0 + i x1) e^{i phi}).'''
    y = x.float().reshape(*x.shape[:-1], x.shape[-1] // 2, 2)
    return torch.complex(y[..., 0], y[..., 1])


def _coherence(lam: float, omega: torch.Tensor) -> torch.Tensor:
    '''g(w) = E_{s~Exp(lam)} e^{-i w s} = lam / (lam + i w). complex128.'''
    return lam / torch.complex(torch.full_like(omega, lam), omega)


def _band_decay(rope: RoPE, lam: float, tokens: int, device) -> torch.Tensor:
    '''Per-band stepped-gate decay over `tokens` positions, interleaved
    to (Dh,): exp(-lam_b * tokens) with lam_b = lam * ln(1/gamma_b),
    i.e. gamma_b ** (lam * tokens). Semigroup in `tokens` by
    construction; at tokens = 1/lam the cumulative decay equals the
    static damping gamma_b.'''
    omega = _band_freqs(rope, device)
    gamma = _coherence(lam, omega).abs()              # (nb,) float64
    dec = gamma.pow(lam * tokens).float()
    return dec.repeat_interleave(2)


def _unrotate_ring(rope: RoPE, ring_q: torch.Tensor,
                   ring_pos: torch.Tensor) -> torch.Tensor:
    '''Rotate ring queries back to the 0-frame: multiply band b of the
    query at position p by e^{-i omega_b p}. Returns complex (B,H,W,Dh/2).'''
    rope.prepare_m(int(ring_pos.max().item()) + 1)
    mcos = rope.mcos.float()[ring_pos]          # (W, Dh/2)
    msin = rope.msin.float()[ring_pos]
    phase = torch.complex(mcos, -msin)          # e^{-i omega p}
    return _as_complex(ring_q) * phase


def _measure_stats(rope: RoPE, st: LayerState, t_dec: int, lam: float):
    '''Per-atom logit statistics under the position-factorized measure
    (theory §6(a′)), band-isotropic content model. Returns fp32 (B,H,T):
      mu      — joint-measure mean logit
      var_c   — content variance (offset-invariant under isotropy)
      V       — offset variance of the oscillating mean (decoherence)
      cap     — variance captured by the gamma-damped slope
    All include the 1/sqrt(Dh) logit scaling.'''
    B, H, T, Dh = st.k.shape
    beta = 1.0 / math.sqrt(Dh)
    dev = st.k.device

    u = _unrotate_ring(rope, st.ring_q, st.ring_pos)      # (B,H,W,nb) complex
    m = u.mean(dim=2)                                     # (B,H,nb)
    s2 = 0.5 * (u - m.unsqueeze(2)).abs().pow(2).mean(dim=2)  # (B,H,nb)

    omega = _band_freqs(rope, dev)                        # (nb,) float64
    g = _coherence(lam, omega)                            # (nb,) complex128
    gamma2 = g.abs().pow(2).float()                       # (nb,)

    z = _as_complex(st.k)                                 # (B,H,T,nb)
    zabs2 = z.abs().pow(2)                                # (B,H,T,nb)
    var_c = beta ** 2 * torch.einsum('bhn,bhtn->bht', s2, zabs2)
    cap = beta ** 2 * torch.einsum('bhn,bhtn->bht', s2, zabs2 * gamma2)

    # c_b = conj(m_b) z_b e^{-i omega_b t_dec}: the deterministic decision-
    # time phase; the atom's own e^{+i omega t_j} sits inside z, so the
    # relative phase is -omega * age as required.
    phase_dec = torch.exp(torch.complex(
        torch.zeros_like(omega), -omega * float(t_dec))).to(torch.complex64)
    C = m.conj().unsqueeze(2) * z * phase_dec             # (B,H,T,nb)

    gc = g.to(torch.complex64)
    mu = beta * torch.einsum('bhtn,n->bht', C, gc).real

    om = omega.unsqueeze(0)
    Gm = _coherence(lam, om - om.T).to(torch.complex64)   # g(w_b - w_b')
    Gp = _coherence(lam, om + om.T).to(torch.complex64)   # g(w_b + w_b')
    e2 = 0.5 * beta ** 2 * (
        torch.einsum('bhtn,nm,bhtm->bht', C, Gp, C)
        + torch.einsum('bhtn,nm,bhtm->bht', C, Gm, C.conj())).real
    V = (e2 - mu.pow(2)).clamp(min=0)
    return mu, var_c, V, cap


# --------------------------------------------------------------------------
# Readout (unchanged from v1)
# --------------------------------------------------------------------------

def _pool_terms(q32: torch.Tensor, st_t0, st_t1, st_T0, st_T1, scale: float,
                Gk=None):
    '''Pool contributions for queries q32 (B, H, L, Dh) fp32.
    Returns Z_p (B, H, L) and N_p (B, H, L, Dv). With Gk given (delta
    mode, v4) the slopes read through the damped-key Gram whitener:
    (t1, T1) -> tau (Gk + eps tau I)^-1 (t1, T1), tau = tr(Gk)/d —
    the RLS/DeltaNet fixed point on the pooled subtable; isotropic Gk
    recovers the Hebbian slopes exactly. Constants untouched.'''
    t1, T1 = st_t1, st_T1
    if Gk is not None:
        d = q32.shape[-1]
        tau = (torch.diagonal(Gk, dim1=-2, dim2=-1).sum(-1) / d)  # (B,H)
        live = tau > 0
        if bool(live.any()):
            eps = 1e-4 * tau.clamp_min(1e-30)
            A = Gk + (eps[..., None, None]
                      * torch.eye(d, device=Gk.device, dtype=Gk.dtype))
            rhs = torch.cat([st_t1.unsqueeze(-1), st_T1], dim=-1)
            sol = torch.linalg.solve(
                A + (~live)[..., None, None] * torch.eye(
                    d, device=Gk.device, dtype=Gk.dtype), rhs)
            sol = sol * tau.clamp_min(0)[..., None, None]
            t1w, T1w = sol[..., 0], sol[..., 1:]
            keep = live[..., None]
            t1 = torch.where(keep, t1w, st_t1)
            T1 = torch.where(keep.unsqueeze(-1), T1w, st_T1)
    Z_p = st_t0.unsqueeze(-1) + torch.einsum('bhld,bhd->bhl', q32, t1) * scale
    N_p = st_T0.unsqueeze(-2) + torch.einsum('bhld,bhde->bhle', q32, T1) * scale
    return Z_p, N_p


def _combined_readout(q32, k32, v32, logits_bias, allow, st, scale, eps_z,
                      health: Health | None, pool_write: str = 'hebbian'):
    '''Shared readout: q32 (B,H,L,Dh) fp32; k32/v32 (B,H,T,*) fp32;
    logits_bias (B,H,T) fp32 added to raw logits (the mass column);
    allow (B,H,L,T) bool. Returns out (B,H,L,Dv), logZ (B,H,L),
    xraw (B,H,L,T) raw logits without bias (for the score path).'''
    xraw = torch.einsum('bhld,bhtd->bhlt', q32, k32) * scale
    x = xraw + logits_bias.unsqueeze(-2)
    neg = torch.finfo(torch.float32).min
    x = torch.where(allow, x, torch.full_like(x, neg))
    M = x.amax(dim=-1, keepdim=True).detach()
    M = torch.clamp(M, min=-1e30)                      # all-dead guard
    e = torch.exp(x - M)
    Z_a = e.sum(dim=-1)                                # (B,H,L)
    N_a = torch.einsum('bhlt,bhte->bhle', e, v32)      # (B,H,L,Dv)
    Z_p, N_p = _pool_terms(q32, st.t0, st.t1, st.T0, st.T1, scale,
                           Gk=(st.Gk if pool_write == 'delta' else None))
    em = torch.exp(-M.squeeze(-1))
    Z = Z_a + em * Z_p
    N = N_a + em.unsqueeze(-1) * N_p
    bad = Z < eps_z * Z_a                              # signed-pool guardrail
    Z_a_safe = torch.clamp(Z_a, min=1e-30)
    out = torch.where(bad.unsqueeze(-1),
                      N_a / Z_a_safe.unsqueeze(-1),
                      N / torch.where(bad, Z_a_safe, Z).unsqueeze(-1))
    logZ = M.squeeze(-1) + torch.log(torch.where(bad, Z_a_safe, Z))
    if health is not None:
        with torch.no_grad():
            health.z_fallback += bad.float().mean().item()
            health.z_events += 1
    return out, logZ, xraw


def _transport(rope: RoPE, ring_q: torch.Tensor, delta: torch.Tensor):
    '''Rotate ring queries forward by per-query offsets delta (W,).'''
    rope.prepare_m(int(delta.max().item()) + 1)
    mcos = rope.mcos.to(ring_q.dtype)[delta]      # (W, Dh/2)
    msin = rope.msin.to(ring_q.dtype)[delta]
    x = ring_q.reshape(*ring_q.shape[:-1], ring_q.shape[-1] // 2, 2)
    x = x * mcos[..., None] + torch.stack((-x[..., 1], x[..., 0]), dim=-1) * msin[..., None]
    return x.reshape(*ring_q.shape)


# --------------------------------------------------------------------------
# Management (v2 scoring)
# --------------------------------------------------------------------------

@torch.no_grad()
def _manage(att: SoftmaxAttention, st: LayerState, t_end: int,
            mcfg: ManageCfg, health: Health):
    '''Score alive atoms, pick the cheapest exits down to budget.
    Returns (sel_evict, sel_p0, sel_p1, mu, sigma_tot2) — (B,H,T) masks
    plus the fp32 measure statistics for the write path.'''
    B, H, T, Dh = st.k.shape
    scale = 1.0 / math.sqrt(Dh)
    alive_cnt = int(st.alive[0, 0].sum().item())     # uniform across (B,H)
    r = alive_cnt - mcfg.budget
    if r <= 0:
        return None

    # ---- eviction score: v1 transported-linear MC, unchanged ----
    delta = (t_end - st.ring_pos).clamp(min=0)
    qt = _transport(att.rope, st.ring_q, delta)      # (B,H,W,Dh) fp32
    k32, v32 = st.k.float(), st.v.float()
    allow = st.alive.unsqueeze(-2).expand(B, H, qt.shape[2], T)
    f, logZ, xraw = _combined_readout(qt, k32, v32, st.logc, allow, st,
                                      scale, mcfg.eps_z, None,
                                      pool_write=mcfg.pool_write)
    d2 = (v32.pow(2).sum(-1).unsqueeze(-2) - 2 * torch.einsum(
        'bhwe,bhte->bhwt', f, v32) + f.pow(2).sum(-1).unsqueeze(-1))
    d2 = d2.clamp(min=0)
    logw = xraw + st.logc.unsqueeze(-2) - logZ.unsqueeze(-1)   # log w_j(q)
    s_evict = (torch.exp(2 * logw) * d2).mean(dim=-2)          # (B,H,T)
    log_s_evict = torch.log(s_evict + 1e-45)

    # ---- demotion score: position-factorized measure (§6(a′)/5′-7) ----
    mu, var_c, V, cap = _measure_stats(att.rope, st, t_end, mcfg.lam)
    sigma_tot2 = var_c + V
    logZbar = logZ.mean(dim=-1)                                # (B,H)
    d2bar = torch.log(d2.mean(dim=-2) + 1e-45)                 # log ||v-f̄||²
    amp = 2 * (st.logc + mu - logZbar.unsqueeze(-1))
    # residual terms, log-domain; expm1 keeps small-sigma precision
    res_p1 = 2 * V + var_c + torch.log(
        (torch.expm1(var_c) - var_c).clamp(min=1e-45))
    res_p0 = sigma_tot2 + torch.log(torch.expm1(sigma_tot2).clamp(min=1e-45))
    p0_branch = cap < mcfg.slope_eps          # slope degeneracy (5′-8)
    log_s_dem = amp + torch.where(p0_branch, res_p0, res_p1) + d2bar
    if not mcfg.demote:
        log_s_dem = torch.full_like(log_s_dem, float('inf'))

    inf = torch.finfo(torch.float32).max
    dead = ~st.alive
    log_s_evict = log_s_evict.masked_fill(dead, inf)
    log_s_dem = log_s_dem.masked_fill(dead, inf)
    s_min = torch.minimum(log_s_evict, log_s_dem)
    idx = s_min.topk(r, dim=-1, largest=False).indices          # (B,H,r)
    sel = torch.zeros_like(st.alive)
    sel.scatter_(-1, idx, True)
    sel_dem = sel & (log_s_dem <= log_s_evict)
    sel_evict = sel & ~sel_dem
    sel_p0 = sel_dem & p0_branch
    sel_p1 = sel_dem & ~p0_branch
    health.demoted_p0 += int(sel_p0.sum().item())
    health.demoted_p1 += int(sel_p1.sum().item())
    health.evicted += int(sel_evict.sum().item())
    return sel_evict, sel_p0, sel_p1, mu, sigma_tot2


def attn_block_step(att: SoftmaxAttention, x, st: LayerState, pos0: int,
                    mcfg: ManageCfg, manage: bool, health: Health):
    '''One attention block step: project the new block, read out against
    atoms + pool, then (optionally) manage down to budget. x is the
    post-rmsnorm block input (B, L, dim). Returns (out, new_state).'''
    B, L = x.shape[0], x.shape[1]
    H, Dh = att.kv_head_count, att.dim // att.head_count
    assert att.head_count == H and att.v_dim_mult == 1 \
        and att.short_conv_size is None, 'unified path covers the SAX subset'
    assert L >= mcfg.ring_window
    scale = 1.0 / math.sqrt(Dh)

    qp, kp, vp = att.wq(x), att.wk(x), att.wv(x)
    q = qp.reshape(B, L, H, Dh).transpose(1, 2)
    k = kp.reshape(B, L, H, Dh).transpose(1, 2)
    if att.qk_norm:
        q, k = att.q_norm(q), att.k_norm(k)
    att.rope.prepare_m(pos0 + L)
    q = att.rope(q, pos0)
    k = att.rope(k, pos0)
    v = vp.reshape(B, L, H, Dh).transpose(1, 2)

    # append the block as fresh atoms (mass 1, alive)
    k_all = torch.cat([st.k, k], dim=2)
    v_all = torch.cat([st.v, v], dim=2)
    logc_all = torch.cat(
        [st.logc, torch.zeros(B, H, L, device=x.device, dtype=torch.float32)], dim=2)
    alive_all = torch.cat(
        [st.alive, torch.ones(B, H, L, device=x.device, dtype=torch.bool)], dim=2)
    pos_all = torch.cat(
        [st.pos, torch.arange(pos0, pos0 + L, device=x.device)], dim=0)
    T = k_all.shape[2]

    # visibility: alive atoms, and within the new block causal lower-right
    allow = alive_all.unsqueeze(-2).expand(B, H, L, T).clone()
    causal = torch.ones(L, L, device=x.device, dtype=torch.bool).tril()
    allow[:, :, :, T - L:] &= causal

    out, _, _ = _combined_readout(
        q.float(), k_all.float(), v_all.float(), logc_all, allow, st,
        scale, mcfg.eps_z, health, pool_write=mcfg.pool_write)
    out = out.to(x.dtype).transpose(1, 2).reshape(B, L, H * Dh)
    out = att.wo(out)

    # ring: the last W queries of the block (positions t_end-W .. t_end-1)
    W = mcfg.ring_window
    ring_q = q[:, :, -W:, :].detach().float()
    ring_pos = torch.arange(pos0 + L - W, pos0 + L, device=x.device)

    # v3 stepped gate: the elapsed block ages the pool's phase-carrying
    # moments (t1/T1) by one block of per-band decay, before this
    # boundary's writes; constants (t0/T0, the P=0 tier) never decay.
    t1_in, T1_in = st.t1, st.T1
    if manage and mcfg.pool_gate == 'stepped':
        dec = _band_decay(att.rope, mcfg.lam, L, x.device)
        t1_in = t1_in * dec
        T1_in = T1_in * dec.unsqueeze(-1)
    st2 = LayerState(k_all, v_all, logc_all, alive_all, pos_all,
                     st.t0, t1_in, st.T0, T1_in, st.Gk, ring_q, ring_pos)

    if manage:
        picks = _manage(att, st2, pos0 + L, mcfg, health)
        if picks is not None:
            sel_evict, sel_p0, sel_p1, mu, sig2 = picks
            sel_dem = sel_p0 | sel_p1
            alive_new = alive_all & ~(sel_evict | sel_dem)
            if sel_dem.any():
                mu_c = mu.clamp(max=mcfg.mu_clamp)
                with torch.no_grad():
                    health.mu_clamped += ((mu > mcfg.mu_clamp) & sel_dem
                                          ).float().sum().item() / max(
                                              1, int(sel_dem.sum().item()))
                    health.mu_events += 1
                # projected-pool writes (5′-7/5′-8): mass factor
                # e^{sigma^2/2}; P=1 slope = per-band damped key.
                w_all = torch.exp(mu_c + 0.5 * sig2 + logc_all)   # (B,H,T)
                w0 = w_all * sel_p0.float()
                w1 = w_all * sel_p1.float()
                kf, vf = k_all.float(), v_all.float()
                if mcfg.pool_gate == 'static':          # v2: damp at write
                    omega = _band_freqs(att.rope, x.device)
                    gamma = _coherence(mcfg.lam, omega).abs().float()
                    kd = kf * gamma.repeat_interleave(2)
                else:                                   # v3: age via decay
                    kd = kf
                t0 = st2.t0 + (w0 + w1 * (1 - mu_c)).sum(-1)
                T0 = st2.T0 + torch.einsum('bht,bhte->bhe',
                                           w0 + w1 * (1 - mu_c), vf)
                t1 = st2.t1 + torch.einsum('bht,bhtd->bhd', w1, kd)
                T1 = st2.T1 + torch.einsum('bht,bhtd,bhte->bhde', w1, kd, vf)
                Gk_new = (st2.Gk + torch.einsum('bht,bhtd,bhte->bhde',
                                                w1, kd, kd)
                          if mcfg.pool_write == 'delta' else st2.Gk)
                with torch.no_grad():
                    # max over weights actually written (post-selection);
                    # w_all over unselected atoms can be astronomically
                    # larger and is never absorbed.
                    health.pool_w_max = max(health.pool_w_max,
                                            float((w0 + w1).max()))
                    health.pool_t1_max = max(health.pool_t1_max,
                                             float(t1.abs().max()))
                st2 = LayerState(k_all, v_all, logc_all, alive_new, pos_all,
                                 t0, t1, T0, T1, Gk_new, ring_q, ring_pos)
            else:
                st2 = LayerState(k_all, v_all, logc_all, alive_new, pos_all,
                                 st2.t0, st2.t1, st2.T0, st2.T1, st2.Gk,
                                 ring_q, ring_pos)
    return out, st2


def _block_cell(blk, xb, pos0: int, mcfg: ManageCfg, manage: bool,
                health: Health, *state_tensors):
    '''One (layer, block) cell: rmsnorm→attn(stream)→residual→ffn. Shaped
    for torch.utils.checkpoint: tensor state flattened in/out.'''
    st = LayerState.from_tensors(state_tensors)
    h = blk.rmsnorm1(xb)
    attn_out, st2 = attn_block_step(blk.att, h, st, pos0, mcfg, manage, health)
    xb = xb + attn_out
    xb = xb + blk.ffn(blk.rmsnorm2(xb))
    return (xb, *st2.tensors())


def stream_hidden(model, tokens: torch.Tensor, mcfg: ManageCfg,
                  manage: bool = True, use_checkpoint: bool = False,
                  health: Health | None = None) -> torch.Tensor:
    '''Managed block-streaming forward. Returns post-rms_head hidden
    states (B, L, dim) — same contract as model(x, return_hidden=True).
    With manage=False and empty pools this is exactly softmax attention
    (gate 0). Gradients flow through all surviving state (BPTT across
    blocks); use_checkpoint recomputes each (layer, block) cell.'''
    health = health if health is not None else Health()
    B, L = tokens.shape
    bl = mcfg.block_len
    x = model.embedding(tokens)
    dtype = x.dtype
    states = [init_state(blk.att, B, tokens.device, dtype, mcfg)
              for blk in model.blocks]
    outs = []
    for pos0 in range(0, L, bl):
        xb = x[:, pos0:pos0 + bl]
        for li, blk in enumerate(model.blocks):
            ts = states[li].tensors()
            if use_checkpoint and torch.is_grad_enabled():
                res = checkpoint(_block_cell, blk, xb, pos0, mcfg, manage,
                                 health, *ts, use_reentrant=False)
            else:
                res = _block_cell(blk, xb, pos0, mcfg, manage, health, *ts)
            assert res is not None   # checkpoint's stub says Optional
            xb, states[li] = res[0], LayerState.from_tensors(res[1:])
        outs.append(xb)
    return model.rms_head(torch.cat(outs, dim=1))
