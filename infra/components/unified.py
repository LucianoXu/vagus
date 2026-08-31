# Unified-block streaming: atom table + P=1 moment pool, managed by the
# distortion calculus (engram theory §4/§5′/§7′c′), runnable frozen
# (inference) or differentiably (management-aware training).
#
# Scope (Tier 1): the SAX architecture subset — MHA (H_kv == H),
# v_dim_mult == 1, no short conv. Operations: eviction and type-change
# (demote to a P=1 pooled subtable). Merge and soft selection are out of
# scope (§7′(h)(ii) gradient risk); selection is HARD (argsort of scores,
# no gradient through the choice) while every surviving contribution —
# atoms, masses, pool moments — stays differentiable, so training adapts
# the model to a fixed policy (the Tier-1 question), not the policy.
#
# Readout per kv-head over a typed atom list:
#     F(q) = (N_a(q) + e^{-M} N_p(q)) / (Z_a(q) + e^{-M} Z_p(q))
# where atoms contribute N_a, Z_a through exp(logits - M) with the mass
# column log c folded into logits, and the pool contributes the P=1
# moment form (centered per §5′-2, e^{a} folded into the pooled weight):
#     Z_p(q) = t0 + q·t1/sqrt(Dh)      N_p(q) = T0 + q^T T1/sqrt(Dh)
# Denominator guardrail: where the signed pool term drives the total
# partition function below eps * Z_a, fall back to the atoms-only
# readout for that query (bounded output; occurrences are counted).
#
# Scores are Monte-Carlo estimates of the two exit costs over the ring
# of recent queries, RoPE-transported to the decision position (E1's
# transported ensemble): per atom j and ring query q,
#     evict:  (w_j(q))^2 ||v_j - F(q)||^2            (Lemma 3 kernel)
#     demote: (c_j e^{a_j} R_1(x_j - a_j) / Z(q))^2 ||v_j - F(q)||^2
#                                                     (Lemma 4, P=1)
# with R_1(u) = e^u - 1 - u and a_j the ring-mean raw logit. Both are
# the same substitution identity under different kernel perturbations,
# so the per-atom exit is argmin and removal is top-k of the min.

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .attention import SoftmaxAttention
from .pos_embed import RoPE


@dataclass
class ManageCfg:
    block_len: int = 256      # decisions between blocks of this many tokens
    budget: int = 512         # max alive atoms per (batch, kv-head)
    ring_window: int = 32     # recent queries kept for the score ensemble
    demote: bool = True       # enable the type-change exit (P=1 pool)
    eps_z: float = 1e-4       # denominator guardrail threshold
    a_clamp: float = 25.0     # cap on the centering logit inside exp()


@dataclass
class LayerState:
    '''Per-layer streaming state. Functional: block steps return new
    instances; tensors are never mutated in place (autograd safety).'''
    k: torch.Tensor        # (B, H, T, Dh) post-RoPE keys, absolute positions
    v: torch.Tensor        # (B, H, T, Dv)
    logc: torch.Tensor     # (B, H, T) fp32 log-mass
    alive: torch.Tensor    # (B, H, T) bool
    t0: torch.Tensor       # (B, H)        pool: sum w (1 - a)
    t1: torch.Tensor       # (B, H, Dh)    pool: sum w k
    T0: torch.Tensor       # (B, H, Dv)    pool: sum w (1 - a) v
    T1: torch.Tensor       # (B, H, Dh, Dv) pool: sum w k v^T
    ring_q: torch.Tensor   # (B, H, W, Dh) fp32 post-RoPE recent queries
    ring_pos: torch.Tensor # (W,) int64 absolute positions of ring queries

    def tensors(self) -> tuple:
        return (self.k, self.v, self.logc, self.alive, self.t0, self.t1,
                self.T0, self.T1, self.ring_q, self.ring_pos)

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
        t0=z(batch, H, dt=torch.float32), t1=z(batch, H, Dh, dt=torch.float32),
        T0=z(batch, H, Dv, dt=torch.float32),
        T1=z(batch, H, Dh, Dv, dt=torch.float32),
        ring_q=z(batch, H, 0, Dh, dt=torch.float32),
        ring_pos=torch.zeros(0, device=device, dtype=torch.int64),
    )


class Health:
    '''Accumulates stream-health counters across blocks/layers.'''
    def __init__(self):
        self.z_fallback = 0.0   # fraction of (query, head) readouts that
        self.z_events = 0       # tripped the denominator guardrail
        self.a_clamped = 0.0    # fraction of demoted atoms with clamped a
        self.a_events = 0
        self.demoted = 0        # atoms demoted (summed over B, H)
        self.evicted = 0

    def as_dict(self) -> dict:
        d = {}
        if self.z_events:
            d['unified_z_fallback'] = self.z_fallback / self.z_events
        if self.a_events:
            d['unified_a_clamped'] = self.a_clamped / self.a_events
        d['unified_demoted'] = float(self.demoted)
        d['unified_evicted'] = float(self.evicted)
        return d


def _pool_terms(q32: torch.Tensor, st_t0, st_t1, st_T0, st_T1, scale: float):
    '''Pool contributions for queries q32 (B, H, L, Dh) fp32.
    Returns Z_p (B, H, L) and N_p (B, H, L, Dv).'''
    Z_p = st_t0.unsqueeze(-1) + torch.einsum('bhld,bhd->bhl', q32, st_t1) * scale
    N_p = st_T0.unsqueeze(-2) + torch.einsum('bhld,bhde->bhle', q32, st_T1) * scale
    return Z_p, N_p


def _combined_readout(q32, k32, v32, logits_bias, allow, st, scale, eps_z,
                      health: Health | None):
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
    Z_p, N_p = _pool_terms(q32, st.t0, st.t1, st.T0, st.T1, scale)
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
    '''Rotate ring queries forward by per-query offsets delta (W,) —
    RoPE phases add, so rotating an already-rotated query by delta moves
    it to its position + delta.'''
    rope.prepare_m(int(delta.max().item()) + 1)
    mcos = rope.mcos.to(ring_q.dtype)[delta]      # (W, Dh/2)
    msin = rope.msin.to(ring_q.dtype)[delta]
    x = ring_q.reshape(*ring_q.shape[:-1], ring_q.shape[-1] // 2, 2)
    x = x * mcos[..., None] + torch.stack((-x[..., 1], x[..., 0]), dim=-1) * msin[..., None]
    return x.reshape(*ring_q.shape)


@torch.no_grad()
def _manage(att: SoftmaxAttention, st: LayerState, t_end: int,
            mcfg: ManageCfg, health: Health):
    '''Score alive atoms with the transported ring ensemble, pick the
    cheapest exits down to budget. Returns (sel_evict, sel_demote, a)
    as (B,H,T) masks + fp32 centers; caller applies them differentiably.'''
    B, H, T, Dh = st.k.shape
    scale = 1.0 / math.sqrt(Dh)
    alive_cnt = int(st.alive[0, 0].sum().item())     # uniform across (B,H)
    r = alive_cnt - mcfg.budget
    if r <= 0:
        return None
    delta = (t_end - st.ring_pos).clamp(min=0)
    qt = _transport(att.rope, st.ring_q, delta)      # (B,H,W,Dh) fp32
    k32, v32 = st.k.float(), st.v.float()
    allow = st.alive.unsqueeze(-2).expand(B, H, qt.shape[2], T)
    f, logZ, xraw = _combined_readout(qt, k32, v32, st.logc, allow, st,
                                      scale, mcfg.eps_z, None)
    # squared value displacement per (query, atom): ||v_j - f(q)||^2
    d2 = (v32.pow(2).sum(-1).unsqueeze(-2) - 2 * torch.einsum(
        'bhwe,bhte->bhwt', f, v32) + f.pow(2).sum(-1).unsqueeze(-1))
    d2 = d2.clamp(min=0)
    logw = xraw + st.logc.unsqueeze(-2) - logZ.unsqueeze(-1)   # log w_j(q)
    s_evict = (torch.exp(2 * logw) * d2).mean(dim=-2)          # (B,H,T)
    a = xraw.mean(dim=-2)                                      # (B,H,T)
    u = xraw - a.unsqueeze(-2)
    R1 = torch.expm1(u) - u
    amp = torch.exp((a + st.logc).unsqueeze(-2) - logZ.unsqueeze(-1))
    s_dem = ((amp * R1).pow(2) * d2).mean(dim=-2)              # (B,H,T)
    inf = torch.finfo(torch.float32).max
    dead = ~st.alive
    s_evict = s_evict.masked_fill(dead, inf)
    s_dem = s_dem.masked_fill(dead, inf) if mcfg.demote \
        else torch.full_like(s_dem, inf)
    s_min = torch.minimum(s_evict, s_dem)
    idx = s_min.topk(r, dim=-1, largest=False).indices          # (B,H,r)
    sel = torch.zeros_like(st.alive)
    sel.scatter_(-1, idx, True)
    sel_demote = sel & (s_dem <= s_evict)
    sel_evict = sel & ~sel_demote
    health.demoted += int(sel_demote.sum().item())
    health.evicted += int(sel_evict.sum().item())
    return sel_evict, sel_demote, a


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
    T = k_all.shape[2]

    # visibility: alive atoms, and within the new block causal lower-right
    allow = alive_all.unsqueeze(-2).expand(B, H, L, T).clone()
    causal = torch.ones(L, L, device=x.device, dtype=torch.bool).tril()
    allow[:, :, :, T - L:] &= causal

    out, _, _ = _combined_readout(
        q.float(), k_all.float(), v_all.float(), logc_all, allow, st,
        scale, mcfg.eps_z, health)
    out = out.to(x.dtype).transpose(1, 2).reshape(B, L, H * Dh)
    out = att.wo(out)

    # ring: the last W queries of the block (positions t_end-W .. t_end-1)
    W = mcfg.ring_window
    ring_q = q[:, :, -W:, :].detach().float()
    ring_pos = torch.arange(pos0 + L - W, pos0 + L, device=x.device)

    st2 = LayerState(k_all, v_all, logc_all, alive_all,
                     st.t0, st.t1, st.T0, st.T1, ring_q, ring_pos)

    if manage:
        picks = _manage(att, st2, pos0 + L, mcfg, health)
        if picks is not None:
            sel_evict, sel_demote, a = picks
            if sel_demote.any():
                a_c = a.clamp(max=mcfg.a_clamp)
                with torch.no_grad():
                    health.a_clamped += ((a > mcfg.a_clamp) & sel_demote
                                         ).float().sum().item() / max(
                                             1, int(sel_demote.sum().item()))
                    health.a_events += 1
                w_d = torch.exp(a_c + logc_all) * sel_demote.float()  # (B,H,T)
                kf, vf = k_all.float(), v_all.float()
                st2 = LayerState(
                    k_all, v_all, logc_all,
                    alive_all & ~(sel_evict | sel_demote),
                    st2.t0 + (w_d * (1 - a_c)).sum(-1),
                    st2.t1 + torch.einsum('bht,bhtd->bhd', w_d, kf),
                    st2.T0 + torch.einsum('bht,bhte->bhe', w_d * (1 - a_c), vf),
                    st2.T1 + torch.einsum('bht,bhtd,bhte->bhde', w_d, kf, vf),
                    ring_q, ring_pos)
            else:
                st2 = LayerState(k_all, v_all, logc_all,
                                 alive_all & ~sel_evict,
                                 st2.t0, st2.t1, st2.T0, st2.T1,
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
    states (B, L, dim) — same contract as model(x, return_hidden=True),
    but every position beyond the first block reads a managed cache.
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
            xb, states[li] = res[0], LayerState.from_tensors(res[1:])
        outs.append(xb)
    return model.rms_head(torch.cat(outs, dim=1))
