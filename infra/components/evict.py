# Eviction-only streaming block (branch evict-only, 2026-09-02).
#
# The simplest member of the triage family: hard eviction under the
# E1-validated transported error-law score and nothing else — no mass
# column, no pool, no demotion, no denominator guardrail. Why it exists
# (experiments/evict-only.md): tier-1's three training kills all traced
# to the demote/pool readout term (a signed, un-normalized denominator
# contribution weighted by a detached forecast of the logit); hard
# selection on the eviction exit never misbehaved. Here every atom that
# enters the readout does so through the current query's softmax, so
# the gradient reaching any surviving (k, v) is an ordinary attention
# probability — the same gradient structure as the stateless path.
#
# Readout = exact masked softmax attention over the alive slots, run on
# F.scaled_dot_product_attention (fused, no materialized probabilities;
# bf16 under autocast exactly like the stateless forward). Scoring = a
# W-query Monte-Carlo estimate over the RoPE-transported ring, fp32,
# under no_grad; it is the only extra work per block (three W x cap
# matmuls per layer, W = ring_window).
#
# torch.compile: the (layer, block) cell takes the RoPE tables and the
# query positions as TENSORS, so one graph serves every block position;
# only the python-static (do_manage, r) pair specializes (<= 3 graphs
# per run) — unlike the tier-1 cell, which took pos0 as an int.
#
# Score forms (atom j; transported ring queries q; p_j(q) its softmax
# mass; f(q) the full readout; r_j = p_j / (1 - p_j)):
#   lin : mean_q  r_j   ||v_j - f(q)||     exact single-token error norm
#                                          (theory Lemma 3); E1's default
#   sq  : mean_q  r_j^2 ||v_j - f(q)||^2   exact squared error
#   p2  : mean_q  p_j^2 ||v_j - f(q)||^2   unified.py's v1 form — kept
#                                          only as the regression anchor
#                                          for the tier-1 m<budget>e cells
# Optional Gumbel perturbation of the log-score before top-k
# (gumbel_tau > 0, training only): T -> inf is random dropping, T = 0
# the deterministic policy; deploy at T = 0.

import math
from dataclasses import dataclass, replace

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .attention import SoftmaxAttention


@dataclass
class EvictCfg:
    block_len: int = 256      # decisions between blocks of this many tokens
    budget: int = 512         # alive atoms per (batch, head) after a decision
    ring_window: int = 32     # recent queries forming the score ensemble
    lookahead: int = 0        # transport target = block end + lookahead
    score: str = 'lin'        # 'lin' | 'sq' | 'p2'
    gumbel_tau: float = 0.0   # > 0: perturbed top-k (grad-enabled runs only)
    manage_every: int = 1     # decide at every k-th block boundary
    compile_cell: bool = False


@dataclass
class EvictState:
    '''Per-layer streaming state over fixed-capacity slots. Functional:
    block steps return new instances; nothing is mutated in place.'''
    k: torch.Tensor          # (B, H, cap, Dh) post-RoPE keys
    v: torch.Tensor          # (B, H, cap, Dv)
    alive: torch.Tensor      # (B, H, cap) bool
    pos: torch.Tensor        # (B, H, cap) int64 absolute positions
    ring_q: torch.Tensor     # (B, H, W, Dh) fp32 post-RoPE recent queries
    ring_pos: torch.Tensor   # (W,) int64

    def tensors(self) -> tuple:
        return (self.k, self.v, self.alive, self.pos, self.ring_q, self.ring_pos)

    @staticmethod
    def from_tensors(ts) -> 'EvictState':
        return EvictState(*ts)


def init_state(att: SoftmaxAttention, batch: int, device, dtype,
               cfg: EvictCfg, cap: int) -> EvictState:
    H, Dh = att.head_count, att.dim // att.head_count
    z = lambda *s, dt=dtype: torch.zeros(*s, device=device, dtype=dt)
    return EvictState(
        k=z(batch, H, cap, Dh), v=z(batch, H, cap, Dh),
        alive=torch.zeros(batch, H, cap, device=device, dtype=torch.bool),
        pos=torch.zeros(batch, H, cap, device=device, dtype=torch.int64),
        ring_q=z(batch, H, cfg.ring_window, Dh, dt=torch.float32),
        ring_pos=torch.zeros(cfg.ring_window, device=device, dtype=torch.int64))


class Health:
    '''Stream-health counters (python-side, no device syncs).'''
    def __init__(self):
        self.evicted = 0      # atoms evicted, summed over batch/head/layer
        self.decisions = 0    # management boundaries hit

    def as_dict(self) -> dict:
        return {'evict_evicted': float(self.evicted),
                'evict_decisions': float(self.decisions)}


def _rope_apply(x: torch.Tensor, mcos: torch.Tensor, msin: torch.Tensor):
    '''RoPE.forward's arithmetic on pre-sliced tables: x (..., L, Dh),
    mcos/msin (L, Dh/2). Bit-identical to RoPE.forward(x, offset).'''
    shape = x.shape
    x = x.reshape(*shape[:-1], shape[-1] // 2, 2)
    x = x * mcos[..., None] + torch.stack((-x[..., 1], x[..., 0]), dim=-1) * msin[..., None]
    return x.reshape(shape)


@torch.no_grad()
def _scores(st: EvictState, cfg: EvictCfg, mcos_tr, msin_tr, scale: float):
    '''Transported error-law eviction log-score per slot, (B, H, cap)
    fp32. Dead slots carry +max so topk(largest=False) never picks them.'''
    qt = _rope_apply(st.ring_q, mcos_tr, msin_tr)               # (B,H,W,Dh)
    k32, v32 = st.k.float(), st.v.float()
    x = torch.einsum('bhwd,bhtd->bhwt', qt, k32) * scale
    x = x.masked_fill(~st.alive.unsqueeze(-2), float('-inf'))
    p = torch.softmax(x, dim=-1)                                 # (B,H,W,cap)
    f = p @ v32                                                  # (B,H,W,Dv)
    d2 = (v32.pow(2).sum(-1).unsqueeze(-2)
          - 2 * (f @ v32.transpose(-1, -2))
          + f.pow(2).sum(-1).unsqueeze(-1)).clamp_min(0)        # (B,H,W,cap)
    if cfg.score == 'lin':
        r = p / (1 - p).clamp_min(1e-6)
        s = (r * d2.sqrt()).mean(dim=-2)
    elif cfg.score == 'sq':
        r = p / (1 - p).clamp_min(1e-6)
        s = (r * r * d2).mean(dim=-2)
    elif cfg.score == 'p2':
        s = (p * p * d2).mean(dim=-2)
    else:
        raise ValueError(f'unknown score form {cfg.score!r}')
    log_s = torch.log(s + 1e-45)
    return log_s.masked_fill(~st.alive, torch.finfo(torch.float32).max)


@torch.no_grad()
def _manage(st: EvictState, cfg: EvictCfg, mcos_tr, msin_tr, scale: float,
            r: int, stochastic: bool) -> torch.Tensor:
    '''Pick the r cheapest alive atoms per (batch, head). r is a python
    int from the caller's static bookkeeping (static topk shape).'''
    log_s = _scores(st, cfg, mcos_tr, msin_tr, scale)
    if stochastic:
        u = torch.rand_like(log_s).clamp(1e-20, 1.0 - 1e-7)
        g = -torch.log(-torch.log(u))
        log_s = torch.where(st.alive, log_s + cfg.gumbel_tau * g, log_s)
    idx = log_s.topk(r, dim=-1, largest=False).indices          # (B,H,r)
    return torch.zeros_like(st.alive).scatter_(-1, idx, True)


def attn_block_step(att: SoftmaxAttention, x, st: EvictState,
                    mcos_blk, msin_blk, qpos, mcos_tr, msin_tr,
                    cfg: EvictCfg, do_manage: bool, r: int, stochastic: bool):
    '''One attention block over fixed-capacity slot state: scatter the L
    new tokens into free slots, fused masked attention over the slots,
    then (optionally) evict r atoms per head. Visibility is one rule:
    alive & (pos_slot <= pos_query), which covers both old atoms and
    within-block causality.'''
    B, L = x.shape[0], x.shape[1]
    H, Dh = att.head_count, att.dim // att.head_count
    assert att.kv_head_count == H and att.v_dim_mult == 1 \
        and att.short_conv_size is None, 'evict path covers the SAX subset'
    scale = 1.0 / math.sqrt(Dh)

    qp, kp, vp = att.wq(x), att.wk(x), att.wv(x)
    q = qp.reshape(B, L, H, Dh).transpose(1, 2)
    k = kp.reshape(B, L, H, Dh).transpose(1, 2)
    if att.qk_norm:
        q, k = att.q_norm(q), att.k_norm(k)
    q = _rope_apply(q, mcos_blk, msin_blk)
    k = _rope_apply(k, mcos_blk, msin_blk)
    v = vp.reshape(B, L, H, Dh).transpose(1, 2)

    # scatter the block into L free slots (capacity guarantees they exist)
    free_idx = (~st.alive).float().topk(L, dim=-1).indices        # (B,H,L)
    idx_d = free_idx.unsqueeze(-1).expand(B, H, L, Dh)
    k_all = st.k.scatter(2, idx_d, k.to(st.k.dtype))
    v_all = st.v.scatter(2, idx_d, v.to(st.v.dtype))
    alive_all = st.alive.scatter(2, free_idx, torch.ones(
        B, H, L, device=x.device, dtype=torch.bool))
    pos_all = st.pos.scatter(2, free_idx, qpos.expand(B, H, L))

    allow = alive_all.unsqueeze(-2) & (
        pos_all.unsqueeze(-2) <= qpos.view(1, 1, L, 1))           # (B,H,L,cap)
    out = F.scaled_dot_product_attention(q, k_all, v_all, attn_mask=allow)
    out = att.wo(out.transpose(1, 2).reshape(B, L, H * Dh))

    W = cfg.ring_window
    st2 = EvictState(k_all, v_all, alive_all, pos_all,
                     q[:, :, -W:, :].detach().float(), qpos[-W:])
    if do_manage and r > 0:
        sel = _manage(st2, cfg, mcos_tr, msin_tr, scale, r, stochastic)
        st2 = replace(st2, alive=alive_all & ~sel)
    return out, st2


def _block_cell(blk, xb, mcos_blk, msin_blk, qpos, mcos_tr, msin_tr,
                cfg: EvictCfg, do_manage: bool, r: int, stochastic: bool,
                *state_tensors):
    '''One (layer, block) cell: rmsnorm -> attn(stream) -> residual ->
    ffn. Tensor state flattened in/out for torch.utils.checkpoint.'''
    st = EvictState.from_tensors(state_tensors)
    h = blk.rmsnorm1(xb)
    a, st2 = attn_block_step(blk.att, h, st, mcos_blk, msin_blk, qpos,
                             mcos_tr, msin_tr, cfg, do_manage, r, stochastic)
    xb = xb + a
    xb = xb + blk.ffn(blk.rmsnorm2(xb))
    return (xb, *st2.tensors())


_COMPILED_CELL = None


def _cell_fn(compile_cell: bool):
    global _COMPILED_CELL
    if not compile_cell:
        return _block_cell
    if _COMPILED_CELL is None:
        import torch._dynamo
        torch._dynamo.config.cache_size_limit = max(
            torch._dynamo.config.cache_size_limit, 64)
        _COMPILED_CELL = torch.compile(_block_cell, dynamic=False)
    return _COMPILED_CELL


def stream_hidden(model, tokens: torch.Tensor, cfg: EvictCfg,
                  manage: bool = True, use_checkpoint: bool = False,
                  health: Health | None = None) -> torch.Tensor:
    '''Eviction-managed block-streaming forward. Returns post-rms_head
    hidden states (B, L, dim) — same contract as model(x,
    return_hidden=True). With manage=False (or a non-binding budget)
    this is exactly softmax attention (gate 0). Gradients flow through
    all surviving state (BPTT across blocks); use_checkpoint recomputes
    each (layer, block) cell.'''
    B, L = tokens.shape
    bl, W = cfg.block_len, cfg.ring_window
    assert L % bl == 0, 'sequence must be a whole number of blocks'
    assert bl >= W, 'ring_window must fit in one block'
    x = model.embedding(tokens)
    dev = x.device.type
    sdt = (torch.get_autocast_dtype(dev) if torch.is_autocast_enabled(dev)
           else x.dtype)
    # capacity: with management the state never holds more than
    # budget + manage_every*block atoms; without, the whole sequence
    cap = min(L, cfg.budget + cfg.manage_every * bl) if manage else L
    rope = model.rope
    rope.prepare_m(L + W + cfg.lookahead + 1)     # hoisted, once
    mcos, msin = rope.mcos, rope.msin             # (Lmax, Dh/2)
    # transport: ring query i sits at block_end - W + i; rotate it forward
    # by W - i (+ lookahead) to the decision horizon — constant per run
    delta = torch.arange(W, 0, -1, device=x.device) + cfg.lookahead
    mcos_tr, msin_tr = mcos[delta].float(), msin[delta].float()

    states = [init_state(blk.att, B, x.device, sdt, cfg, cap)
              for blk in model.blocks]
    cell = _cell_fn(cfg.compile_cell)
    stochastic = cfg.gumbel_tau > 0 and torch.is_grad_enabled()
    H, n_layers = model.blocks[0].att.head_count, len(model.blocks)
    outs, n_alive = [], 0                          # python-side bookkeeping
    for bi, pos0 in enumerate(range(0, L, bl)):
        n_alive += bl
        do_manage = manage and (bi + 1) % cfg.manage_every == 0
        r = max(n_alive - cfg.budget, 0) if do_manage else 0
        xb = x[:, pos0:pos0 + bl]
        mc, ms = mcos[pos0:pos0 + bl], msin[pos0:pos0 + bl]
        qpos = torch.arange(pos0, pos0 + bl, device=x.device)
        for li, blk in enumerate(model.blocks):
            ts = states[li].tensors()
            args = (blk, xb, mc, ms, qpos, mcos_tr, msin_tr, cfg,
                    do_manage, r, stochastic)
            if use_checkpoint and torch.is_grad_enabled():
                res = checkpoint(cell, *args, *ts, use_reentrant=False)
            else:
                res = cell(*args, *ts)
            assert res is not None
            xb, states[li] = res[0], EvictState.from_tensors(res[1:])
        if do_manage and r > 0:
            n_alive = cfg.budget
            if health is not None:
                health.evicted += r * B * H * n_layers
                health.decisions += 1
        outs.append(xb)
    return model.rms_head(torch.cat(outs, dim=1))
