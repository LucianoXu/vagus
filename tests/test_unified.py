# Gates 0/1 + v2 closed-form checks for the unified streaming path.
# Gate 0: with management off (or budget never binding) the streaming
#         readout IS softmax attention: logits match forward().
# Gate 1: the training loss through stream_hidden(manage=False) equals
#         the stateless loss.
# v2:     projection coefficients (5′-7 Stein closed form), coherence
#         factor gamma_b (§6(a′)), pool additivity under damped +
#         mass-factor writes.

import math

import torch

from infra.components.unified import (Health, ManageCfg, stream_hidden,
                                      _coherence)
from infra.models.transformer_pp import TransformerPP


def tiny_model(seed=0):
    torch.manual_seed(seed)
    return TransformerPP(vocab_size=512, dim=128, head_dim=32,
                         context_len=1024, layer_count=3,
                         tie_embedding=True, qk_norm=True).float().eval()


def tokens(B=2, L=512, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 512, (B, L), generator=g)


MCFG = ManageCfg(block_len=64, budget=96, ring_window=16)


def test_gate0_stream_equals_forward():
    model, x = tiny_model(), tokens()
    with torch.no_grad():
        ref = model(x, return_hidden=True)
        out = stream_hidden(model, x, MCFG, manage=False)
    diff = (ref - out).abs().max().item()
    assert diff < 2e-4, f'gate 0 failed: max hidden diff {diff}'
    loose = ManageCfg(block_len=64, budget=10_000, ring_window=16)
    with torch.no_grad():
        out2 = stream_hidden(model, x, loose, manage=True)
    assert (ref - out2).abs().max().item() < 2e-4


def test_gate1_loss_matches_stateless():
    model, x = tiny_model(), tokens()
    y = torch.roll(x, -1, dims=1)
    with torch.no_grad():
        lref = torch.nn.functional.cross_entropy(
            model.head(model(x, return_hidden=True)).transpose(1, 2), y)
        lstr = torch.nn.functional.cross_entropy(
            model.head(stream_hidden(model, x, MCFG, manage=False)).transpose(1, 2), y)
    assert abs(lref.item() - lstr.item()) < 1e-4


def test_managed_run_and_budget():
    model, x = tiny_model(), tokens()
    h = Health()
    with torch.no_grad():
        out = stream_hidden(model, x, MCFG, manage=True, health=h)
    assert torch.isfinite(out).all()
    assert h.evicted + h.demoted_p0 + h.demoted_p1 > 0
    d = h.as_dict()
    assert d.get('unified_z_fallback', 0.0) < 0.05, d


def test_managed_deterministic():
    model, x = tiny_model(), tokens()
    with torch.no_grad():
        a = stream_hidden(model, x, MCFG, manage=True)
        b = stream_hidden(model, x, MCFG, manage=True)
    assert torch.equal(a, b)


def test_backward_finite_and_checkpoint_matches():
    model, x = tiny_model(), tokens(B=1, L=256)
    model.train()
    y = torch.roll(x, -1, dims=1)

    def loss_of(use_ckpt):
        for p in model.parameters():
            p.grad = None
        out = stream_hidden(model, x, MCFG, manage=True, use_checkpoint=use_ckpt)
        loss = torch.nn.functional.cross_entropy(
            model.head(out).transpose(1, 2), y)
        loss.backward()
        g = torch.cat([p.grad.flatten() for p in model.parameters()
                       if p.grad is not None])
        assert torch.isfinite(g).all()
        return loss.item(), g.clone()

    l0, g0 = loss_of(False)
    l1, g1 = loss_of(True)
    assert abs(l0 - l1) < 1e-5
    assert (g0 - g1).abs().max().item() < 1e-4, 'checkpointed grads diverge'
    assert g0.abs().max().item() > 0


def test_demotion_engages_pool():
    model, x = tiny_model(), tokens()
    h = Health()
    tight = ManageCfg(block_len=64, budget=64, ring_window=16, demote=True)
    with torch.no_grad():
        stream_hidden(model, x, tight, manage=True, health=h)
    assert h.demoted_p0 + h.demoted_p1 > 0, 'no atom chose a demote exit'


# ---- v2 closed-form checks ----

def test_projection_coefficients_match_stein():
    '''Least-squares fit of a + b x to e^x under N(mu, sigma^2) samples
    must match Pi_1 e^x = e^{mu+sigma^2/2}(1 + (x - mu)) (theory 5′-7):
    a = e^{mu+sigma^2/2}(1 - mu), b = e^{mu+sigma^2/2}.'''
    g = torch.Generator().manual_seed(7)
    for mu, sig in [(0.3, 0.5), (-1.0, 0.8), (1.2, 0.3)]:
        x = mu + sig * torch.randn(400_000, generator=g, dtype=torch.float64)
        X = torch.stack([torch.ones_like(x), x], dim=1)
        coef = torch.linalg.lstsq(X, torch.exp(x).unsqueeze(1)).solution.squeeze()
        s = math.exp(mu + sig ** 2 / 2)
        assert abs(coef[0].item() - s * (1 - mu)) < 0.02 * s, (mu, sig, coef)
        assert abs(coef[1].item() - s) < 0.02 * s, (mu, sig, coef)


def test_coherence_factor_matches_numeric():
    '''gamma_b = |E_{s~Exp(lam)} e^{-i w s}| = lam/sqrt(lam^2+w^2).'''
    g = torch.Generator().manual_seed(11)
    lam = 1.0 / 64
    s = -torch.log(torch.rand(2_000_000, generator=g, dtype=torch.float64)) / lam
    for w in [0.0, 1.0 / 128, 1.0 / 16, 0.5]:
        num = torch.exp(torch.complex(torch.zeros_like(s), -w * s)).mean()
        closed = _coherence(lam, torch.tensor([w], dtype=torch.float64))[0]
        assert abs(num.abs().item() - closed.abs().item()) < 2e-3
        assert abs(num.item().real - closed.item().real) < 2e-3


def test_pool_additivity():
    '''Damped + mass-factor writes are rank-1 updates: writing a batch of
    atoms at once equals the sum of writing them one by one.'''
    g = torch.Generator().manual_seed(13)
    n, Dh, Dv = 5, 8, 8
    k = torch.randn(n, Dh, generator=g, dtype=torch.float64)
    v = torch.randn(n, Dv, generator=g, dtype=torch.float64)
    w1 = torch.rand(n, generator=g, dtype=torch.float64) + 0.5
    mu = torch.randn(n, generator=g, dtype=torch.float64)
    damp = torch.rand(Dh // 2, generator=g, dtype=torch.float64).repeat_interleave(2)
    kd = k * damp
    t0_batch = (w1 * (1 - mu)).sum()
    t1_batch = torch.einsum('t,td->d', w1, kd)
    T1_batch = torch.einsum('t,td,te->de', w1, kd, v)
    t0_seq = sum(w1[i] * (1 - mu[i]) for i in range(n))
    t1_seq = sum(w1[i] * kd[i] for i in range(n))
    T1_seq = sum(w1[i] * torch.outer(kd[i], v[i]) for i in range(n))
    assert abs(t0_batch.item() - t0_seq.item()) < 1e-10
    assert (t1_batch - t1_seq).abs().max().item() < 1e-10
    assert (T1_batch - T1_seq).abs().max().item() < 1e-10


# ---- v3 stepped-gate checks ----

def test_stepped_decay_semigroup_and_calibration():
    '''Two half-block decays compose to one full block, and the
    cumulative decay at the mean query horizon 1/lam equals the static
    damping gamma_b (the v2/v3 alignment).'''
    from infra.components.unified import _band_decay, _band_freqs, _coherence
    from infra.components.pos_embed import RoPE
    rope = RoPE(128, 32, 64)
    lam = 1.0 / 1024
    half = _band_decay(rope, lam, 128, 'cpu').double()
    full = _band_decay(rope, lam, 256, 'cpu').double()
    assert (half * half - full).abs().max().item() < 2e-6  # fp32 factors
    horizon = _band_decay(rope, lam, round(1 / lam), 'cpu').double()
    gamma = _coherence(lam, _band_freqs(rope, 'cpu')).abs().repeat_interleave(2)
    assert (horizon - gamma).abs().max().item() < 2e-6  # fp32 vs f64


def test_pool_gate_switch():
    '''With no demotions the two gate modes are bit-identical (empty
    pool: the eonly sanity); with demotions they must differ (the
    switch is live). The default 'static' path is the untouched v2
    branch — the existing tests above are its regression suite.'''
    model, x = tiny_model(), tokens()
    kw = dict(block_len=64, budget=64, ring_window=16)
    with torch.no_grad():
        a = stream_hidden(model, x, ManageCfg(**kw, demote=False,
                                              pool_gate='static'), manage=True)
        b = stream_hidden(model, x, ManageCfg(**kw, demote=False,
                                              pool_gate='stepped'), manage=True)
        assert torch.equal(a, b), 'empty-pool modes must be bit-identical'
        c = stream_hidden(model, x, ManageCfg(**kw, demote=True,
                                              pool_gate='static'), manage=True)
        d = stream_hidden(model, x, ManageCfg(**kw, demote=True,
                                              pool_gate='stepped'), manage=True)
        assert not torch.equal(c, d), 'gate switch has no effect'
        assert torch.isfinite(d).all()


# ---- v4 delta-write checks ----

def test_delta_isotropic_recovers_hebbian():
    '''With an isotropic damped-key Gram, tau (G+eps tau I)^-1 = ~I and
    the delta readout equals the Hebbian readout.'''
    from infra.components.unified import _pool_terms
    torch.manual_seed(3)
    B, H, L, d, dv = 2, 3, 5, 8, 8
    q = torch.randn(B, H, L, d)
    t0 = torch.rand(B, H) + 1
    t1 = torch.randn(B, H, d)
    T0 = torch.randn(B, H, dv)
    T1 = torch.randn(B, H, d, dv)
    g = 2.7
    Gk = g * torch.eye(d).expand(B, H, d, d).clone()
    Zh, Nh = _pool_terms(q, t0, t1, T0, T1, 0.5)
    Zd, Nd = _pool_terms(q, t0, t1, T0, T1, 0.5, Gk=Gk)
    assert (Zh - Zd).abs().max().item() < 1e-3
    assert (Nh - Nd).abs().max().item() < 1e-3
    # empty Gram: delta mode must fall back to Hebbian slopes untouched
    Z0, N0 = _pool_terms(q, t0, t1, T0, T1, 0.5, Gk=torch.zeros(B, H, d, d))
    assert torch.equal(Z0, Zh) and torch.equal(N0, Nh)


def test_delta_mode_runs_and_differs():
    model, x = tiny_model(), tokens()
    kw = dict(block_len=64, budget=64, ring_window=16, demote=True)
    with torch.no_grad():
        a = stream_hidden(model, x, ManageCfg(**kw, pool_write='hebbian'),
                          manage=True)
        b = stream_hidden(model, x, ManageCfg(**kw, pool_write='delta'),
                          manage=True)
    assert torch.isfinite(b).all()
    assert not torch.equal(a, b), 'delta switch has no effect'


# ---- upstream regression (engram triage-port cross-check, 2026-09-01):
# Gm orientation in _measure_stats. mu and V must match direct numerical
# integration of the offset measure; the transposed Gm is also real and
# passes symmetry intuition but biases V by tens of percent.

def test_measure_stats_matches_numeric_integration():
    import math as _m
    from infra.components.unified import (LayerState, _measure_stats,
                                          _as_complex, _band_freqs)
    from infra.components.pos_embed import RoPE
    torch.manual_seed(5)
    B, H, T, Dh, W = 1, 1, 5, 16, 8
    rope = RoPE(Dh * 4, Dh, 4096)
    lam, t_dec = 1.0 / 256, 700
    k = torch.randn(B, H, T, Dh)
    ring_q = torch.randn(B, H, W, Dh)
    ring_pos = torch.arange(t_dec - W, t_dec)
    z = lambda *s, dt=torch.float32: torch.zeros(*s, dtype=dt)
    st = LayerState(k=k, v=torch.randn(B, H, T, Dh),
                    logc=z(B, H, T), alive=torch.ones(B, H, T, dtype=torch.bool),
                    pos=torch.arange(T), t0=z(B, H), t1=z(B, H, Dh),
                    T0=z(B, H, Dh), T1=z(B, H, Dh, Dh),
                    Gk=z(B, H, Dh, Dh), logzbar=z(B, H),
                    ring_q=ring_q.float(), ring_pos=ring_pos)
    mu_mod, var_c, V_mod, cap = _measure_stats(rope, st, t_dec, lam)

    # independent numeric integration over s ~ Exp(lam)
    from infra.components.unified import _unrotate_ring
    u = _unrotate_ring(rope, st.ring_q, st.ring_pos)
    m = u.mean(dim=2)[0, 0].to(torch.complex128)          # (nb,)
    zc = _as_complex(k)[0, 0].to(torch.complex128)        # (T, nb)
    omega = _band_freqs(rope, 'cpu')                      # (nb,) f64
    beta = 1.0 / _m.sqrt(Dh)
    s_grid = torch.linspace(0, 24 / lam, 200_001, dtype=torch.float64)
    w = lam * torch.exp(-lam * s_grid)
    w = w / torch.trapz(w, s_grid)
    phase = torch.exp(-1j * omega[None, :] * (t_dec + s_grid)[:, None])
    mu_s = beta * (m.conj()[None, None, :] * zc[:, None, :]
                   * phase[None, :, :]).sum(-1).real      # (T, S)
    mu_num = torch.trapz(mu_s * w, s_grid, dim=-1)
    V_num = torch.trapz(mu_s.pow(2) * w, s_grid, dim=-1) - mu_num.pow(2)
    rel_mu = (mu_mod[0, 0].double() - mu_num).abs() / mu_num.abs().clamp_min(1e-9)
    rel_V = (V_mod[0, 0].double() - V_num).abs() / V_num.clamp_min(1e-12)
    print('rel_mu max', float(rel_mu.max()), 'rel_V max', float(rel_V.max()),
          'rel_V per-atom', [round(float(x), 4) for x in rel_V])
    assert float(rel_mu.max()) < 0.02, f'mu off: {rel_mu}'
    assert float(rel_V.max()) < 0.02, f'V off: {rel_V}'


# ---- v2.2 write-normalization checks ----

def test_ledger_normalization_forward_invariant():
    '''The ledger reparameterization (pool stored / e^b, readout factor
    e^{-M+b}, per-kv-head EMA b) must leave the managed forward exactly
    invariant — tolerance covers only float reordering.'''
    model, x = tiny_model(), tokens()
    kw = dict(block_len=64, budget=64, ring_window=16, demote=True)
    with torch.no_grad():
        a = stream_hidden(model, x, ManageCfg(**kw, pool_norm='off'),
                          manage=True)
        b = stream_hidden(model, x, ManageCfg(**kw, pool_norm='ledger'),
                          manage=True)
    diff = (a - b).abs().max().item()
    assert diff < 2e-5, f'ledger reparameterization changed the forward: {diff}'


def test_ledger_keeps_pool_intermediates_small():
    '''Conditioning claim: with the ledger on, the stored pool write
    weights are O(1)-ish rather than e^{mu+sigma^2/2}-scaled.'''
    model, x = tiny_model(), tokens()
    kw = dict(block_len=64, budget=64, ring_window=16, demote=True)
    h_off, h_on = Health(), Health()
    with torch.no_grad():
        stream_hidden(model, x, ManageCfg(**kw, pool_norm='off'),
                      manage=True, health=h_off)
        stream_hidden(model, x, ManageCfg(**kw, pool_norm='ledger'),
                      manage=True, health=h_on)
    assert h_on.pool_t1_max < h_off.pool_t1_max * 0.5 or \
        h_off.pool_t1_max < 10, (h_on.pool_t1_max, h_off.pool_t1_max)
