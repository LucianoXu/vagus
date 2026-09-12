# Kalman Delta Networks: the uncertainty scan against its own definition,
# the gains against the equations, and streaming == stateless forward.
#
# The gains were additionally checked against the reference implementation
# (github.com/ngocbh/kalman-delta-networks, kdn_ops/{iso,diag}_kdn_naive.py)
# when this landed: max abs 2.2e-16 on beta and 3.3e-16 on kappa in fp64.
# That check needs the upstream clone, so what lives here is the same
# recurrence written out independently — the oracle this file trusts.

from typing import NamedTuple

import pytest
import torch

from infra.components.kalman_delta import (
    KalmanDeltaNet, kalman_gain_diag, kalman_gain_iso, mobius_scan)
from infra.models import build_model

ARGS = dict(vocab_size=101, dim=64, layer_count=3, head_count=2, key_head_dim=16,
            value_head_dim=32, gate_rank=8, chunk_size=8, la_impl='torch', context_len=48)


class Gains(NamedTuple):
    k: torch.Tensor
    alpha: torch.Tensor
    omega_h: torch.Tensor
    omega_c: torch.Tensor
    r: torch.Tensor
    c0: torch.Tensor
    p0: torch.Tensor
    dk: int


def gains(B=3, L=37, H=4, K=16, seed=0) -> Gains:
    '''fp64 inputs shaped as the mixer produces them (keys L2-normalised).'''
    torch.manual_seed(seed)
    dt = torch.float64
    k = torch.randn(B, L, H, K, dtype=dt)
    k = k / k.norm(dim=-1, keepdim=True)
    return Gains(
        k=k,
        alpha=torch.rand(B, L, H, K, dtype=dt) * 0.5 + 0.5,
        omega_h=0.05 + torch.rand(B, L, H, dtype=dt),
        omega_c=0.05 + torch.rand(B, L, H, K, dtype=dt),
        r=0.05 + torch.rand(B, L, H, dtype=dt),
        c0=torch.ones(B, H, dtype=dt),
        p0=torch.ones(B, H, K, dtype=dt),
        dk=K,
    )


@pytest.mark.parametrize('L', [1, 8, 37])
def test_mobius_scan_parallel_matches_the_definition(L):
    '''The prefix-product form against the token recurrence. L is not a
    power of two on purpose: the doubling loop must handle the ragged tail.'''
    torch.manual_seed(1)
    shape = (3, L, 4)
    A, B, C, D = (torch.rand(*shape, dtype=torch.float64) + 0.1 for _ in range(4))
    x0 = torch.rand(3, 4, dtype=torch.float64) + 0.1
    par = mobius_scan(A, B, C, D, x0, impl='parallel')
    rec = mobius_scan(A, B, C, D, x0, impl='recurrent')
    assert torch.allclose(par, rec, rtol=1e-12, atol=1e-12)


def test_mobius_scan_survives_a_long_sequence():
    '''Unnormalised 2x2 prefix products overflow fp32 within a few hundred
    steps; the scan renormalises, which a Mobius map is invariant to.'''
    torch.manual_seed(2)
    A, B, C, D = (torch.rand(2, 1024, 3, dtype=torch.float64) * 3 + 0.5 for _ in range(4))
    x0 = torch.ones(2, 3, dtype=torch.float64)
    par = mobius_scan(A, B, C, D, x0, impl='parallel')
    rec = mobius_scan(A, B, C, D, x0, impl='recurrent')
    assert torch.isfinite(par).all()
    assert torch.allclose(par, rec, rtol=1e-9, atol=1e-12)


def test_iso_gain_matches_the_written_recurrence():
    '''a_t = mean_i(alpha_ti^2); bhat_t = a_t / c_{t-1} + omega_t;
    beta_t = bhat_t / (r_t + bhat_t |k_t|^2); c_t = 1/bhat_t + s |k_t|^2/(dk r_t).'''
    g = gains()
    k, alpha, omega, r, c0, K = g.k, g.alpha, g.omega_h, g.r, g.c0, g.dk
    beta, c_final = kalman_gain_iso(k, alpha, omega, r, c0, info_scale=float(K))

    c, betas = c0, []
    for t in range(k.shape[1]):
        a = alpha[:, t].square().mean(-1)
        kn2 = k[:, t].square().sum(-1)
        bhat = a / c + omega[:, t]
        betas.append(bhat / (r[:, t] + bhat * kn2))
        c = 1.0 / bhat + kn2 / r[:, t]                     # info_scale = dk cancels
    assert torch.allclose(beta, torch.stack(betas, 1), rtol=1e-12, atol=1e-12)
    assert torch.allclose(c_final, c, rtol=1e-12, atol=1e-12)
    # with L2-normalised keys the gain is a valid write strength
    assert (beta > 0).all() and (beta < 1).all()


def test_diag_gain_matches_the_written_recurrence():
    '''phat_t = alpha_t^2 p_{t-1} + omega_t;
    kappa_t = phat_t k_t / (r_t + sum_i phat_ti k_ti^2);
    p_t = phat_t / (1 + s k_t^2 / r_t * phat_t).'''
    g = gains()
    k, alpha, omega, r, p0, K = g.k, g.alpha, g.omega_c, g.r, g.p0, g.dk
    kappa, p_final = kalman_gain_diag(k, alpha, omega, r, p0, info_scale=float(K))

    p, kaps = p0, []
    for t in range(k.shape[1]):
        phat = alpha[:, t].square() * p + omega[:, t]
        den = r[:, t][..., None] + (phat * k[:, t].square()).sum(-1, keepdim=True)
        kaps.append(phat * k[:, t] / den)
        p = phat / (1.0 + K * k[:, t].square() / r[:, t][..., None] * phat)
    assert torch.allclose(kappa, torch.stack(kaps, 1), rtol=1e-12, atol=1e-12)
    assert torch.allclose(p_final, p, rtol=1e-12, atol=1e-12)
    # kappa is NOT parallel to k — that is what makes the write asymmetric
    cos = (kappa * k).sum(-1) / (kappa.norm(dim=-1) * k.norm(dim=-1))
    assert cos.min() < 0.999


def mixer(kalman: str, **over) -> KalmanDeltaNet:
    torch.manual_seed(0)
    return KalmanDeltaNet(dim=32, head_count=2, key_head_dim=8, value_head_dim=16,
                          gate_rank=4, chunk_size=4, impl='torch', kalman=kalman,
                          **over).double().eval()


@pytest.mark.parametrize('kalman', ['iso', 'diag'])
def test_write_shape_and_scan_route(kalman):
    '''Isotropic keeps beta a scalar per head, so the chunkwise kernels
    still apply; diagonal returns an explicit asymmetric w, which routes to
    the token recurrence.'''
    m = mixer(kalman)
    x = torch.randn(2, 12, 32, dtype=torch.float64)
    q, k, v = m._heads(m.wq(x), m.wk(x), m.wv(x))
    g, beta0 = m._gates(x)
    assert beta0 is None and not hasattr(m, 'wb')
    beta, w = m._write(x, k, g, beta0)
    if kalman == 'iso':
        assert w is None and beta.shape == (2, 12, 2)
    else:
        assert beta is None and w.shape == (2, 12, 2, 8)


@pytest.mark.parametrize('kalman', ['iso', 'diag'])
def test_streaming_matches_stateless_forward(kalman):
    '''Prefill a block, then step token by token: the uncertainty state has
    to ride along with the memory for these to agree.'''
    m = mixer(kalman)
    x = torch.randn(2, 20, 32, dtype=torch.float64)
    with torch.no_grad():
        full = m(x)
        m.reset_cache(2)
        head = m.decode_step(x[:, :12])
        tail = torch.cat([m.decode_step(x[:, t:t + 1]) for t in range(12, 20)], dim=1)
    assert torch.allclose(head, full[:, :12], rtol=1e-9, atol=1e-9)
    assert torch.allclose(tail, full[:, 12:], rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize('kalman', ['iso', 'diag'])
def test_cache_roundtrip_carries_the_uncertainty(kalman):
    m = mixer(kalman)
    x = torch.randn(2, 10, 32, dtype=torch.float64)
    with torch.no_grad():
        m.reset_cache(2)
        m.decode_step(x[:, :6])
        cache = m.export_cache()
        want = m.decode_step(x[:, 6:])
        m.reset_cache(2)
        m.load_cache(cache)
        got = m.decode_step(x[:, 6:])
    assert 'gain_state' in cache and torch.allclose(got, want, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize('write', ['kalman_iso', 'kalman_diag'])
def test_model_builds_trains_and_round_trips(write):
    torch.manual_seed(0)
    m = build_model('GDNLM', {**ARGS, 'write': write})
    ids = torch.randint(0, 101, (2, 24))
    m(ids).float().mean().backward()
    assert [n for n, p in m.named_parameters() if p.grad is None] == []
    m2 = type(m).from_config(m.config)
    assert m2.config['write'] == write
    assert [p.shape for p in m2.parameters()] == [p.shape for p in m.parameters()]


@pytest.mark.parametrize('write', ['delta', 'kalman_iso', 'kalman_diag'])
def test_slow_metric_probe_runs_for_every_write(write):
    """The trainer's slow metrics call GatedDeltaNet.gate_stats, which reads
    the gates directly. It has to go through _write like forward does: for a
    Kalman mixer _gates alone returns beta=None, and reading the gates
    without the hook asserted inside chunk_scan_vec and killed the LAX2
    smoke at its first slow-metric step (job 30199344)."""
    from infra.train.metrics import MetricCtx
    m = build_model('GDNLM', {**ARGS, 'write': write})
    ctx = MetricCtx(model=m, last_batch=torch.randint(0, 101, (2, 16)))
    out = m.metric_hooks()['slow'][0](ctx)
    for key in ('gdn/alpha_mean', 'gdn/mem_len_median', 'gdn/mem_len_max',
                'gdn/beta_mean', 'gdn/state_rms_max'):
        assert key in out, (write, key, sorted(out))
        assert torch.isfinite(torch.tensor(out[key])), (write, key, out[key])
    # the write-strength metric keeps its meaning: |w| = beta when w = beta k
    assert 0.0 < out['gdn/beta_mean'] < 2.0, out['gdn/beta_mean']


def test_noise_projections_route_to_adamw():
    '''The skinny gate / noise matrices must land in the AdamW group, as wb
    does for the plain delta rule — Muon on a d x H matrix is the wrong
    shape.'''
    from infra.optimizer import build_optimizer
    m = build_model('GDNLM', {**ARGS, 'write': 'kalman_iso', 'gate_proj_optimizer': 'adamw'})
    groups = m.param_groups()
    adamw = {id(p) for k in ('adamw_decay', 'adamw_no_decay') for p in groups[k]}
    for blk in m.blocks:
        for name in ('wr', 'womega'):
            assert id(getattr(blk.att, name).weight) in adamw, name
    # must not raise on the new shapes, and must cover every parameter
    build_optimizer('muon', m, {'lr': 1e-3, 'adamw': {'lr': 1e-3}})


def test_kdn_rejects_the_configurations_it_cannot_serve():
    with pytest.raises(AssertionError, match='kalman must be one of'):
        KalmanDeltaNet(dim=32, head_count=2, key_head_dim=8, value_head_dim=16, kalman='full')
    with pytest.raises(AssertionError, match='per-channel decay'):
        KalmanDeltaNet(dim=32, head_count=2, key_head_dim=8, value_head_dim=16, gate='scalar')
