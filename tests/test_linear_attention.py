# GatedDeltaNet: the three execution paths must agree with the token
# recurrence (the definition), the streaming path must reproduce the
# stateless forward, and the module must be causal. CPU, fp32/fp64.

import itertools

import pytest
import torch

from infra.components.linear_attention import (
    GatedDeltaNet, HAS_FLA, chunk_scan, recurrent_scan)

FLAGS = list(itertools.product([True, False], [True, False]))   # (gate, delta)


def inputs(B=2, L=37, H=3, dk=8, dv=12, dtype=torch.float64, seed=0):
    torch.manual_seed(seed)
    q = torch.randn(B, L, H, dk, dtype=dtype)
    k = torch.nn.functional.normalize(torch.randn(B, L, H, dk, dtype=dtype), dim=-1)
    v = torch.randn(B, L, H, dv, dtype=dtype)
    g = torch.nn.functional.logsigmoid(torch.randn(B, L, H, dtype=dtype) + 2.0)   # a in (0,1)
    beta = torch.sigmoid(torch.randn(B, L, H, dtype=dtype))
    S0 = 0.3 * torch.randn(B, H, dk, dv, dtype=dtype)
    return q, k, v, g, beta, S0


@pytest.mark.parametrize('gate,delta', FLAGS)
@pytest.mark.parametrize('chunk', [8, 64])       # L=37: partial chunks, and one chunk
def test_chunk_matches_recurrent(gate, delta, chunk):
    q, k, v, g, beta, S0 = inputs()
    g = g if gate else None
    kw = dict(scale=0.5, delta=delta)
    o_r, S_r = recurrent_scan(q, k, v, g, beta, S0, **kw)
    o_c, S_c = chunk_scan(q, k, v, g, beta, S0, chunk_size=chunk, **kw)
    assert torch.allclose(o_c, o_r, atol=1e-12, rtol=1e-12)
    assert torch.allclose(S_c, S_r, atol=1e-12, rtol=1e-12)
    # the fp32 output dtype contract
    o32, _ = chunk_scan(q.float(), k.float(), v.float(), None if g is None else g.float(),
                        beta.float(), S0.float(), chunk_size=chunk, **kw)
    assert o32.dtype == torch.float32
    assert torch.allclose(o32.double(), o_r, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize('gate,delta', FLAGS)
def test_chunk_gradients_match_recurrent(gate, delta):
    q, k, v, g, beta, S0 = inputs(L=19)
    leaves = [t.requires_grad_() for t in (q, k, v, beta, S0)] + ([g.requires_grad_()] if gate else [])
    kw = dict(scale=0.5, delta=delta)

    def loss(fn, **extra):
        o, S = fn(q, k, v, g if gate else None, beta, S0, **kw, **extra)
        return (o * torch.arange(1, o.shape[1] + 1, dtype=o.dtype)[None, :, None, None]).sum() + S.pow(2).sum()

    gr = torch.autograd.grad(loss(recurrent_scan), leaves, allow_unused=True)
    gc = torch.autograd.grad(loss(chunk_scan, chunk_size=8), leaves, allow_unused=True)
    for a, b, leaf in zip(gr, gc, leaves):
        if a is None:            # beta unused when delta=False
            assert b is None or b.abs().max() == 0
            continue
        assert torch.allclose(a, b, atol=1e-11, rtol=1e-11)


def test_scan_empty_and_no_initial_state():
    q, k, v, g, beta, _ = inputs(L=5)
    o1, S1 = recurrent_scan(q, k, v, g, beta, None, scale=1.0, delta=True)
    o2, S2 = chunk_scan(q, k, v, g, beta, None, scale=1.0, delta=True, chunk_size=4)
    assert torch.allclose(o1, o2) and torch.allclose(S1, S2)
    o0, S0 = recurrent_scan(q[:, :0], k[:, :0], v[:, :0], g[:, :0], beta[:, :0], S1,
                            scale=1.0, delta=True)
    assert o0.shape[1] == 0 and torch.equal(S0, S1)


# --- the module ----------------------------------------------------------

def make(gate=True, delta=True, conv=4, seed=0, **kw):
    torch.manual_seed(seed)
    m = GatedDeltaNet(dim=32, head_count=2, key_head_dim=8, value_head_dim=16,
                      short_conv_size=conv, gate=gate, delta=delta, chunk_size=8,
                      impl='torch', layer_count=4, **kw)
    return m.double().eval()


@pytest.mark.parametrize('gate,delta', FLAGS)
@pytest.mark.parametrize('conv', [4, None])
def test_decode_matches_forward(gate, delta, conv):
    m = make(gate, delta, conv)
    x = torch.randn(2, 23, 32, dtype=torch.float64)
    ref = m(x)
    # one prefill block, then single tokens, then a multi-token block
    m.reset_cache(2, None)
    parts = [m.decode_step(x[:, :9])]
    parts += [m.decode_step(x[:, t:t + 1]) for t in range(9, 14)]
    parts.append(m.decode_step(x[:, 14:]))
    out = torch.cat(parts, dim=1)
    assert torch.allclose(out, ref, atol=1e-10, rtol=1e-10)
    assert m.cache_len == 23
    # export / load mid-stream
    m.reset_cache(2, None)
    m.decode_step(x[:, :10])
    snap = m.export_cache()
    a = m.decode_step(x[:, 10:])
    m.load_cache(snap, None)
    b = m.decode_step(x[:, 10:])
    assert torch.equal(a, b) and m.cache_len == 23
    # L == 0 is a no-op
    assert torch.equal(m.decode_step(x[:, :0]), x[:, :0])


def test_causal():
    m = make()
    x = torch.randn(1, 20, 32, dtype=torch.float64)
    y = m(x)
    x2 = x.clone()
    x2[:, 12:] += torch.randn(1, 8, 32, dtype=torch.float64)
    y2 = m(x2)
    assert torch.allclose(y[:, :12], y2[:, :12])
    assert not torch.allclose(y[:, 12:], y2[:, 12:])


def test_module_grad_flows_everywhere():
    m = make().train()
    x = torch.randn(2, 17, 32, dtype=torch.float64, requires_grad=True)
    m(x).pow(2).sum().backward()
    for n, p in m.named_parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0, n


def test_gate_init_and_stats():
    m = make()
    a0 = torch.exp(-m.A_log.exp() * torch.nn.functional.softplus(m.dt_bias))   # a at zero input
    assert (a0 > 0.8).all() and (a0 < 1.0).all()          # dt in [1e-3, 1e-1], A in [1, 16]
    st = m.gate_stats(torch.randn(1, 30, 32, dtype=torch.float64))
    assert set(st) == {'state_rms', 'alpha', 'mem_len', 'beta'}
    assert st['alpha'].shape == (2,) and (st['mem_len'] > 1).all()
    assert set(make(gate=False, delta=False).gate_stats(torch.randn(1, 5, 32, dtype=torch.float64))) == {'state_rms'}
    assert len(m.gate_projections) == 2 and len(make(gate=False).gate_projections) == 1


def test_impl_selection():
    m = make()
    assert m._pick_impl(torch.zeros(1)) == 'torch'
    with pytest.raises(RuntimeError):
        GatedDeltaNet(32, 2, 8, 16, impl='fla') if not HAS_FLA else (_ for _ in ()).throw(RuntimeError)


@pytest.mark.skipif(not (HAS_FLA and torch.cuda.is_available()), reason='needs fla + CUDA')
@pytest.mark.parametrize('gate,delta', FLAGS)
def test_fla_matches_torch(gate, delta):
    from infra.components.linear_attention import fla_scan
    q, k, v, g, beta, S0 = inputs(B=2, L=200, H=4, dk=64, dv=128, dtype=torch.float32)
    dev = 'cuda'
    q, k, v, beta, S0 = (t.to(dev) for t in (q, k, v, beta, S0))
    g = g.to(dev) if gate else None
    kw = dict(scale=64 ** -0.5, delta=delta)
    o_t, S_t = chunk_scan(q, k, v, g, beta, S0, chunk_size=64, **kw)
    o_f, S_f = fla_scan(q.bfloat16(), k.bfloat16(), v.bfloat16(), g, beta, S0, **kw)
    assert torch.allclose(o_f.float(), o_t, atol=3e-2, rtol=3e-2)
    assert torch.allclose(S_f.float(), S_t, atol=3e-2, rtol=3e-2)
