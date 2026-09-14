# GDNLM: Decodable protocol, streaming == stateless forward via the
# Generator, hybrid pattern, optimizer coverage, metric hook, FLOPs.

import pytest
import torch

from infra.inference import Generator, SamplingConfig
from infra.models import build_model
from infra.models.decodable import Decodable, missing_decodable
from infra.optimizer import build_optimizer
from infra.train.metrics import MetricCtx

ARGS = dict(vocab_size=101, dim=64, layer_count=3, head_count=2, key_head_dim=16,
            value_head_dim=32, gate_rank=8, chunk_size=8, la_impl='torch', context_len=48)


def build(**over):
    torch.manual_seed(0)
    return build_model('GDNLM', {**ARGS, **over}).eval()


def ref_greedy(model, ids, n):
    with torch.no_grad():
        for _ in range(n):
            ids = torch.cat([ids, model(ids)[:, -1].argmax(-1, keepdim=True)], dim=1)
    return ids


def test_protocol_and_stream_len():
    m = build()
    assert isinstance(m, Decodable) and missing_decodable(m) == []
    assert m.max_stream_len is None                      # pure: no positional limit
    assert build(layer_pattern='gdn,softmax').max_stream_len == 48
    assert build(layer_pattern='gdn,gdn,softmax').kinds == ['gdn', 'gdn', 'softmax']


@pytest.mark.parametrize('pattern', ['gdn', 'gdn,softmax'])
def test_generation_matches_full_forward(pattern):
    m = build(layer_pattern=pattern)
    g = Generator(m)
    ids = torch.randint(2, 101, (2, 7))
    cfg = SamplingConfig(max_new_tokens=10, temperature=0, stop_ids=())
    out = g.generate_ids(ids, cfg, max_len=64) if pattern == 'gdn' else g.generate_ids(ids, cfg)
    ref = ref_greedy(m, ids, 10)
    assert torch.equal(out, ref[:, 7:])
    # split prefill + export/load mid-stream
    g.reset(2, max_len=64); g.prefill_ids(ids[:, :3]); g.prefill_ids(ids[:, 3:])
    snap = g.export_state()
    a = g.gen_ids(cfg)
    h = Generator(m); h.load_state(snap)
    assert torch.equal(a, h.gen_ids(cfg)) and torch.equal(a, out)


def test_pure_model_runs_past_context_len():
    m = build()
    ids = torch.randint(2, 101, (1, 3 * ARGS['context_len']))
    assert m(ids).shape == (1, ids.shape[1], 101)


def test_optimizer_coverage_and_gate_routing():
    m = build()
    groups = m.param_groups()
    n_gate = sum(1 for blk in m.blocks for _ in blk.att.gate_projections)
    assert n_gate == 9                                  # wa1, wa2, wb per layer
    assert all(id(p) not in {id(q) for q in groups['muon']}
               for blk in m.blocks for p in blk.att.gate_projections)
    assert all(p.dim() == 2 for p in groups['muon'])
    names = {id(p): n for n, p in m.named_parameters()}
    nd = {names[id(p)] for p in groups['adamw_no_decay']}
    assert any('A_log' in n for n in nd) and any('dt_bias' in n for n in nd)
    opt = build_optimizer('muon', m, dict(lr=1e-3, momentum=0.95, weight_decay=0.1,
                                          adamw=dict(lr=1e-3, weight_decay=0.1)))
    assert sum(len(g['params']) for g in opt.param_groups) == sum(1 for _ in m.parameters())
    m2 = build(gate_proj_optimizer='muon')
    assert len(m2.param_groups()['muon']) == len(groups['muon']) + n_gate


def test_train_step_bf16_autocast_cpu():
    m = build().train()
    ids = torch.randint(2, 101, (2, 20))
    with torch.autocast('cpu', dtype=torch.bfloat16):
        h = m(ids[:, :-1], return_hidden=True)
        loss = torch.nn.functional.cross_entropy(
            (h @ m.head.weight.T).float().flatten(0, 1), ids[:, 1:].flatten())
    loss.backward()
    assert all(p.grad is not None for p in m.parameters())


def test_metric_hook_and_flops():
    m = build(layer_pattern='gdn,gdn,softmax')
    ctx = MetricCtx(model=m, last_batch=torch.randint(2, 101, (2, 30)))
    out = m.metric_hooks()['slow'][0](ctx)
    GATES = {'gdn/state_rms_max', 'gdn/alpha_mean', 'gdn/mem_len_median',
             'gdn/mem_len_max', 'gdn/beta_mean'}
    assert set(out) == GATES | {'attn_logit_max'}
    assert 0 < out['gdn/alpha_mean'] < 1 and out['gdn/mem_len_max'] >= out['gdn/mem_len_median'] > 1
    assert m.metric_hooks()['slow'][0](MetricCtx(model=m)) == {}
    # chunk 8, dk 16, dv 32, delta: 3 * (6*16*32 + 2*8*(3*16 + 2*32)) per head
    per_head = 3 * (6 * 16 * 32 + 2 * 8 * (3 * 16 + 2 * 32))
    assert m.attn_flops_per_token(2048) == 2 * 2 * per_head + 12 * 64 * 2048
    m0 = build(gate='none', delta=False)          # no WY solve: 4 dk dv + 2C (dk + dv)
    assert m0.attn_flops_per_token(2048) == 3 * 2 * 3 * (4 * 16 * 32 + 2 * 8 * (16 + 32))   # 3 layers x 2 heads
    assert set(build(gate='none', delta=False).metric_hooks()['slow'][0](ctx)) == {'gdn/state_rms_max'}
    assert set(build(gate='scalar').metric_hooks()['slow'][0](ctx)) == GATES
    assert set(build().metric_hooks()['slow'][0](ctx)) == GATES          # pure: no softmax probe


def test_config_roundtrip():
    m = build(layer_pattern='gdn,softmax', gate='scalar')
    m2 = type(m).from_config(m.config)
    assert m2.kinds == m.kinds and [p.shape for p in m2.parameters()] == [p.shape for p in m.parameters()]


def test_mixer_protocol():
    from torch import nn
    from infra.components.attention import SoftmaxAttention
    from infra.components.block import Block
    from infra.components.linear_attention import GatedDeltaNet
    from infra.components.cache import WithCache, missing_members
    from infra.components.mixer import Mixer, missing_mixer
    m = build(layer_pattern='gdn,softmax')
    assert isinstance(m, WithCache) and missing_members(WithCache, m) == []
    for blk in m.blocks:
        assert isinstance(blk, WithCache) and missing_members(WithCache, blk) == []
        assert isinstance(blk.att, Mixer) and missing_mixer(blk.att) == []
        assert isinstance(blk.att, WithCache)
    assert sorted(Mixer.__protocol_attrs__) == ['__call__', 'decode_step', 'export_cache', 'forward', 'load_cache', 'reset_cache']
    assert isinstance(m.blocks[0].att, GatedDeltaNet) and isinstance(m.blocks[1].att, SoftmaxAttention)
    with pytest.raises(TypeError, match='not a Mixer'):
        Block(dim=16, mixer=nn.Linear(16, 16), layer_count=1)


# --- intra-layer hybrid (parallel kind) + NoPE ---------------------------

PAR = dict(layer_kinds=['gdn', 'parallel', 'gdn'], softmax_head_dim=16, softmax_rope=False)


def test_decay_init_range_reaches_the_mixer():
    """A_init_range / dt_init_range set the width of the timescale prior the
    model starts from: the resting forgetting rate is A * softplus(dt_bias),
    which at init is A * dt with A ~ U(A_init_range), dt ~ logU(dt_init_range)."""
    import torch.nn.functional as F
    narrow = build(A_init_range=(1.0, 2.0), dt_init_range=(1e-2, 2e-2))
    wide = build(A_init_range=(0.5, 32.0), dt_init_range=(5e-4, 1.5e-1))

    def rest_tau(m):
        att = m.blocks[0].att
        A = att.A_log.exp()[:, None]
        b = att.dt_bias.view(att.head_count, att.key_head_dim)
        return 1.0 / (A * F.softplus(b))

    for m, (lo, hi) in ((narrow, (1.0, 2.0)), (wide, (0.5, 32.0))):
        A = m.blocks[0].att.A_log.exp()
        assert (A >= lo - 1e-4).all() and (A <= hi + 1e-4).all()
    n, w = rest_tau(narrow).log10(), rest_tau(wide).log10()
    assert w.std() > 2 * n.std(), (w.std().item(), n.std().item())
    # and it survives the config round-trip (checkpoints carry model_args)
    m2 = type(wide).from_config(wide.config)
    assert m2.config['A_init_range'] == (0.5, 32.0)
    assert m2.config['dt_init_range'] == (5e-4, 1.5e-1)
    # the parallel branch's mixer gets it too
    par = build(**PAR, A_init_range=(0.5, 32.0)).blocks[1].att
    assert (par.la.A_log.exp() <= 32.0 + 1e-4).all()


def test_layer_kinds_and_parallel_shapes():
    from infra.components.parallel_mixer import ParallelMixer
    m = build(**PAR)
    assert m.kinds == ['gdn', 'parallel', 'gdn'] and m.rope is None and m.max_stream_len is None
    par = m.blocks[1].att
    assert isinstance(par, ParallelMixer)
    assert par.att.head_count == 2 and par.att.in_dim == 64 and par.att.out_width == 32
    assert par.la.head_count == 1 and par.la.out_width == 32
    assert not hasattr(par.att, 'wo') and not hasattr(par.la, 'wo') and par.wo.weight.shape == (64, 64)
    # a parallel layer keeps the softmax layer's 4 d^2 budget (+ gates/norms)
    n_par = sum(p.numel() for p in par.parameters())
    n_sm = sum(p.numel() for p in build(layer_pattern='gdn,softmax,gdn', softmax_head_dim=16).blocks[1].att.parameters())
    assert 4 * 64 * 64 <= n_par < n_sm + 3000
    with pytest.raises(AssertionError, match='layer_kinds'):
        build(layer_kinds=['gdn', 'parallel'])
    with pytest.raises(AssertionError, match='parallel_width'):
        build(layer_kinds=['gdn', 'parallel', 'gdn'], softmax_head_dim=64)
    # RoPE hybrid keeps the window; NoPE whole-softmax layer has none
    assert build(layer_kinds=['gdn', 'parallel', 'gdn'], softmax_head_dim=16).max_stream_len == 48
    assert build(layer_pattern='gdn,softmax', softmax_rope=False).max_stream_len is None


@pytest.mark.parametrize('kinds', [['gdn', 'parallel', 'gdn'], ['softmax', 'gdn', 'parallel']])
def test_parallel_generation_matches_full_forward(kinds):
    m = build(**{**PAR, 'layer_kinds': kinds})
    g = Generator(m)
    ids = torch.randint(2, 101, (2, 7))
    cfg = SamplingConfig(max_new_tokens=10, temperature=0, stop_ids=())
    out = g.generate_ids(ids, cfg, max_len=64)
    assert torch.equal(out, ref_greedy(m, ids, 10)[:, 7:])
    g.reset(2, max_len=64); g.prefill_ids(ids[:, :3]); g.prefill_ids(ids[:, 3:])
    snap = g.export_state()
    assert set(snap['cache']['blocks'][kinds.index('parallel')]['att']) == {'att', 'la'}
    a = g.gen_ids(cfg)
    h = Generator(m); h.load_state(snap)
    assert torch.equal(a, h.gen_ids(cfg)) and torch.equal(a, out)


def test_parallel_metrics_flops_and_groups():
    m = build(**PAR)
    ctx = MetricCtx(model=m, last_batch=torch.randint(2, 101, (2, 30)))
    out = m.metric_hooks()['slow'][0](ctx)
    assert 'attn_logit_max' in out and 'gdn/alpha_mean' in out
    # 2 pure layers (2 heads) + parallel: 1 linear head + softmax at width 32
    per_head = 3 * (6 * 16 * 32 + 2 * 8 * (3 * 16 + 2 * 32))
    assert m.attn_flops_per_token(2048) == 2 * 2 * per_head + per_head + 12 * 32 * 2048
    groups = m.param_groups()
    par = m.blocks[1].att
    muon_ids = {id(p) for p in groups['muon']}
    assert all(id(p) not in muon_ids for p in par.gate_projections)      # branch gates -> AdamW
    assert id(par.wo.weight) in muon_ids and id(par.att.wq.weight) in muon_ids
    opt = build_optimizer('muon', m, dict(lr=1e-3, momentum=0.95, weight_decay=0.1,
                                          adamw=dict(lr=1e-3, weight_decay=0.1)))
    assert sum(len(g['params']) for g in opt.param_groups) == sum(1 for _ in m.parameters())
    m2 = type(m).from_config(m.config)
    assert m2.kinds == m.kinds and [p.shape for p in m2.parameters()] == [p.shape for p in m.parameters()]


@pytest.mark.parametrize('pattern', ['gdn,softmax', 'gdn,parallel'])
def test_softmax_out_gate(pattern):
    m = build(layer_pattern=pattern, softmax_out_gate=True, softmax_head_dim=16)
    att = m.blocks[1].att if pattern == 'gdn,softmax' else m.blocks[1].att.att
    assert att.out_gate and att.wg.weight.shape == (att.out_width, att.in_dim)
    assert any(p is att.wg.weight for p in m.param_groups()['muon'])
    ids = torch.randint(2, 101, (2, 7))
    ref = m(ids)
    m.reset_cache(2, 64)
    out = torch.cat([m.decode_step(ids[:, :4]), m.decode_step(ids[:, 4:])], dim=1)
    assert torch.allclose(out, ref, atol=1e-5)
    m0 = build(layer_pattern=pattern, softmax_head_dim=16)
    assert sum(p.numel() for p in m.parameters()) > sum(p.numel() for p in m0.parameters())
