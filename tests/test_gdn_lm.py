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
    assert set(out) == {'gdn/state_rms_max', 'gdn/alpha_mean', 'gdn/mem_len_median',
                        'gdn/mem_len_max', 'gdn/beta_mean'}
    assert 0 < out['gdn/alpha_mean'] < 1 and out['gdn/mem_len_max'] >= out['gdn/mem_len_median'] > 1
    assert m.metric_hooks()['slow'][0](MetricCtx(model=m)) == {}
    assert m.attn_flops_per_token(2048) == 2 * 18 * 2 * 16 * 32 + 12 * 64 * 2048
    assert set(build(gate='none', delta=False).metric_hooks()['slow'][0](ctx)) == {'gdn/state_rms_max'}
    assert set(build(gate='scalar').metric_hooks()['slow'][0](ctx)) == set(out)


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
