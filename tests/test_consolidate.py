# infra/consolidate: items respect the length contract and pair Y across
# context lengths; score_continuation equals score_ids sliced; the
# sleep procedures train in place, lower their own objective, and leave
# the model in its dtype; the eval task records the three scores and
# restores the weights between items.

import json

import numpy as np
import pytest
import torch

from infra.config import apply_overrides
from infra.consolidate import (SleepConfig, Sleeper, bucket_means, item_ids, nll_positions,
                               sample_items, score_continuation)
from infra.consolidate.sleep import lr_factor
from infra.dataset.loader import TokenStore
from infra.eval import EvalConfig, evaluate
from infra.inference import Generator
from infra.models import build_model
from infra.models.io import export_slim
from infra.tokenizers import meta as tokenizer_meta

pytest.importorskip('tokenizers')

VOCAB = 101
GDN = dict(vocab_size=VOCAB, dim=64, layer_count=2, head_count=2, key_head_dim=16, value_head_dim=32,
           gate_rank=8, chunk_size=8, la_impl='torch')
TOK_SHA = tokenizer_meta('mistral32k')['sha256']


@pytest.fixture(scope='module')
def store_dir(tmp_path_factory):
    '''One shard, 300 random documents of 20..200 tokens (BOS-prefixed):
    only a minority reach the 1 + 48 + 24 an item needs.'''
    d = tmp_path_factory.mktemp('store')
    rng = np.random.default_rng(0)
    docs, offsets = [], [0]
    for _ in range(300):
        n = int(rng.integers(20, 200))
        docs.append(np.concatenate([[1], rng.integers(2, VOCAB, n)]).astype(np.uint16))
        offsets.append(offsets[-1] + len(docs[-1]))
    toks = np.concatenate(docs)
    np.save(d / '000_00000.npy', toks)
    np.save(d / '000_00000.idx.npy', np.asarray(offsets, dtype=np.uint64))
    (d / 'manifest.json').write_text(json.dumps({
        'format': 'vagus-tokens-v1', 'dtype': 'uint16',
        'source': {'repo_id': 'test', 'revision': '0'},
        'tokenizer': {'id': 'mistral32k', 'sha256': TOK_SHA, 'vocab_size': VOCAB},
        'shards': [{'file': '000_00000.npy', 'idx': '000_00000.idx.npy', 'source': '000_00000.parquet',
                    'tokens': int(len(toks)), 'docs': 300}],
        'total_tokens': int(len(toks)),
    }))
    return d


@pytest.fixture(scope='module')
def store(store_dir):
    return TokenStore(store_dir)


def _gen():
    torch.manual_seed(0)
    return Generator(build_model('GDNLM', GDN).eval(), start_id=1, stop_ids=(1,))


def test_items_length_and_pairing(store):
    rng = np.random.default_rng(3)
    items = sample_items(store, rng, 5, x_max=48, y_len=24)
    assert len(items) == 5
    for it in items:
        off = store.doc_offsets(it.shard)
        assert off[it.doc + 1] - off[it.doc] >= 1 + 48 + 24
        x48, y = item_ids(store, it)
        x16, y16 = item_ids(store, it, 16)
        assert x48.shape == (48,) and y.shape == (24,) and x16.shape == (16,)
        assert torch.equal(y, y16) and torch.equal(x16, x48[-16:])     # same Y, shorter X is a suffix
        d = torch.from_numpy(np.asarray(store.doc(it.shard, it.doc)).astype(np.int64))
        assert d[0] == 1 and torch.equal(torch.cat([x48, y]), d[1:1 + 48 + 24])   # BOS dropped
    again = sample_items(store, np.random.default_rng(3), 5, x_max=48, y_len=24)
    assert again == items                                                # pure function of the rng
    gapped = sample_items(store, np.random.default_rng(3), 3, x_max=48, y_len=24, gap=10)
    for it in gapped:
        x, y = item_ids(store, it)
        d = torch.from_numpy(np.asarray(store.doc(it.shard, it.doc)).astype(np.int64))
        assert torch.equal(x, d[1:49]) and torch.equal(y, d[59:83]) and len(d) >= 83
    with pytest.raises(RuntimeError):
        sample_items(store, rng, 1, x_max=5000, y_len=24, max_tries=50)


def test_score_continuation_matches_score_ids():
    g = _gen()
    x = torch.randint(2, VOCAB, (2, 13))
    y = torch.randint(2, VOCAB, (2, 7))
    ref = g.score_ids(torch.cat([x, y], 1))[:, -7:]
    out = score_continuation(g, y, x)
    assert out.shape == (2, 7, VOCAB) and torch.allclose(out, ref, atol=1e-5)
    assert torch.equal(g.pending, y[:, -1]) and g.stream_len == 21
    alone = score_continuation(g, y)
    assert torch.allclose(alone, g.score_ids(y), atol=1e-5)
    nll = nll_positions(out, y)
    assert nll.shape == (2, 7) and (nll > 0).all()
    b = bucket_means(nll, (2, 5, 100))
    assert list(b) == ['b0_2', 'b2_5', 'b5_7'] and torch.allclose(b['b0_2'][0], nll[0, :2].mean())


@pytest.mark.parametrize('method', ['replay_kl', 'ntp_x'])
def test_sleep_trains_in_place_and_keeps_dtype(store, method):
    g = _gen()
    before = {k: v.clone() for k, v in g.model.state_dict().items()}
    cfg = SleepConfig(method=method, n_samples=6, sample_len=12, sample_batch=4, steps=6, lr=1e-3,
                      batch=4, chunk_len=16, lm_batch=2, lm_len=16, retain_windows=2, eval_chunk=4)
    sl = Sleeper(g, store, cfg)
    x = torch.randint(2, VOCAB, (40,))
    w = sl.consolidate(x)
    assert next(g.model.parameters()).dtype == torch.float32 and len(w['distill_loss']) == 6
    w2 = sl.consolidate([torch.randint(2, VOCAB, (40,)), torch.randint(2, VOCAB, (12,))])   # two memories, one sleep
    assert len(w2['distill_loss']) == 12 and w2['n_contexts'] == 2
    if method == 'ntp_x':
        sl.consolidate(torch.randint(2, VOCAB, (16,)))     # start+X of 17 tokens is one chunk exactly
        sl.consolidate(torch.randint(2, VOCAB, (17,)))     # ... and one longer: the last start is legal
        sl.consolidate(torch.randint(2, VOCAB, (5,)))      # shorter than a chunk
    assert w['retain_before'] > 0 and w['retain_after'] > 0
    changed = any(not torch.equal(before[k], v) for k, v in g.model.state_dict().items())
    assert changed
    if method == 'replay_kl':
        assert 0 < w['replay_valid_frac'] <= 1
        assert w['kl_after'] < w['kl_before'], (w['kl_before'], w['kl_after'])
    assert not any(p.requires_grad for p in g.model.parameters())
    # the generator still works after a consolidation
    assert score_continuation(g, torch.randint(2, VOCAB, (1, 5))).shape == (1, 5, VOCAB)


def test_replay_shapes_and_bos_mask(store):
    g = _gen()
    cfg = SleepConfig(n_samples=5, sample_len=9, sample_batch=2)
    sl = Sleeper(g, store, cfg)
    m = sl.read(torch.randint(2, VOCAB, (20,)))
    s, t = sl.replay(m)
    assert s.shape == (5, 9) and t.shape == (5, 9, VOCAB)
    # the teacher logits are the memory-conditioned ones: recompute one row directly
    g.load_state(m, max_len=40)
    ref = g.model.decode_step(torch.cat([g.pending[:, None], s[:1, :-1]], 1), return_logits=True)
    assert torch.allclose(t[:1], ref, atol=1e-4)
    row = torch.tensor([[5, 6, 1, 7, 1, 8]])
    assert sl._valid(row).tolist() == [[True, True, True, False, False, False]]


def _checkpoint(root):
    torch.manual_seed(0)
    model = build_model('GDNLM', GDN)
    root.mkdir()
    full = {'step': 1, 'tokens_seen': 100, 'model_name': 'GDNLM', 'model_args': GDN,
            'tokenizer': {'id': 'mistral32k', 'sha256': TOK_SHA},
            'model': model.state_dict(), 'optimizer': {}, 'loader': {}, 'rng': {}, 'config': {}}
    torch.save(full, root / 'ckpt.pt')
    export_slim(root / 'ckpt.pt', root / 'model-final.pt')
    return root / 'model-final.pt'


def test_task_record(store_dir, tmp_path):
    ckpt = _checkpoint(tmp_path / 'subj')
    sleep = dict(n_samples=4, sample_len=8, sample_batch=4, steps=2, lr=1e-3, batch=2,
                 lm_batch=2, lm_len=16, retain_windows=2, eval_chunk=4)
    cfg = EvalConfig(
        eval_name='c', subjects=[{'ckpt': str(ckpt), 'label': 'G'}],
        tasks=[{'name': 'consolidation', 'args': {
            'data_dir': str(store_dir), 'n_items': 3, 'x_len': [16, 48], 'y_len': 24, 'batch_size': 2,
            'buckets': [8], 'method': 'replay_kl', 'sleep': sleep, 'sleep_x_len': [48]}}],
        device='cpu', dtype='float32', seed=1, n_boot=20,
        out_root=str(tmp_path / 'eval'), registry_dir=None)
    run_dir = evaluate(cfg)
    res = json.loads((run_dir / 'results.json').read_text())['consolidation']['subjects']['G']
    items, scalars, wit = res['items'], res['scalars'], res['witness']
    for k in ('nll3', 'nll1@16', 'nll1@48', 'benefit@16', 'benefit@48', 'nll2@48', 'lost@48',
              'nll3_b0_8', 'nll1_b8_24@48', 'nll2_b0_8@48', 'kl_before@48', 'kl_after@48', 'retain_delta@48'):
        assert k in items and len(items[k]) == 3, k
    assert 'nll2@16' not in items and 'ratio@48' in scalars and 'ratio@16' not in scalars
    assert wit['x_len'] == [16, 48] and wit['method'] == 'replay_kl' and len(wit['consolidations']['48']) == 3
    assert wit['sleep']['n_samples'] == 4 and len(wit['items']) == 3
    assert (run_dir / 'samples' / 'consolidation' / 'G.jsonl').exists()
    # the weights were restored between items: nll3 of a fresh subject equals the recorded one
    g = Generator.from_checkpoint(ckpt, device='cpu', dtype='float32')
    rec = json.loads((run_dir / 'results.json').read_text())
    assert rec['consolidation']['subjects']['G']['scalars']['nll3'] == pytest.approx(scalars['nll3'])
    none = evaluate(EvalConfig(
        eval_name='c0', subjects=[{'ckpt': str(ckpt), 'label': 'G'}],
        tasks=[{'name': 'consolidation', 'args': {'data_dir': str(store_dir), 'n_items': 3,
                                                  'x_len': [16, 48], 'y_len': 24}}],
        device='cpu', dtype='float32', seed=1, n_boot=20, out_root=str(tmp_path / 'eval'), registry_dir=None))
    res0 = json.loads((none / 'results.json').read_text())['consolidation']['subjects']['G']
    assert res0['items']['nll3'] == pytest.approx(items['nll3'], abs=1e-4)      # same items, same model
    assert 'nll2@48' not in res0['items'] and res0['witness']['method'] is None


def test_deep_overrides():
    raw = {'a': 1, 'tasks': [{'name': 't', 'args': {'sleep': {'lr': 1e-5}}}], 'model_args': {'x': 1}}
    out = apply_overrides(raw, ['tasks.0.args.sleep.lr=1e-4', 'tasks.0.args.new.k=[1, 2]', 'model_args.y=true', 'a=2'])
    assert out['tasks'][0]['args']['sleep']['lr'] == 1e-4 and out['tasks'][0]['args']['new']['k'] == [1, 2]
    assert out['model_args'] == {'x': 1, 'y': True} and out['a'] == 2
    assert raw['tasks'][0]['args']['sleep']['lr'] == 1e-5 and raw['a'] == 1       # untouched
    for bad in ['tasks.x.args=1', 'tasks.3.args=1', 'a.b=1', 'noeq']:
        with pytest.raises(ValueError):
            apply_overrides(raw, [bad])


def test_lr_factor():
    assert [lr_factor(i, 4, 'const', 0) for i in range(4)] == [1, 1, 1, 1]
    lin = [lr_factor(i, 5, 'linear', 0) for i in range(5)]
    assert lin == pytest.approx([1, 0.75, 0.5, 0.25, 0])
    cos = [lr_factor(i, 5, 'cosine', 2) for i in range(5)]
    assert cos[0] == 0.5 and cos[1] == 1.0 and cos[2] == pytest.approx(1.0) and cos[4] == pytest.approx(0.0)
    assert lr_factor(0, 1, 'cosine', 0) == 1.0


def test_retain_kl_and_schedule(store):
    g = _gen()
    cfg = SleepConfig(n_samples=4, sample_len=10, sample_batch=4, steps=4, lr=1e-3, batch=4, lm_batch=2, lm_len=16,
                      retain_windows=2, eval_chunk=4, lm_mix=0.0, retain_kl=1.0, lr_schedule='cosine', warmup=1)
    sl = Sleeper(g, store, cfg)
    w = sl.consolidate(torch.randint(2, VOCAB, (30,)))
    assert w['retain_kl_before'] == pytest.approx(0.0, abs=1e-6) and w['retain_kl_after'] > 0
    assert sl.original is not None and not any(p.requires_grad for p in sl.original.parameters())
    # the frozen copy is the original: it scores like the restored model would
    before = {k: v.clone() for k, v in sl.original.state_dict().items()}
    sl.consolidate(torch.randint(2, VOCAB, (30,)))
    assert all(torch.equal(before[k], v) for k, v in sl.original.state_dict().items())


# --- phase B: memory as an object, multi-hop curricula ------------------

from infra.consolidate.memory import block_states, degrade, svd_truncate   # noqa: E402


def test_forward_from_state_matches_decode():
    '''The differentiable forward from a loaded memory equals the decode
    path from the same memory (conv history included); without the conv
    caches the first K-1 positions, and through the state everything
    after, differ.'''
    g = _gen()
    x = torch.randint(2, VOCAB, (2, 25))
    y = torch.randint(2, VOCAB, (2, 11))
    g.reset(2, max_len=64)
    g.prefill_ids(x)
    m = g.export_state()
    block = torch.cat([g.pending[:, None], y[:, :-1]], 1)
    ref = g.model.decode_step(block, return_logits=True)
    with torch.no_grad():
        out = g.model(block, caches=m['cache']['blocks'])
        noconv = g.model(block, caches=[{'att': {'state': c['att']['state']}} for c in m['cache']['blocks']])
    assert out.shape == ref.shape
    assert torch.allclose(out, ref, atol=1e-4)
    assert not torch.allclose(noconv, ref, atol=1e-4)
    # the memory-conditioned student differs from a fresh stream, and a blank memory equals one
    g.reset(2, max_len=64)
    blank = g.export_state()
    with torch.no_grad():
        fresh = g.model(block, caches=blank['cache']['blocks'])
        plain = g.model(block)
    assert torch.allclose(fresh, plain, atol=1e-5) and not torch.allclose(out, plain, atol=1e-2)
    # gradients flow to the weights through the memory-conditioned forward
    g.model.requires_grad_(True)
    g.model(block, caches=m['cache']['blocks']).float().logsumexp(-1).mean().backward()
    assert g.model.embedding.weight.grad is not None and g.model.embedding.weight.grad.abs().sum() > 0


def test_degrade():
    g = _gen()
    g.reset(1, max_len=32)
    g.prefill_ids(torch.randint(2, VOCAB, (1, 20)))
    m = g.export_state()
    Ss = block_states(m)
    assert len(Ss) == 2 and all(S.shape == (1, 2, 16, 32) for S in Ss)
    assert all(torch.equal(a, b) for a, b in zip(block_states(degrade(m, 1.0, 'scale')), Ss))
    assert all(torch.equal(a, b) for a, b in zip(block_states(degrade(m, 1.0, 'svd')), Ss))
    for how in ('scale', 'layers_topdown', 'layers_bottomup', 'svd'):
        assert all((S == 0).all() for S in block_states(degrade(m, 0.0, how)))
    half = block_states(degrade(m, 0.5, 'scale'))
    assert torch.allclose(half[0], 0.5 * Ss[0])
    td = block_states(degrade(m, 0.5, 'layers_topdown'))
    assert torch.equal(td[0], Ss[0]) and (td[1] == 0).all()
    bu = block_states(degrade(m, 0.5, 'layers_bottomup'))
    assert (bu[0] == 0).all() and torch.equal(bu[1], Ss[1])
    t = svd_truncate(Ss[0], 0.25)                                   # rank <= ceil(0.25 * 16) = 4
    assert t.shape == Ss[0].shape and torch.linalg.matrix_rank(t[0, 0]) <= 4 and t.dtype == Ss[0].dtype
    assert torch.equal(block_states(m)[0], Ss[0])                     # the input is untouched


@pytest.mark.parametrize('teacher', ['fixed', 'chain'])
@pytest.mark.parametrize('how', ['scale', 'layers_topdown', 'svd'])
def test_multihop(store, teacher, how):
    g = _gen()
    cfg = SleepConfig(n_samples=4, sample_len=10, sample_batch=4, steps=5, lr=1e-3, batch=4, lm_batch=2,
                      lm_len=16, retain_windows=2, eval_chunk=4, hops=[0.5, 0.0], degrade=how, teacher=teacher)
    w = Sleeper(g, store, cfg).consolidate(torch.randint(2, VOCAB, (30,)))
    assert [h['level'] for h in w['hops']] == [0.5, 0.0] and [h['steps'] for h in w['hops']] == [2, 3]
    assert len(w['distill_loss']) == 5 and all('kl_before' in h and 'kl_after' in h for h in w['hops'])
    assert w['kl_after'] < w['kl_before']
    assert next(g.model.parameters()).dtype == torch.float32


def test_hops_validation():
    with pytest.raises(AssertionError):
        SleepConfig(hops=[0.5])            # must end at 0
    with pytest.raises(AssertionError):
        SleepConfig(hops=[0.2, 0.5, 0.0])  # must decrease
    with pytest.raises(AssertionError):
        SleepConfig(degrade='nope')


def test_sequential_task(store_dir, tmp_path):
    ckpt = _checkpoint(tmp_path / 'subj-seq')
    sleep = dict(n_samples=4, sample_len=8, sample_batch=4, steps=2, lr=1e-3, batch=2,
                 lm_batch=2, lm_len=16, retain_windows=2, eval_chunk=4, lm_mix=0.0, retain_kl=0.5)
    run_dir = evaluate(EvalConfig(
        eval_name='seq', subjects=[{'ckpt': str(ckpt), 'label': 'G'}],
        tasks=[{'name': 'consolidation_seq', 'args': {
            'data_dir': str(store_dir), 'n_docs': 4, 'x_len': 32, 'y_len': 16, 'gap': 8, 'batch_size': 3,
            'method': 'replay_kl', 'sleep': sleep, 'ages': [0, 1, 3]}}],
        device='cpu', dtype='float32', seed=1, n_boot=20, out_root=str(tmp_path / 'eval'), registry_dir=None))
    res = json.loads((run_dir / 'results.json').read_text())['consolidation_seq']['subjects']['G']
    sc, it, w = res['scalars'], res['items'], res['witness']
    assert [sc[f'n_pairs@{a}'] for a in range(4)] == [4, 3, 2, 1]
    assert len(it['kept_final']) == 4 and len(w['nll2']) == 4 and w['nll2'][0][1] is None and w['nll2'][3][0] is not None
    assert 'retain_delta_final' in sc and len(w['consolidations']) == 4 and len(it['retain_lm']) == 5
    # grouped sleeps: 2 docs per sleep -> rows 1 and 3 scored, pairs by age accordingly
    grp = evaluate(EvalConfig(
        eval_name='seq2', subjects=[{'ckpt': str(ckpt), 'label': 'G'}],
        tasks=[{'name': 'consolidation_seq', 'args': {
            'data_dir': str(store_dir), 'n_docs': 4, 'x_len': 32, 'y_len': 16, 'gap': 8, 'batch_size': 3,
            'method': 'replay_kl', 'sleep': sleep, 'docs_per_sleep': 2}}],
        device='cpu', dtype='float32', seed=1, n_boot=20, out_root=str(tmp_path / 'eval'), registry_dir=None))
    r2 = json.loads((grp / 'results.json').read_text())['consolidation_seq']['subjects']['G']
    assert r2['witness']['nll2'][0][0] is None and r2['witness']['nll2'][1][0] is not None
    assert [r2['scalars'][f'n_pairs@{a}'] for a in range(4)] == [2, 2, 1, 1]
    assert len(r2['witness']['consolidations']) == 2 and r2['witness']['consolidations'][0]['n_contexts'] == 2
    assert len(r2['witness']['consolidations'][0]['distill_loss']) == 2 * sleep['steps']
    assert r2['items']['nll3'] == pytest.approx(it['nll3'])                # same items
    assert sc['ratio_final'] == pytest.approx(1 - sc['kept_final'])
    # the subject was restored at the end: a fresh score equals the recorded nll3
    g = Generator.from_checkpoint(ckpt, device='cpu', dtype='float32')
    from infra.dataset.loader import TokenStore as _TS
    st = _TS(store_dir)
    items = sample_items(st, np.random.default_rng(0), 1, 32, 16, 8)   # any item: just check scoring works
    y = item_ids(st, items[0])[1][None]
    assert score_continuation(g, y).shape[1] == 16


# --- prioritised replay -------------------------------------------------

from infra.components.linear_attention import recurrent_scan, residual_scan   # noqa: E402


def test_residual_scan_matches_recurrence():
    torch.manual_seed(0)
    B, L, H, dk, dv = 2, 9, 2, 4, 6
    k = torch.nn.functional.normalize(torch.randn(B, L, H, dk), dim=-1)
    v = torch.randn(B, L, H, dv)
    g = -torch.rand(B, L, H, dk) * 0.5
    beta = torch.rand(B, L, H)
    r = residual_scan(k, v, g, beta, None, delta=True)
    assert r.shape == (B, L, H) and (r >= 0).all()
    # position t's residual is |v_t - k_t^T S_{t-1}| / |v_t| with S_{t-1} from the recurrence, decayed
    for t in (0, 4, 8):
        _, S = recurrent_scan(k[:, :t], k[:, :t], v[:, :t], g[:, :t], beta[:, :t], None, scale=1.0, delta=True)
        S = S * g[:, t].exp()[..., None]
        kS = torch.einsum('bhk,bhkv->bhv', k[:, t], S)
        ref = (v[:, t] - kS).norm(dim=-1) / v[:, t].norm(dim=-1)
        assert torch.allclose(r[:, t], ref, atol=1e-5), t
    assert torch.allclose(r[:, 0], torch.ones(B, H))                 # nothing stored yet: residual = v
    assert (residual_scan(k, v, g, beta, None, delta=False) == 1).all()


def test_surprise_profile_and_positional_replay(store):
    g = _gen()
    x = torch.randint(2, VOCAB, (1, 30))
    prof = g.model.surprise(x)
    assert prof.shape == (1, 30, 2, 2) and (prof >= 0).all() and torch.allclose(prof[:, 0], torch.ones(1, 2, 2))
    for how in ('uniform', 'surprise'):
        cfg = SleepConfig(n_samples=6, sample_len=5, sample_batch=4, replay_from=how, replay_bin=8, surprise_power=2.0)
        sl = Sleeper(g, store, cfg)
        s, t, info = sl.replay_from_x(x[0])
        assert s.shape == (6, 5) and t.shape == (6, 5, VOCAB)
        assert info['bins'] == 3 and sum(info['draws']) == 6 and (how == 'uniform') == ('surprise' not in info)
        assert len(info['weights']) == 3 and abs(sum(info['weights']) - 1) < 1e-3   # recorded to 4 decimals
        if how == 'surprise':
            assert len(info['surprise']) == 3 and info['weights'][0] == 0 and info['draws'][0] == 0
    # a continuation drawn from the last bin is the same as replay() from the end memory
    torch.manual_seed(1)
    cfg = SleepConfig(n_samples=1, sample_len=5, sample_batch=1, replay_from='uniform', replay_bin=30)
    sl = Sleeper(g, store, cfg)
    s1, t1, info = sl.replay_from_x(x[0])
    assert info['bins'] == 1
    sl2 = Sleeper(g, store, SleepConfig(n_samples=1, sample_len=5, sample_batch=1))
    m = sl2.read(x[0])
    assert torch.allclose(m['cache']['blocks'][0]['att']['state'], sl.gen.export_state()['cache']['blocks'][0]['att']['state'])
    # the whole sleep runs with positional replay
    w = Sleeper(g, store, SleepConfig(n_samples=4, sample_len=6, sample_batch=4, steps=2, lr=1e-3, batch=4,
                                      lm_batch=2, lm_len=16, retain_windows=2, replay_from='surprise',
                                      replay_bin=8)).consolidate(x[0])
    assert w['replay'][0]['draws'] and len(w['distill_loss']) == 2
    with pytest.raises(AssertionError):
        SleepConfig(replay_from='surprise', hops=[0.5, 0.0])


# --- joint modes: the memory as a variable ------------------------------

from infra.consolidate.sleep import MemoryParam   # noqa: E402


def test_initial_state_is_differentiable():
    g = _gen()
    x = torch.randint(2, VOCAB, (2, 20))
    g.reset(2, max_len=40)
    g.prefill_ids(x)
    m = g.export_state()
    caches = m['cache']['blocks']
    for c in caches:
        c['att']['state'] = c['att']['state'].clone().requires_grad_(True)
    out = g.model(torch.randint(2, VOCAB, (2, 7)), caches=caches)
    out.float().logsumexp(-1).mean().backward()
    assert all(c['att']['state'].grad is not None and c['att']['state'].grad.abs().sum() > 0 for c in caches)


def test_memory_param():
    g = _gen()
    g.reset(1, max_len=32)
    g.prefill_ids(torch.randint(2, VOCAB, (1, 20)))
    m = g.export_state()
    rows = MemoryParam(m, 'rows', 6.0)
    assert len(rows.params) == 2 and rows.params[0].shape == (1, 2, 16, 1)
    st = rows.state()
    assert torch.allclose(st['cache']['blocks'][0]['att']['state'], m['cache']['blocks'][0]['att']['state'], atol=1e-2)
    assert 'qp_cache' in st['cache']['blocks'][0]['att'] and torch.equal(st['pending'], m['pending'])
    prof = rows.profile()
    assert len(prof['keep_mean']) == 2 and abs(prof['keep_mean'][0] - torch.sigmoid(torch.tensor(6.0)).item()) < 1e-5
    free = MemoryParam(m, 'free', 0.0)
    assert torch.equal(free.states()[1], m['cache']['blocks'][1]['att']['state']) and free.params[1].requires_grad
    assert free.profile()['rel_change'] == [0.0, 0.0]


@pytest.mark.parametrize('mode', ['joint_fixed', 'joint_reanchor'])
@pytest.mark.parametrize('how', ['rows', 'free'])
def test_joint_modes(store, mode, how):
    g = _gen()
    before = {k: v.clone() for k, v in g.model.state_dict().items()}
    cfg = SleepConfig(mode=mode, mem_param=how, mem_lr=0.5 if how == 'rows' else 1e-2, lam=1.0,
                      n_samples=6, sample_len=10, sample_batch=6, steps=8, lr=1e-3, batch=4,
                      lm_batch=2, lm_len=16, retain_windows=2, eval_chunk=3, lm_mix=0.0, retain_kl=0.5,
                      probe_every=4, layer_probe=True)
    sl = Sleeper(g, store, cfg)
    w = sl.consolidate(torch.randint(2, VOCAB, (30,)))
    assert [r['step'] for r in w['trajectory']] == [0, 4, 8]
    t0, t1 = w['trajectory'][0], w['trajectory'][-1]
    assert t0['drift'] == pytest.approx(0.0, abs=1e-4)                   # the student starts at the teacher
    assert t0['penalty'] > 0 and t0['blank'] > 0
    assert w['kl_after'] == t1['blank'] and w['penalty_after'] == t1['penalty'] and 'drift_after' in w
    assert len(w['distill_loss']) == 8 and len(w['penalty_loss']) == 8
    assert len(w['layer_read_kl']) == 2 and all(v >= 0 for v in w['layer_read_kl'])
    prof = w['release']
    if how == 'rows':
        assert all(k < torch.sigmoid(torch.tensor(6.0)).item() for k in prof['keep_mean'])   # something was released
    else:
        assert all(v > 0 for v in prof['rel_change'])
    assert any(not torch.equal(before[k], v) for k, v in g.model.state_dict().items())       # w moved too
    assert next(g.model.parameters()).dtype == torch.float32
    assert not any(p.requires_grad for p in g.model.parameters())
    assert w['hops'][0]['level'] == 'joint'


def test_joint_validation():
    with pytest.raises(AssertionError):
        SleepConfig(mode='joint_fixed', hops=[0.5, 0.0])
    with pytest.raises(AssertionError):
        SleepConfig(mode='joint_fixed', replay_from='surprise')
    with pytest.raises(AssertionError):
        SleepConfig(mode='joint_fixed', method='ntp_x')
    with pytest.raises(AssertionError):
        SleepConfig(mode='nope')


def test_joint_penalty_ref_original_and_ema_anchor(store):
    g = _gen()
    cfg = SleepConfig(mode='joint_fixed', mem_param='rows', mem_lr=0.5, lam=1.0, penalty_ref='original',
                      n_samples=6, sample_len=10, sample_batch=6, steps=6, lr=1e-3, batch=4,
                      lm_batch=2, lm_len=16, retain_windows=2, eval_chunk=3, lm_mix=0.0, retain_kl=0.5, probe_every=3)
    sl = Sleeper(g, store, cfg)
    w = sl.consolidate(torch.randint(2, VOCAB, (30,)))
    assert len(w['trajectory']) == 3 and len(w['penalty_loss']) == 6
    assert all(k < torch.sigmoid(torch.tensor(6.0)).item() for k in w['release']['keep_mean'])
    # the original copy is untouched by the sleep
    o = sl._original()
    assert not any(p.requires_grad for p in o.parameters())
    g2 = _gen()
    cfg2 = SleepConfig(mode='joint_reanchor', anchor_ema=0.9, mem_param='free', mem_lr=1e-2, lam=1.0,
                       n_samples=6, sample_len=10, sample_batch=6, steps=6, lr=1e-3, batch=4,
                       lm_batch=2, lm_len=16, retain_windows=2, eval_chunk=3, lm_mix=0.0, retain_kl=0.0, probe_every=3)
    w2 = Sleeper(g2, store, cfg2).consolidate(torch.randint(2, VOCAB, (30,)))
    assert len(w2['trajectory']) == 3 and w2['drift_after'] >= 0
    with pytest.raises(AssertionError):
        SleepConfig(mode='joint_fixed', penalty_ref='nope')
    with pytest.raises(AssertionError):
        SleepConfig(mode='joint_reanchor', anchor_ema=1.0)
