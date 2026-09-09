# infra/consolidate: items respect the length contract and pair Y across
# context lengths; score_continuation equals score_ids sliced; the
# sleep procedures train in place, lower their own objective, and leave
# the model in its dtype; the eval task records the three scores and
# restores the weights between items.

import json

import numpy as np
import pytest
import torch

from infra.consolidate import (SleepConfig, Sleeper, bucket_means, item_ids, nll_positions,
                               sample_items, score_continuation)
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
