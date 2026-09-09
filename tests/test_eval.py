# infra/eval: the record an eval run leaves, subject-independent items
# and pairing, per-task sanity on tiny models over a tiny store.

import json
import math

import numpy as np
import pytest
import torch

from infra.eval import EvalConfig, evaluate
from infra.eval.core import check_tokenizer
from infra.eval.tasks.holdout import metric_rep_at_l
from infra.inference import Generator
from infra.models import build_model
from infra.models.io import export_slim
from infra.tokenizers import meta as tokenizer_meta

tokenizers = pytest.importorskip('tokenizers')

VOCAB = 101
TPP = dict(vocab_size=VOCAB, dim=64, head_dim=16, context_len=64, layer_count=2, qk_norm=True)
GDN = dict(vocab_size=VOCAB, dim=64, layer_count=3, head_count=2, key_head_dim=16, value_head_dim=32,
           gate_rank=8, chunk_size=8, la_impl='torch', context_len=48,
           layer_kinds=['gdn', 'parallel', 'gdn'], parallel_width=32, softmax_head_dim=16,
           softmax_rope=False)
TOK_SHA = tokenizer_meta('mistral32k')['sha256']


@pytest.fixture(scope='module')
def store_dir(tmp_path_factory):
    '''A one-shard vagus-tokens-v1 store of 200 random documents (BOS +
    40..400 tokens each) under the real tokenizer's identity.'''
    d = tmp_path_factory.mktemp('store')
    rng = np.random.default_rng(0)
    docs, offsets = [], [0]
    for _ in range(200):
        n = int(rng.integers(40, 400))
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
                    'tokens': int(len(toks)), 'docs': 200}],
        'total_tokens': int(len(toks)),
    }))
    return d


def _checkpoint(dir_, name, args, tokens_seen, with_run_meta):
    torch.manual_seed(0)
    model = build_model(name, args)
    dir_.mkdir()
    full = {'step': 7, 'tokens_seen': tokens_seen, 'model_name': name, 'model_args': args,
            'tokenizer': {'id': 'mistral32k', 'sha256': TOK_SHA},
            'model': model.state_dict(), 'optimizer': {}, 'loader': {}, 'rng': {}, 'config': {}}
    torch.save(full, dir_ / 'ckpt-00000007.pt')
    export_slim(dir_ / 'ckpt-00000007.pt', dir_ / 'model-final.pt')
    if with_run_meta:
        (dir_ / 'meta.json').write_text(json.dumps({
            'config': {}, 'data': {'source': {'repo_id': 'test', 'revision': '0'},
                                   'tokenizer': {'id': 'mistral32k', 'sha256': TOK_SHA},
                                   'total_tokens': 1000}}))
    return dir_


@pytest.fixture(scope='module')
def subjects(tmp_path_factory):
    root = tmp_path_factory.mktemp('subjects')
    a = _checkpoint(root / 'tpp-run', 'TransformerPP', TPP, tokens_seen=500, with_run_meta=True)
    b = _checkpoint(root / 'gdn-run', 'GDNLM', GDN, tokens_seen=250, with_run_meta=False)
    return a, b


def _config(store_dir, subjects, tmp_path, name='t'):
    a, b = subjects
    return EvalConfig(
        eval_name=name,
        subjects=[{'run': str(a), 'label': 'A'}, {'ckpt': str(b / 'model-final.pt'), 'label': 'B'}],
        tasks=[
            {'name': 'holdout', 'args': {'data_dir': str(store_dir), 'n_seq': 6,
                                         'context_len': [32, 64, 128], 'batch_size': 4, 'rep_l': [4, 16]}},
            {'name': 'degeneration', 'args': {'data_dir': str(store_dir), 'n_prompts': 4, 'prompt_tokens': 8,
                                              'new_tokens': 12, 'seeds': 2, 'batch_size': 4}},
            {'name': 'recall_probe', 'args': {'data_dir': str(store_dir), 'lengths': [64, 128], 'kv_pairs': [4],
                                              'n': 4, 'pool': [10, 90], 'seps': [3, 4], 'batch_size': 2}},
        ],
        device='cpu', dtype='float32', seed=1, n_boot=50,
        out_root=str(tmp_path / 'eval'), registry_dir=str(tmp_path / 'registry'))


@pytest.fixture(scope='module')
def record(store_dir, subjects, tmp_path_factory):
    tmp = tmp_path_factory.mktemp('run')
    run_dir = evaluate(_config(store_dir, subjects, tmp))
    return run_dir, json.loads((run_dir / 'results.json').read_text()), json.loads((run_dir / 'meta.json').read_text())


def test_record_layout(record, subjects):
    run_dir, results, meta = record
    assert run_dir.name.startswith('t-') and (run_dir / 'config.yaml').exists()
    assert (run_dir / 'samples' / 'degeneration' / 'A.jsonl').exists()
    assert (run_dir.parent.parent / 'registry' / f'{run_dir.name}.json').exists()
    assert meta['reference'] == 'A' and [s['label'] for s in meta['subjects']] == ['A', 'B']
    a, b = meta['subjects']
    assert len(a['sha256']) == 64 and a['checkpoint']['tokens_seen'] == 500 and a['max_stream_len'] == 64   # = context_len: L=64 scores, 128 skips
    assert a['train_data']['total_tokens'] == 1000 and b['train_data'] is None    # run meta found beside A only
    assert b['max_stream_len'] is None
    assert set(results) == {'holdout', 'degeneration', 'recall_probe'}
    with pytest.raises(FileExistsError):
        evaluate(EvalConfig(eval_name='t', subjects=[], tasks=[], out_root=str(run_dir.parent)))


def test_holdout_lengths_items_and_pairs(record):
    _, results, _ = record
    hold = results['holdout']['subjects']
    assert hold['A']['witness']['lengths'] == [32, 64] and hold['A']['witness']['lengths_skipped'] == [128]
    assert hold['B']['witness']['lengths'] == [32, 64, 128]
    assert hold['A']['witness']['windows'] == hold['B']['witness']['windows']   # subject-independent items
    for lab in 'AB':
        s, it = hold[lab]['scalars'], hold[lab]['items']
        assert len(it['nll@32']) == 6 and math.isclose(s['ppl@32'], math.exp(s['nll@32']))
        assert 0 < s['entropy@32'] <= math.log(VOCAB) + 1e-6 and 1 <= s['top_p_support@32'] <= VOCAB
        assert set(k.split('@')[0] for k in it) >= {'nll', 'entropy', 'top_p_support', 'rep_4', 'gold_rep_4',
                                                    'rep_excess_4', 'rep_16'}
    exp = hold['A']['witness']['exposure']
    assert exp['tokens_seen'] == 500 and exp['same_store'] is True and exp['expected_views'] == 500 / exp['store_tokens']
    assert hold['B']['witness']['exposure']['same_store'] is None
    pairs = results['holdout']['pairs']['B']
    assert 'nll@64' in pairs and 'nll@128' not in pairs                         # only shared lengths pair
    p = pairs['nll@32']
    assert p['n'] == 6 and p['ci95'][0] <= p['mean_diff'] <= p['ci95'][1]
    assert math.isclose(p["mean_diff"], np.mean(hold["B"]["items"]["nll@32"]) - np.mean(hold["A"]["items"]["nll@32"]), abs_tol=1e-9)


def test_degeneration_items_floor_and_samples(record):
    run_dir, results, _ = record
    deg = results['degeneration']['subjects']
    for lab in 'AB':
        it = deg[lab]['items']
        assert len(it['rep4_greedy']) == 4 and len(it['rep4_floor']) == 4 and len(it['rep4_sampled']) == 8
        assert all(0 <= x <= 1 for v in it.values() for x in v)
    assert deg['A']['items']['rep4_floor'] == deg['B']['items']['rep4_floor']     # same prompts, same references
    assert deg['A']['witness']['prompts'] == deg['B']['witness']['prompts']
    rows = [json.loads(l) for l in (run_dir / 'samples' / 'degeneration' / 'B.jsonl').read_text().splitlines()]
    assert len(rows) == 4 * (1 + 1 + 2) and {r['kind'] for r in rows} == {'floor', 'greedy', 'sampled'}
    assert all(isinstance(r['text'], str) and 'rep4' in r for r in rows)


def test_recall_cells_and_skips(record):
    _, results, _ = record
    rec = results['recall_probe']['subjects']
    assert rec['A']['witness']['cells'] == ['L64_p4'] and rec['A']['witness']['cells_skipped'] == ['L128_p4']
    assert rec['B']['witness']['cells'] == ['L64_p4', 'L128_p4']
    for lab in 'AB':
        it = rec[lab]['items']
        assert len(it['acc_L64_p4']) == 4 and all(0 <= x <= 1 for x in it['acc_L64_p4'] + it['copy_L64_p4'])
        assert all(x <= 0 for x in it['logprob_L64_p4'])
    assert set(results['recall_probe']['pairs']['B']) == {'acc_L64_p4', 'logprob_L64_p4', 'copy_L64_p4'}


def test_determinism(store_dir, subjects, tmp_path, record):
    _, results, _ = record
    again = evaluate(_config(store_dir, subjects, tmp_path, name='t2'))
    r2 = json.loads((again / 'results.json').read_text())
    for task in results:
        for lab in 'AB':
            assert r2[task]['subjects'][lab]['items'] == results[task]['subjects'][lab]['items']


def test_rep_at_l_metric():
    # gold = a period-3 pattern: every token recurs 3 back, so gold_rep_l = 1
    # once l >= 3 (except the first 3 positions, which have no such history)
    T = 12
    targets = torch.tensor([[7, 8, 9] * (T // 3)])
    logp = torch.full((1, T, 20), -10.0)
    pred = torch.tensor([[7, 8, 9] * (T // 3)]).roll(1, dims=1)  # predict the previous token
    logp.scatter_(-1, pred[..., None], 0.0)
    out = metric_rep_at_l(logp, targets, {'rep_l': (1, 3)})
    assert math.isclose(out['gold_rep_3'].item(), (T - 3) / T, abs_tol=1e-6)
    assert out['gold_rep_1'].item() == 0.0                       # no immediate repeats in the text
    assert math.isclose(out['rep_1'].item(), (T - 1) / T, abs_tol=1e-6)   # the prediction is always the previous token
    assert math.isclose(out['rep_excess_1'].item(), (T - 1) / T, abs_tol=1e-6)


def test_score_ids_matches_forward_and_leaves_stream_usable():
    torch.manual_seed(0)
    m = build_model('GDNLM', GDN).eval()
    g = Generator(m, start_id=1)
    ids = torch.randint(2, VOCAB, (2, 19))
    logits = g.score_ids(ids)
    with torch.no_grad():
        ref = m(torch.cat([torch.ones(2, 1, dtype=torch.int64), ids[:, :-1]], 1))
    assert logits.shape == (2, 19, VOCAB) and torch.allclose(logits, ref, atol=1e-5)
    assert torch.equal(g.pending, ids[:, -1]) and g.stream_len == 20
    g.gen_ids(__import__('infra.inference', fromlist=['SamplingConfig']).SamplingConfig(
        max_new_tokens=2, temperature=0, stop_ids=()))                  # scoring then generating works


def test_tokenizer_mismatch_refused(store_dir):
    from infra.eval.core import open_store
    store = open_store(store_dir)

    class S:
        label = 'x'
        meta = {'tokenizer': {'sha256': 'deadbeef'}}
    with pytest.raises(ValueError, match='tokenizer'):
        check_tokenizer(S(), store)
