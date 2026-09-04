# Generator invariants on a tiny random TransformerPP (CPU, fp32).
# Reference = stateless full forward over the whole stream: the streaming
# path must reproduce it position by position.

import pytest
import torch

from infra.inference import Generator, SamplingConfig, sample
from infra.models import build_model
from infra.models.decodable import Decodable, missing_decodable
from infra.models.io import export_slim, load_model, slim_state

ARGS = dict(vocab_size=101, dim=64, head_dim=16, context_len=48,
            layer_count=2, qk_norm=True)


@pytest.fixture
def model():
    torch.manual_seed(0)
    return build_model('TransformerPP', ARGS).eval()


def ref_greedy(model, ids, n):
    '''Greedy continuation via full forward (no cache).'''
    with torch.no_grad():
        for _ in range(n):
            nxt = model(ids)[:, -1].argmax(-1, keepdim=True)
            ids = torch.cat([ids, nxt], dim=1)
    return ids


def test_protocol(model):
    assert isinstance(model, Decodable)          # nominal, via explicit inheritance
    assert missing_decodable(model) == []        # structural
    assert model.max_stream_len == ARGS['context_len']


def test_greedy_matches_full_forward(model):
    g = Generator(model)
    ids = torch.randint(2, 101, (2, 7))
    out = g.generate_ids(ids, SamplingConfig(max_new_tokens=10, temperature=0, stop_ids=()))
    ref = ref_greedy(model, ids, 10)
    assert torch.equal(out, ref[:, 7:])
    assert g.stream_len == 17 and g.stop_reason == 'max_new_tokens'


def test_start_id(model):
    cfg = SamplingConfig(max_new_tokens=5, temperature=0, stop_ids=())
    ids = torch.randint(2, 101, (2, 6))
    g = Generator(model, start_id=7)
    g.reset(2)
    assert g.stream_len == 1 and g.fed_len == 0 and g.pending.tolist() == [7, 7]
    out = g.generate_ids(ids, cfg)
    ref = ref_greedy(model, torch.cat([torch.tensor([[7], [7]]), ids], dim=1), 5)
    assert torch.equal(out, ref[:, 7:])
    assert g.stream_len == 12


def test_prefill_in_pieces_and_after_gen(model):
    cfg = SamplingConfig(max_new_tokens=4, temperature=0, stop_ids=())
    ids = torch.randint(2, 101, (1, 9))
    # one prefill
    g = Generator(model); g.reset(1)
    g.prefill_ids(ids); a = g.gen_ids(cfg)
    # split prefill
    g.reset(1); g.prefill_ids(ids[:, :4]); g.prefill_ids(ids[:, 4:]); b = g.gen_ids(cfg)
    assert torch.equal(a, b)
    # gen, then more prompt, then gen: equals the reference over the
    # concatenated stream (the pending token is fed exactly once)
    more = torch.randint(2, 101, (1, 3))
    g.prefill_ids(more); c = g.gen_ids(cfg)
    stream = torch.cat([ids, a, more], dim=1)
    ref = ref_greedy(model, stream, 4)
    assert torch.equal(c, ref[:, stream.shape[1]:])


def test_stream_equals_gen_and_break_is_safe(model):
    g = Generator(model)
    ids = torch.randint(2, 101, (2, 5))
    cfg = SamplingConfig(max_new_tokens=8, temperature=0.9, top_p=0.9, seed=7, stop_ids=())
    a = g.generate_ids(ids, cfg)
    g.reset(2); g.prefill_ids(ids)
    b = torch.stack(list(g.gen_ids_stream(cfg)), dim=1)
    assert torch.equal(a, b)
    # break after 3 tokens, continue with a fresh gen: same as the
    # reference over what was actually emitted
    g.reset(2); g.prefill_ids(ids)
    it = g.gen_ids_stream(cfg); first = [next(it) for _ in range(3)]; it.close()
    tail = g.gen_ids(SamplingConfig(max_new_tokens=2, temperature=0, stop_ids=()))
    stream = torch.cat([ids, torch.stack(first, 1)], dim=1)
    assert torch.equal(tail, ref_greedy(model, stream, 2)[:, stream.shape[1]:])


def test_export_load_state_mid_generation(model):
    g = Generator(model)
    ids = torch.randint(2, 101, (1, 6))
    cfg = SamplingConfig(max_new_tokens=5, temperature=1.0, seed=3, stop_ids=())
    g.reset(1); g.prefill_ids(ids)
    it = g.gen_ids_stream(cfg); first = [next(it) for _ in range(2)]; it.close()
    snap = g.export_state()
    rest_a = g.gen_ids(SamplingConfig(max_new_tokens=3, temperature=1.0, stop_ids=()))
    h = Generator(model); h.load_state(snap)
    assert h.stream_len == 8
    rest_b = h.gen_ids(SamplingConfig(max_new_tokens=3, temperature=1.0, stop_ids=()))
    assert torch.equal(rest_a, rest_b)   # RNG state travels with the snapshot


def test_stop_ids_and_capacity(model):
    g = Generator(model)
    ids = torch.randint(2, 101, (1, 5))
    ref = ref_greedy(model, ids, 10)[0, 5:]
    stop = int(ref[3])
    first = int((ref == stop).nonzero()[0])   # the random model may repeat
    out = g.generate_ids(ids, SamplingConfig(max_new_tokens=10, temperature=0, stop_ids=(stop,)))
    assert out.shape[1] == first + 1 and int(out[0, -1]) == stop and g.stop_reason == 'stop_id'
    # capacity: context 48, prompt 5 -> at most 43 more fed tokens
    g.reset(1); g.prefill_ids(ids)
    out = g.gen_ids(SamplingConfig(max_new_tokens=100, temperature=0, stop_ids=()))
    assert g.stop_reason == 'capacity' and g.fed_len == 48 and out.shape[1] == 44
    with pytest.raises(RuntimeError):
        g.prefill_ids(torch.zeros(1, 1, dtype=torch.int64))


def test_sample_knobs():
    torch.manual_seed(0)
    logits = torch.randn(4, 50)
    assert torch.equal(sample(logits, SamplingConfig(temperature=0)), logits.argmax(-1))
    top = logits.topk(3, dim=-1).indices
    for _ in range(20):
        t = sample(logits, SamplingConfig(top_k=3))
        assert all(t[i] in top[i] for i in range(4))
    t = sample(logits, SamplingConfig(top_p=1e-6))
    assert torch.equal(t, logits.argmax(-1))


def test_checkpoint_formats(model, tmp_path):
    full = {'step': 5, 'tokens_seen': 10, 'model_name': 'TransformerPP', 'model_args': ARGS,
            'model': model.state_dict(), 'optimizer': {}, 'loader': {}, 'rng': {}, 'config': {}}
    torch.save(full, tmp_path / 'ckpt-00000005.pt')
    slim = export_slim(tmp_path / 'ckpt-00000005.pt', tmp_path / 'model-final.pt', dtype='bfloat16')
    s = torch.load(slim, weights_only=False)
    assert set(s) == {'model_name', 'model_args', 'model', 'step', 'tokens_seen'}
    assert s['model']['blocks.0.att.wq.weight'].dtype == torch.bfloat16
    m_full, meta_full = load_model(tmp_path / 'ckpt-00000005.pt')
    m_slim, meta_slim = load_model(slim, dtype='float32')
    assert meta_full['format'] == 'full' and meta_slim['format'] == 'slim'
    assert meta_slim['step'] == 5
    ids = torch.randint(2, 101, (1, 4))
    a = Generator(m_full).generate_ids(ids, SamplingConfig(max_new_tokens=3, temperature=0, stop_ids=()))
    b = Generator(m_slim).generate_ids(ids, SamplingConfig(max_new_tokens=3, temperature=0, stop_ids=()))
    # bf16 round trip may flip a near-tie; just check shapes and no error
    assert a.shape == b.shape == (1, 3)
    assert not any(v.requires_grad is False for v in m_full.parameters())


tokenizers = pytest.importorskip('tokenizers')


def test_text_roundtrip_and_stream(model):
    from infra.tokenizers import mistral32k
    tok = mistral32k.load()
    torch.manual_seed(0)
    m = build_model('TransformerPP', dict(ARGS, vocab_size=32000)).eval()
    g = Generator(m, tok, start_id=mistral32k.BOS_ID, stop_ids=(mistral32k.BOS_ID,))
    cfg = SamplingConfig(max_new_tokens=6, temperature=0, stop_ids=())
    out = g.generate('The cat', cfg)
    assert len(out) == 1
    g.reset(1)
    assert g.stream_len == 1 and int(g.pending[0]) == mistral32k.BOS_ID
    g.prefill('The cat')
    # start + raw pieces == the tokenizer's own BOS-prepending encode
    assert g.stream_len == len(tok.encode('The cat').ids)
    g.prefill(' sat')                                   # continuation: no second BOS
    # piecewise text is not tokenization-equivalent to the joined text
    # (this legacy-Llama serialization prepends a word marker per encode
    # call, see mistral32k META "leading spaces"); the stream is exactly
    # start + raw pieces of each call, nothing more
    assert g.stream_len == 1 + len(tok.encode('The cat', add_special_tokens=False).ids) \
                             + len(tok.encode(' sat', add_special_tokens=False).ids)
    # text stream == batch text, same prompt
    g.reset(1); g.prefill('The cat')
    pieces = list(g.gen_stream(cfg))
    assert ''.join(p[0] for p in pieces) == out[0]
    # a fresh stream with only start_id is unconditional generation
    g.reset(1); assert g.gen_ids(cfg).shape == (1, 6)
    with pytest.raises(ValueError):
        g.reset(2); g.prefill(['a', 'a much longer prompt'])
