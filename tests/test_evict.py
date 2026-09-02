# Gates for the eviction-only streaming path (infra/components/evict.py).
# Gate 0: management off (or budget never binding) => the streaming
#         readout IS softmax attention (hidden states match forward()).
# Gate 1: training loss through stream_hidden(manage=False) equals the
#         stateless loss.
# Port check: score='p2' reproduces unified.py's eviction-only path
#         (ManageCfg(demote=False)) — the tier-1 m<budget>e cells.
# Gradient sanity: managed backward finite, checkpoint-invariant, and
#         its gradient norm within a small factor of the unmanaged one
#         (the pool amplifier has no counterpart here).

import torch

from infra.components.evict import EvictCfg, Health, stream_hidden
from infra.components.unified import ManageCfg
from infra.components.unified import stream_hidden as stream_unified
from infra.models.transformer_pp import TransformerPP


def tiny_model(seed=0):
    torch.manual_seed(seed)
    return TransformerPP(vocab_size=512, dim=128, head_dim=32,
                         context_len=1024, layer_count=3,
                         tie_embedding=True, qk_norm=True).float().eval()


def tokens(B=2, L=512, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 512, (B, L), generator=g)


CFG = EvictCfg(block_len=64, budget=96, ring_window=16)


def _expected_evicted(L, bl, budget, B, H, layers, every=1):
    n_alive, total = 0, 0
    for bi in range(L // bl):
        n_alive += bl
        if (bi + 1) % every == 0:
            r = max(n_alive - budget, 0)
            if r > 0:
                total += r
                n_alive = budget
    return total * B * H * layers


def test_gate0_stream_equals_forward():
    model, x = tiny_model(), tokens()
    with torch.no_grad():
        ref = model(x, return_hidden=True)
        out = stream_hidden(model, x, CFG, manage=False)
    diff = (ref - out).abs().max().item()
    assert diff < 2e-4, f'gate 0 failed: max hidden diff {diff}'
    loose = EvictCfg(block_len=64, budget=10_000, ring_window=16)
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
            model.head(stream_hidden(model, x, CFG, manage=False)).transpose(1, 2), y)
    assert abs(lref.item() - lstr.item()) < 1e-4


def test_managed_finite_deterministic_counts():
    model, x = tiny_model(), tokens()
    h = Health()
    with torch.no_grad():
        a = stream_hidden(model, x, CFG, manage=True, health=h)
        b = stream_hidden(model, x, CFG, manage=True)
    assert torch.isfinite(a).all()
    assert torch.equal(a, b)
    assert h.evicted == _expected_evicted(512, 64, 96, 2, 4, 3)
    assert h.decisions == 7
    # management must change the readout somewhere
    with torch.no_grad():
        ref = model(x, return_hidden=True)
    assert (ref - a).abs().max().item() > 1e-3


def test_manage_every_two():
    model, x = tiny_model(), tokens()
    cfg = EvictCfg(block_len=64, budget=96, ring_window=16, manage_every=2)
    h = Health()
    with torch.no_grad():
        out = stream_hidden(model, x, cfg, manage=True, health=h)
    assert torch.isfinite(out).all()
    assert h.evicted == _expected_evicted(512, 64, 96, 2, 4, 3, every=2)


def test_p2_matches_unified_evict_only():
    '''The port anchor: score='p2' must reproduce unified.py's
    eviction-only protocol (same ring, same transport, same score form)
    up to SDPA-vs-einsum rounding.'''
    model, x = tiny_model(), tokens()
    ours = EvictCfg(block_len=64, budget=96, ring_window=16, score='p2')
    theirs = ManageCfg(block_len=64, budget=96, ring_window=16, demote=False)
    with torch.no_grad():
        a = stream_hidden(model, x, ours, manage=True)
        b = stream_unified(model, x, theirs, manage=True)
    diff = (a - b).abs().max().item()
    assert diff < 1e-3, f'evict.py(p2) diverges from unified(demote=False): {diff}'


def test_score_forms_and_lookahead_change_selection():
    model, x = tiny_model(), tokens()
    kw = dict(block_len=64, budget=64, ring_window=16)
    with torch.no_grad():
        a = stream_hidden(model, x, EvictCfg(**kw, score='lin'), manage=True)
        b = stream_hidden(model, x, EvictCfg(**kw, score='sq'), manage=True)
        c = stream_hidden(model, x, EvictCfg(**kw, score='lin', lookahead=64),
                          manage=True)
    assert torch.isfinite(b).all() and torch.isfinite(c).all()
    assert not torch.equal(a, b), 'score form has no effect'
    assert not torch.equal(a, c), 'lookahead has no effect'


def test_gumbel_only_when_grad_enabled():
    model, x = tiny_model(), tokens()
    kw = dict(block_len=64, budget=64, ring_window=16)
    det, sto = EvictCfg(**kw), EvictCfg(**kw, gumbel_tau=2.0)
    with torch.no_grad():
        a = stream_hidden(model, x, det, manage=True)
        b = stream_hidden(model, x, sto, manage=True)
    assert torch.equal(a, b), 'gumbel must be inert under no_grad'
    model.train()
    torch.manual_seed(0)
    c = stream_hidden(model, x, sto, manage=True)
    torch.manual_seed(1)
    d = stream_hidden(model, x, sto, manage=True)
    assert not torch.equal(c.detach(), a), 'gumbel has no effect in training'
    assert not torch.equal(c.detach(), d.detach()), 'gumbel is not random'


def test_backward_finite_checkpoint_matches_and_gnorm_sane():
    model, x = tiny_model(), tokens(B=1, L=256)
    model.train()
    y = torch.roll(x, -1, dims=1)

    def grads(manage, use_ckpt):
        for p in model.parameters():
            p.grad = None
        out = stream_hidden(model, x, CFG, manage=manage, use_checkpoint=use_ckpt)
        loss = torch.nn.functional.cross_entropy(
            model.head(out).transpose(1, 2), y)
        loss.backward()
        g = torch.cat([p.grad.flatten() for p in model.parameters()
                       if p.grad is not None])
        assert torch.isfinite(g).all()
        return loss.item(), g.clone()

    l0, g0 = grads(True, False)
    l1, g1 = grads(True, True)
    assert abs(l0 - l1) < 1e-5
    assert (g0 - g1).abs().max().item() < 1e-4, 'checkpointed grads diverge'
    assert g0.abs().max().item() > 0
    _, gp = grads(False, False)
    ratio = g0.norm().item() / gp.norm().item()
    assert 0.2 < ratio < 5.0, f'managed/unmanaged gnorm ratio {ratio}'


def test_compiled_cell_matches_eager():
    '''On CUDA when available (the inductor backend that trains); CPU
    inductor needs a host C++ toolchain, absent on raven's nodes.'''
    import infra.components.evict as E
    E._COMPILED_CELL = None
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, x = tiny_model().to(dev), tokens(B=1, L=256).to(dev)
    kw = dict(block_len=64, budget=64, ring_window=16)
    with torch.no_grad():
        a = stream_hidden(model, x, EvictCfg(**kw, compile_cell=False), manage=True)
        b = stream_hidden(model, x, EvictCfg(**kw, compile_cell=True), manage=True)
    diff = (a - b).abs().max().item()
    assert diff < 5e-4, f'compiled cell diverges from eager: {diff}'
    with torch.no_grad():
        ref = model(x, return_hidden=True)
        c = stream_hidden(model, x, EvictCfg(**kw, compile_cell=True), manage=False)
    assert (ref - c).abs().max().item() < 5e-4
    E._COMPILED_CELL = None
