# Gates 0/1 for the unified streaming path (see experiments/unified-tier1.md).
# Gate 0: with management off (or budget never binding) the streaming
#         readout IS softmax attention: logits match forward().
# Gate 1: the training loss through stream_hidden(manage=False) equals
#         the stateless loss (managed.py's 'none' arm uses forward()
#         itself, so this checks the shared-loss claim of the pair).
# Plus: managed path runs, respects budget, backprops finite grads, and
#       the checkpointed backward matches the plain one.

import math

import pytest
import torch

from infra.components.unified import Health, ManageCfg, stream_hidden
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
    # non-binding budget behaves identically to manage=False
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
    assert h.evicted + h.demoted > 0, 'management never triggered'
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
    # with random weights some atoms should still pick the demote exit
    assert h.demoted > 0, 'no atom ever chose the type-change exit'
