# Benchmark: full-logits cross-entropy vs chunked/fused CE, fwd+bwd, at
# pretraining loss-path shapes (hidden -> lm_head -> CE only, no blocks).
#
#   python infra/meter/examples/bench_ce.py --device cuda
#
# Cases mirror real steps: 340M micro-batch 8 (16384 rows x d1024) and a
# 1.3B micro-batch 4 (8192 rows x d2048), vocab 32000.

import argparse

import torch
from torch.nn import functional as F

from infra.meter import bench
from infra.components.losses import chunked_cross_entropy

# CUDA/Triton-only, installed manually on GPU boxes (like triton/flash_attn)
try:
    from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
    _liger_flce = LigerFusedLinearCrossEntropyLoss()
except ImportError:
    _liger_flce = None

V = 32000


def make_variant(kind, h, w, t, device, chunk_rows=4096):
    def full():
        h.grad = w.grad = None
        with torch.autocast(device.type, dtype=torch.bfloat16):
            logits = F.linear(h, w)
            loss = F.cross_entropy(logits.reshape(-1, V), t.reshape(-1))
        loss.backward()
        return loss.detach()

    def chunked():
        h.grad = w.grad = None
        with torch.autocast(device.type, dtype=torch.bfloat16):
            loss = chunked_cross_entropy(h, w, t, chunk_rows=chunk_rows)
        loss.backward()
        return loss.detach()

    def liger():
        # mirrors trainer integration: bf16 weight view of the fp32 master
        # (grad flows back to fp32 through the cast); kernel manages its
        # own precision internally, no autocast wrapper needed
        h.grad = w.grad = None
        loss = _liger_flce(w.to(torch.bfloat16), h.reshape(-1, h.shape[-1]),
                           t.reshape(-1))
        loss.backward()
        return loss.detach()

    return {'full': full, 'chunked': chunked, 'liger': liger}[kind]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()
    device = torch.device(args.device)
    torch.manual_seed(0)

    for rows, d, label in ((16384, 1024, '340M shape (B8xT2048, d1024)'),
                           (8192, 2048, '1.3B shape (B4xT2048, d2048)')):
        h = (0.02 * torch.randn(rows, d, device=device)).requires_grad_()
        w = (0.02 * torch.randn(V, d, device=device)).requires_grad_()
        t = torch.randint(0, V, (rows,), device=device)

        # correctness: loss and grads must agree between the two paths
        lf = make_variant('full', h, w, t, device)()
        assert h.grad is not None and w.grad is not None
        gh, gw = h.grad.clone(), w.grad.clone()
        lc = make_variant('chunked', h, w, t, device)()
        assert h.grad is not None and w.grad is not None
        print(f'\n== {label} ==')
        print(f'loss full {lf.item():.6f} vs chunked {lc.item():.6f} | '
              f'grad max rel diff: h {((h.grad-gh).abs().max()/gh.abs().max()).item():.2e}, '
              f'w {((w.grad-gw).abs().max()/gw.abs().max()).item():.2e}')

        if _liger_flce is not None:
            try:
                ll = make_variant('liger', h, w, t, device)()
                print(f'liger loss {ll.item():.6f} | grad max rel diff: '
                      f'h {((h.grad-gh).abs().max()/gh.abs().max()).item():.2e}, '
                      f'w {((w.grad-gw).abs().max()/gw.abs().max()).item():.2e}')
            except Exception as e:
                print(f'liger correctness check failed: {type(e).__name__}: {e}')

        # device passed explicitly: the closures take no tensor args, so
        # bench cannot infer it (and would fall back to the cpu path)
        results = [bench(make_variant('full', h, w, t, device), name='full-logits', device=device),
                   bench(make_variant('chunked', h, w, t, device, 2048), name='chunked-2k', device=device),
                   bench(make_variant('chunked', h, w, t, device, 8192), name='chunked-8k', device=device)]
        if _liger_flce is not None:
            results.append(bench(make_variant('liger', h, w, t, device), name='liger-flce', device=device))
        for r in results:
            if r.error:
                print(f'  {r.name:<12} FAILED: {r.error}')
            else:
                print(f'  {r.name:<12} {r.mean_ms:8.2f} ms | peak {r.peak_mem/2**30:6.2f} GiB')


if __name__ == '__main__':
    main()
