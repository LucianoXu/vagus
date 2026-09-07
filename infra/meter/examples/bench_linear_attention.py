'''
Race GatedDeltaNet: torch chunk reference (eager / compiled) vs the fla
Triton kernels, with SoftmaxAttention at the same dim as the yardstick.

Scenarios:
    train:   forward + backward on one micro-batch (the number that sets
             step time); variants return the input gradient for the
             max-diff column
    prefill: no-grad forward, L = --seq
    decode:  decode_step, one token per call, state pinned to the
             post-prefill value each call so every variant sees the same
             position

Run (cluster, one A100, via SLURM):
    python -m infra.meter.examples.bench_linear_attention --device cuda --dtype bf16
    python -m infra.meter.examples.bench_linear_attention --which train --batch 16

Defaults are the LAX1-340M layer shape (dim 1024, 4 heads, dk 128,
dv 256, conv 4) at the training context (2048). fla variants need CUDA
+ bf16 + flash-linear-attention installed and are skipped otherwise.
'''

import argparse
import copy
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from infra.components.attention import SoftmaxAttention
from infra.components.linear_attention import HAS_FLA, GatedDeltaNet
from infra.components.pos_embed import RoPE
from infra.meter import compare

DTYPES = {'fp32': torch.float32, 'fp16': torch.float16, 'bf16': torch.bfloat16}


def pick_device() -> str:
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def fla_reason(device, dtype) -> str | None:
    if not HAS_FLA:
        return 'fla not installed (pip install flash-linear-attention)'
    if device.type != 'cuda':
        return f'fla needs cuda, device is {device.type}'
    if dtype == torch.float32:
        return 'fla needs fp16/bf16'
    return None


def make_gdn(args, impl, device, dtype) -> GatedDeltaNet:
    torch.manual_seed(0)
    m = GatedDeltaNet(args.dim, args.heads, args.dk, args.dv, args.conv or None,
                      gate=not args.no_gate, delta=not args.no_delta,
                      chunk_size=args.chunk, impl=impl, layer_count=24)
    return m.to(device=device, dtype=dtype)


def make_softmax(args, device, dtype) -> SoftmaxAttention:
    torch.manual_seed(0)
    rope = RoPE(dim=args.dim, head_dim=64, context_len=args.seq)
    m = SoftmaxAttention(args.dim, args.dim // 64, None, 1, None, qk_norm=True,
                         rope=rope, layer_count=24)
    return m.to(device=device, dtype=dtype)


def variants_for(args, device, dtype, train: bool) -> dict:
    ref = make_gdn(args, 'torch', device, dtype)
    out = {'torch_chunk': ref, 'torch_chunk_compiled': torch.compile(copy.deepcopy(ref))}
    reason = fla_reason(device, dtype)
    if reason is None:
        fla = make_gdn(args, 'fla', device, dtype)
        fla.load_state_dict(ref.state_dict())
        out['fla'] = fla
        out['fla_compiled'] = torch.compile(copy.deepcopy(fla))
    else:
        print(f'[meter] skipping fla variants: {reason}')
    if train:
        for m in out.values():
            m.train()
    else:
        for m in out.values():
            m.eval()
    return out


def bench_train(args, device, dtype):
    x = torch.randn(args.batch, args.seq, args.dim, device=device, dtype=dtype)
    mods = variants_for(args, device, dtype, train=True)

    def step_fn(m):
        def fn(t):
            t = t.detach().requires_grad_()
            m(t).float().pow(2).mean().backward()
            for p in m.parameters():
                p.grad = None
            return t.grad
        return fn

    variants = {name: step_fn(m) for name, m in mods.items()}
    print(f'\n== train (fwd+bwd) == x: {tuple(x.shape)} on {device}')
    compare(variants, (x,), warmup=args.warmup, iters=args.iters)

    sm = make_softmax(args, device, dtype).train()
    print(f'\n== yardstick: SoftmaxAttention (SDPA) fwd+bwd, same dim ==')
    compare({'softmax_sdpa': step_fn(sm), 'softmax_sdpa_compiled': step_fn(torch.compile(sm))},
            (x,), warmup=args.warmup, iters=args.iters, check=False)


def bench_prefill(args, device, dtype):
    x = torch.randn(args.batch, args.seq, args.dim, device=device, dtype=dtype)
    mods = variants_for(args, device, dtype, train=False)
    print(f'\n== prefill == x: {tuple(x.shape)} on {device}')
    with torch.no_grad():
        compare(mods, (x,), warmup=args.warmup, iters=args.iters)
        sm = make_softmax(args, device, dtype).eval()
        print(f'\n== yardstick: SoftmaxAttention (SDPA) prefill ==')
        compare({'softmax_sdpa': sm, 'softmax_sdpa_compiled': torch.compile(sm)},
                (x,), warmup=args.warmup, iters=args.iters, check=False)


def bench_decode(args, device, dtype):
    seq = args.seq
    prefix = torch.randn(args.batch, seq, args.dim, device=device, dtype=dtype)
    x_tok = torch.randn(args.batch, 1, args.dim, device=device, dtype=dtype)

    def pinned(m, step):
        m.reset_cache(args.batch, None)
        m.decode_step(prefix)
        snap = m.export_cache()

        def fn(t):
            m.load_cache(snap, None)
            return step(t)
        return fn

    eager = make_gdn(args, 'torch', device, dtype).eval()
    comp = copy.deepcopy(eager)
    variants = {
        'eager': pinned(eager, eager.decode_step),
        'compiled': pinned(comp, torch.compile(comp.decode_step)),
    }
    print(f'\n== decode == one token/call after a {seq}-token prefill, batch {args.batch}')
    with torch.no_grad():
        compare(variants, (x_tok,), warmup=args.warmup, iters=args.iters)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--seq', type=int, default=2048)
    p.add_argument('--dim', type=int, default=1024)
    p.add_argument('--heads', type=int, default=4)
    p.add_argument('--dk', type=int, default=128)
    p.add_argument('--dv', type=int, default=256)
    p.add_argument('--conv', type=int, default=4, help='short conv kernel, 0 = off')
    p.add_argument('--chunk', type=int, default=64)
    p.add_argument('--no-gate', action='store_true')
    p.add_argument('--no-delta', action='store_true')
    p.add_argument('--dtype', choices=DTYPES, default='bf16')
    p.add_argument('--device', default=None)
    p.add_argument('--warmup', type=int, default=5)
    p.add_argument('--iters', type=int, default=20)
    p.add_argument('--which', choices=['train', 'prefill', 'decode', 'all'], default='all')
    args = p.parse_args()

    device = torch.device(args.device or pick_device())
    dtype = DTYPES[args.dtype]

    probe = make_gdn(args, 'torch', 'cpu', torch.float32)
    n_params = sum(p.numel() for p in probe.parameters())
    print(f'GatedDeltaNet dim={args.dim} heads={args.heads} dk={args.dk} dv={args.dv} '
          f'conv={args.conv or None} gate={not args.no_gate} delta={not args.no_delta} '
          f'({n_params / 1e6:.1f}M params) {args.dtype} | fla installed: {HAS_FLA}')

    if args.which in ('train', 'all'):
        bench_train(args, device, dtype)
    if args.which in ('prefill', 'all'):
        bench_prefill(args, device, dtype)
    if args.which in ('decode', 'all'):
        bench_decode(args, device, dtype)


if __name__ == '__main__':
    main()
