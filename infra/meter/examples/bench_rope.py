'''
Race the RoPE forward: eager vs torch.compile vs triton.

Run from anywhere:
    python infra/meter/examples/bench_rope.py --device mps --seq 4096
    python -m infra.meter.examples.bench_rope --device cuda --dtype bf16

Unavailable backends (e.g. triton on this machine) are skipped with a note.
'''

import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from infra.components.pos_embed import RoPE
from infra.meter import compare
from infra.meter.examples.rope_triton import HAS_TRITON, rope_triton

DTYPES = {'fp32': torch.float32, 'fp16': torch.float16, 'bf16': torch.bfloat16}


def pick_device() -> str:
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--heads', type=int, default=8)
    p.add_argument('--seq', type=int, default=2048)
    p.add_argument('--head-dim', type=int, default=64)
    p.add_argument('--offset', type=int, default=0,
                   help='RoPE position offset (decode-style continuation)')
    p.add_argument('--dtype', choices=DTYPES, default='fp32')
    p.add_argument('--device', default=None)
    p.add_argument('--warmup', type=int, default=10)
    p.add_argument('--iters', type=int, default=50)
    p.add_argument('--profile', action='store_true',
                   help='also run torch.profiler and export Chrome traces')
    args = p.parse_args()

    device = torch.device(args.device or pick_device())
    dtype = DTYPES[args.dtype]

    rope = RoPE(dim=args.heads * args.head_dim,
                head_dim=args.head_dim,
                context_len=args.seq + args.offset).to(device=device, dtype=dtype)
    x = torch.randn(args.batch, args.heads, args.seq, args.head_dim,
                    device=device, dtype=dtype)
    print(f'x: {tuple(x.shape)} {args.dtype} on {device}, offset={args.offset}')

    variants = {
        'eager': rope,
        'compiled': torch.compile(rope),
    }
    if HAS_TRITON and device.type == 'cuda':
        variants['triton'] = lambda t, off=0: rope_triton(
            t, rope.mcos[off:off + t.shape[-2]], rope.msin[off:off + t.shape[-2]])
    else:
        reason = ('triton not installed' if not HAS_TRITON
                  else f'triton needs cuda, device is {device.type}')
        print(f'[meter] skipping triton variant: {reason}')

    with torch.no_grad():
        compare(variants, (x, args.offset), warmup=args.warmup, iters=args.iters)

        if args.profile:
            from infra.meter import profile_variants
            profile_variants(variants, (x, args.offset),
                             out_dir=str(ROOT / 'runs' / 'profiles'))


if __name__ == '__main__':
    main()
