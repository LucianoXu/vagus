'''
Race SoftmaxAttention: eager vs torch.compile vs flash-attention.

Two scenarios:
    prefill: full-block causal forward (no cache), L = --seq
    decode:  streaming decode_step, one token per call, after a --seq prefill

Run:
    python -m infra.meter.examples.bench_attention --device cuda --dtype bf16
    python infra/meter/examples/bench_attention.py --which decode --seq 4096

The flash variants need CUDA + fp16/bf16 + flash-attn installed, and are
skipped (with a note) when unavailable. --conv and --v-dim-mult != 1 also
skip flash: the wrapper does not replicate the conv stream state, and
FA2 requires equal K/V head dims.
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
from infra.components.pos_embed import RoPE
from infra.meter import compare

try:
    from flash_attn import flash_attn_func, flash_attn_with_kvcache
    HAS_FLASH = True
except ImportError:
    HAS_FLASH = False

DTYPES = {'fp32': torch.float32, 'fp16': torch.float16, 'bf16': torch.bfloat16}


def pick_device() -> str:
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def flash_reason(args, device, dtype) -> str | None:
    if not HAS_FLASH:
        return 'flash-attn not installed'
    if device.type != 'cuda':
        return f'flash-attn needs cuda, device is {device.type}'
    if dtype == torch.float32:
        return 'flash-attn needs fp16/bf16'
    if args.v_dim_mult != 1:
        return 'FA2 needs equal K/V head dims (--v-dim-mult 1)'
    if args.conv:
        return 'flash wrapper does not replicate the conv stream state'
    return None


def flash_prefill(attn: SoftmaxAttention, x: torch.Tensor) -> torch.Tensor:
    '''forward() with SDPA swapped for flash_attn_func; reuses attn's weights.
    flash-attn handles GQA natively: pass K/V with their own head count.'''
    B, L = x.shape[0], x.shape[1]
    H, Hkv, Dh = attn.head_count, attn.kv_head_count, attn.dim // attn.head_count

    qp, kp, vp = attn.wq(x), attn.wk(x), attn.wv(x)
    # rope wants (..., L, Dh); flash wants (B, L, H, Dh) — transpose back
    q = attn.rope(qp.reshape(B, L, H, Dh).transpose(1, 2)).transpose(1, 2)
    k = attn.rope(kp.reshape(B, L, Hkv, Dh).transpose(1, 2)).transpose(1, 2)
    v = vp.reshape(B, L, Hkv, Dh)

    out = flash_attn_func(q, k, v, causal=True)
    return attn.wo(out.reshape(B, L, -1))


class FlashDecoder:
    '''Streaming decode on flash_attn_with_kvcache, cache in (B, T, H, Dh).'''

    def __init__(self, attn: SoftmaxAttention, batch: int, max_len: int,
                 device, dtype):
        Hkv, Dh = attn.kv_head_count, attn.dim // attn.head_count
        self.attn = attn
        self.k_cache = torch.zeros(batch, max_len, Hkv, Dh, device=device, dtype=dtype)
        self.v_cache = torch.zeros_like(self.k_cache)
        self.seqlens = torch.zeros(batch, dtype=torch.int32, device=device)
        self.offset = 0
        attn.rope.prepare_m(max_len)

    @torch.no_grad()
    def step(self, x: torch.Tensor) -> torch.Tensor:
        attn = self.attn
        B, L = x.shape[0], x.shape[1]
        H, Hkv, Dh = attn.head_count, attn.kv_head_count, attn.dim // attn.head_count

        qp, kp, vp = attn.wq(x), attn.wk(x), attn.wv(x)
        q = attn.rope(qp.reshape(B, L, H, Dh).transpose(1, 2), self.offset).transpose(1, 2)
        k = attn.rope(kp.reshape(B, L, Hkv, Dh).transpose(1, 2), self.offset).transpose(1, 2)
        v = vp.reshape(B, L, Hkv, Dh)

        # appends k/v into the cache at seqlens and attends, in one kernel
        out = flash_attn_with_kvcache(q, self.k_cache, self.v_cache,
                                      k=k.contiguous(), v=v.contiguous(),
                                      cache_seqlens=self.seqlens, causal=True)
        self.offset += L
        self.seqlens += L
        return attn.wo(out.reshape(B, L, -1))


def make_attn(args, device, dtype) -> SoftmaxAttention:
    rope = RoPE(dim=args.dim, head_dim=args.dim // args.heads,
                context_len=args.seq)
    attn = SoftmaxAttention(args.dim, args.heads, args.kv_heads,
                            args.v_dim_mult, args.conv or None, rope=rope)
    return attn.to(device=device, dtype=dtype).eval()


def bench_prefill(args, attn, device, dtype):
    x = torch.randn(args.batch, args.seq, args.dim, device=device, dtype=dtype)

    variants = {
        'eager': attn,
        'compiled': torch.compile(copy.deepcopy(attn)),
    }
    reason = flash_reason(args, device, dtype)
    if reason is None:
        variants['flash_attn'] = lambda t: flash_prefill(attn, t)
    else:
        print(f'[meter] prefill: skipping flash variant: {reason}')

    print(f'\n== prefill == x: {tuple(x.shape)} on {device}')
    with torch.no_grad():
        compare(variants, (x,), warmup=args.warmup, iters=args.iters)


def bench_decode(args, attn, device, dtype):
    # cache_len (a module int attribute) changes every decode step; dynamo
    # specializes module ints, so an unconfigured compile re-specializes
    # per step until the recompile limit, then falls back to eager.
    # Whitelisting the source name makes it a SymInt from the first
    # compile. (The pinned replay below keeps cache_len constant, but a
    # real decode loop needs this — bench the way real use compiles.)
    import torch.compiler.config as compiler_config
    if hasattr(compiler_config, 'dynamic_sources'):
        current = compiler_config.dynamic_sources or ''
        if '.*cache_len' not in current:    # append, never overwrite
            compiler_config.dynamic_sources = ','.join(
                filter(None, [current, '.*cache_len']))
    else:   # torch < 2.7: fall back to the blunt global switch
        torch._dynamo.config.allow_unspec_int_on_nn_module = True

    # The timed fns must be position-stationary: on CUDA the meter defers to
    # triton do_bench, whose repeat count is adaptive, so a fn that advances
    # the cache each call would overflow any preallocated length. Instead
    # every call pins the position back to seq and re-decodes the same slot —
    # state stays constant and all variants compare at the same position.
    seq = args.seq
    max_len = seq + 2
    prefix = torch.randn(args.batch, seq, args.dim, device=device, dtype=dtype)
    x_tok = torch.randn(args.batch, 1, args.dim, device=device, dtype=dtype)

    attn_eager = copy.deepcopy(attn)
    attn_eager.reset_cache(args.batch, max_len)
    attn_eager.decode_step(prefix)

    def eager_step(t):
        attn_eager.cache_len = seq
        return attn_eager.decode_step(t)

    attn_comp = copy.deepcopy(attn)
    attn_comp.reset_cache(args.batch, max_len)
    attn_comp.decode_step(prefix)          # prefill eagerly, compile only L=1
    comp_fn = torch.compile(attn_comp.decode_step)

    def comp_step(t):
        attn_comp.cache_len = seq
        return comp_fn(t)

    variants = {
        'eager': eager_step,
        'compiled': comp_step,
    }

    reason = flash_reason(args, device, dtype)
    if reason is None:
        fd = FlashDecoder(attn, args.batch, max_len, device, dtype)
        fd.step(prefix)

        def flash_step(t):
            fd.offset = seq
            fd.seqlens.fill_(seq)
            return fd.step(t)

        variants['flash_attn'] = flash_step
    else:
        print(f'[meter] decode: skipping flash variant: {reason}')

    print(f'\n== decode == one token/call at position {seq} '
          f'(pinned), batch {args.batch}')
    compare(variants, (x_tok,), warmup=args.warmup, iters=args.iters)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--heads', type=int, default=16)
    p.add_argument('--kv-heads', type=int, default=None,
                   help='K/V head count for GQA (default: = heads, plain MHA)')
    p.add_argument('--dim', type=int, default=1024)
    p.add_argument('--seq', type=int, default=4096)
    p.add_argument('--v-dim-mult', type=int, default=1)
    p.add_argument('--conv', type=int, default=0,
                   help='short conv kernel, 0 = off (flash variants skip if on)')
    p.add_argument('--dtype', choices=DTYPES, default='bf16')
    p.add_argument('--device', default=None)
    p.add_argument('--warmup', type=int, default=10)
    p.add_argument('--iters', type=int, default=50)
    p.add_argument('--which', choices=['prefill', 'decode', 'all'], default='all')
    args = p.parse_args()

    device = torch.device(args.device or pick_device())
    dtype = DTYPES[args.dtype]
    torch.manual_seed(0)

    attn = make_attn(args, device, dtype)
    n_params = sum(p.numel() for p in attn.parameters())
    print(f'SoftmaxAttention dim={args.dim} heads={args.heads} '
          f'kv_heads={attn.kv_head_count} '
          f'v_mult={args.v_dim_mult} conv={args.conv or None} '
          f'({n_params / 1e6:.1f}M params) {args.dtype}')

    if args.which in ('prefill', 'all'):
        bench_prefill(args, attn, device, dtype)
    if args.which in ('decode', 'all'):
        bench_decode(args, attn, device, dtype)


if __name__ == '__main__':
    main()
