'''
Where the KDA layer's A100 time goes, and what the fused kernel path
buys back.

HAX1 trains at ~0.29 MFU on A100 against SAX2's 0.535 at the same
coordinate, and fla's chunk kernels are tuned on H100. Four modes,
increasing in how much they can be trusted and decreasing in how much
they explain:

  parts   attribution: the pieces of one GatedDeltaNet layer timed
          separately (projections, short conv, SiLU + L2 norm, gate
          activation, the fla scan itself, the output norm-gate). Eager,
          so it says what the work *is*, not what it costs after
          inductor has had it.
  layer   the race: the layer fwd+bwd with the fusions switched on one
          at a time (conv / l2norm / gate / beta / norm_gate and
          combinations, plus disable_recompute, chunk 32 and the bounded
          gate), at both HAX1 shapes.
  profile the compiled model step's GPU time, split into
          linear-attention kernels / GEMMs / the rest — the number that
          says whether more kernel work on this path is worth doing.
  model   the whole HAX1-340M step over the same specs, in tokens/s,
          MFU and peak memory. Single GPU, no sharding — good for the
          memory column and for catching an OOM, unreliable as a compile
          verdict (five timed iterations of a freshly compiled model
          measure dynamo as much as the kernels).
  sweep   read back what recipe/slurm/raven_gpudev_sweep.sbatch
          measured. THIS is the verdict: a real recipe, compiled,
          sharded, twenty steps. The eager `layer` race and the compiled
          training step disagree, and they disagree for a reason —
          inductor already fuses the elementwise chain it can see, so a
          fusion only pays if it removes work inductor could not have
          removed. Measure here before believing a microbenchmark.

Shapes (--shape):
  pure      the pure KDA layers, 16 of 24: dim 1024, 4 heads x (128, 256)
  parallel  the KDA branch of a parallel block, 8 of 24: 2 heads, so
            half the grid of the pure layer at the same token count

Run (cluster, one A100 via SLURM; needs fla + bf16 + CUDA):
    python -m infra.meter.examples.bench_kda_kernel --which parts
    python -m infra.meter.examples.bench_kda_kernel --which layer --shape both
    python -m infra.meter.examples.bench_kda_kernel --which model
and, after a sweep job, anywhere (no GPU):
    python -m infra.meter.examples.bench_kda_kernel --which sweep --runs runs
'''

import argparse
import contextlib
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch.nn import functional as F

from infra.components.linear_attention import HAS_FLA, GatedDeltaNet
from infra.meter import compare
from infra.meter.core import bench

SHAPES = {
    # name        dim   H  dk   dv
    'pure':     (1024, 4, 128, 256),
    'parallel': (1024, 2, 128, 256),
}


def make(shape, args, *, fused=False, disable_recompute: bool = False,
         chunk: int | None = None, lower_bound: float | None = None) -> GatedDeltaNet:
    '''One layer at `shape`. `fused` is a FUSIONS spec: False, True, or
    a subset ('scan', 'conv,scan', ...).'''
    dim, H, dk, dv = SHAPES[shape]
    torch.manual_seed(0)
    # a pure layer owns its output projection; a parallel branch does not
    return GatedDeltaNet(dim, H, dk, dv, args.conv or None, gate=args.gate,
                         delta=not args.no_delta, chunk_size=chunk or args.chunk,
                         impl='fla', fused=fused, disable_recompute=disable_recompute,
                         gate_lower_bound=lower_bound,
                         layer_count=24, out_proj=(shape == 'pure')
                         ).to(device='cuda', dtype=torch.bfloat16)


def train_step(m):
    '''fwd + bwd on the layer, the shape of the number that sets step
    time. Returns the input gradient so a caller can diff variants.'''
    def fn(x):
        x = x.detach().requires_grad_()
        m(x).float().pow(2).mean().backward()
        g = x.grad
        for p in m.parameters():
            p.grad = None
        return g
    return fn


def input_for(shape, args):
    dim = SHAPES[shape][0]
    torch.manual_seed(1)
    return torch.randn(args.batch, args.seq, dim, device='cuda', dtype=torch.bfloat16)


# ---------------------------------------------------------------------------
# parts: attribution of the unfused path
# ---------------------------------------------------------------------------

def bench_parts(args, shape):
    '''Each piece of the unfused layer, on its own inputs, fwd+bwd. The
    pieces do not add up to the whole exactly (launch and allocator
    overlap differ) but their shares are what the attribution needs.'''
    from infra.components.linear_attention import fla_scan

    m = make(shape, args, fused=False)
    x = input_for(shape, args)
    dim, H, dk, dv = SHAPES[shape]
    B, L = args.batch, args.seq

    def grad_of(fn, *inputs):
        def run():
            ins = [t.detach().requires_grad_() for t in inputs]
            out = fn(*ins)
            out = out[0] if isinstance(out, tuple) else out
            out.float().pow(2).mean().backward()
            for p in m.parameters():
                p.grad = None
            return ins[0].grad
        return run

    qp, kp, vp = m.wq(x), m.wk(x), m.wv(x)
    q, k, v = m._heads(qp, kp, vp)
    with torch.no_grad():                  # constants for the scan piece:
        g_act, beta_act = m._gates(x)      # backward must not re-enter _gates
    o_scan = v.detach().clone()

    pieces = {
        'proj q,k,v,g': grad_of(lambda t: m.wq(t) + m.wk(t) + m.wv(t)[..., :H * dk]
                                + m.wg(t)[..., :H * dk], x),
        'short conv x3': grad_of(
            lambda a, b, c: m.conv_q(a) + m.conv_k(b) + m.conv_v(c)[..., :H * dk],
            qp, kp, vp),
        'silu + l2norm q,k': grad_of(lambda a, b: m._l2norm(F.silu(a).view(B, L, H, dk))
                                     + m._l2norm(F.silu(b).view(B, L, H, dk)), qp, kp),
        'gate act -> fp32 g': grad_of(lambda t: m._gates(t)[0], x),
        'scan (fla, unfused)': grad_of(
            lambda a, b, c: fla_scan(a, b, c, g_act, beta_act, None, scale=m.scale,
                                     delta=m.delta, chunk_size=m.chunk_size)[0], q, k, v),
        'norm-gate out': grad_of(lambda a, t: m._output(a, t), o_scan, x),
        'whole layer': grad_of(lambda t: m(t), x),
    }
    print(f'\n== parts ({shape}: dim {dim}, H {H}, dk {dk}, dv {dv}) '
          f'x {tuple(x.shape)} fwd+bwd ==')
    results = {name: bench(fn, warmup=args.warmup, iters=args.iters, name=name)
               for name, fn in pieces.items()}
    whole = results['whole layer'].mean_ms
    print(f'  {"piece":<22} {"ms":>9} {"share":>8}')
    for name, r in results.items():
        if r.failed:
            print(f'  {name:<22} {"FAILED":>9}   {r.error}')
            continue
        print(f'  {name:<22} {r.mean_ms:>9.3f} {r.mean_ms / whole:>7.1%}')


# ---------------------------------------------------------------------------
# layer: the fusions, one at a time
# ---------------------------------------------------------------------------

def layer_variants(args, shape) -> dict:
    base = make(shape, args, fused=False)
    state = base.state_dict()

    def like(**kw):
        m = make(shape, args, **kw)
        m.load_state_dict(state)
        return m

    out = {
        'base': base,
        'conv': like(fused='conv'),
        'l2norm': like(fused='l2norm'),
        'gate': like(fused='gate'),
        'beta': like(fused='beta'),
        'norm_gate': like(fused='norm_gate'),
        'conv,gate,beta': like(fused='conv,gate,beta'),
        'all': like(fused=True),
    }
    if args.compile:
        return out          # the compiled race is about the three stages
    out.update({
        'all+no_recompute': like(fused=True, disable_recompute=True),
        'all+chunk32': like(fused=True, chunk=32),
        # bounded decay: a different gate family, not the same model.
        # Here for the size of the prize — it puts the intra-chunk
        # diagonal blocks on the tensor cores.
        'all+bounded(-5)*': like(fused=True, lower_bound=-5.0),
    })
    return out


def bench_layer(args, shape):
    x = input_for(shape, args)
    mods = layer_variants(args, shape)
    variants = {}
    for name, m in mods.items():
        m.train()
        fn = train_step(m)
        variants[name] = torch.compile(fn) if args.compile else fn

    dim, H, dk, dv = SHAPES[shape]
    print(f'\n== layer ({shape}: dim {dim}, H {H}, dk {dk}, dv {dv}) '
          f'x {tuple(x.shape)} fwd+bwd ==')
    compare(variants, (x,), warmup=args.warmup, iters=args.iters, check=False)
    check_layer(mods, x)


def check_layer(mods, x):
    '''Every variant against `base`: forward output and gradients. The
    fusions are meant to be the same computation, so these should sit at
    bf16 rounding, not at a different answer. The starred variants are a
    different model and are left out.'''
    mods = {k: m for k, m in mods.items() if not k.endswith('*')}
    ref = None
    rows = []
    for name, m in mods.items():
        t = x.detach().requires_grad_()
        y = m(t)
        y.float().pow(2).mean().backward()
        gx = t.grad
        gw = m.wo.weight.grad if m.out_proj else m.wq.weight.grad
        for p in m.parameters():
            p.grad = None
        if ref is None:
            ref = (y.float(), gx.float(), gw.float())
            rows.append((name, 0.0, 0.0, 0.0, 0.0))
            continue
        dy = y.float() - ref[0]
        rows.append((
            name,
            dy.abs().max().item(),
            (dy.norm() / ref[0].norm()).item(),
            (gx.float() - ref[1]).abs().max().item(),
            (gw.float() - ref[2]).abs().max().item(),
        ))
    y_max = ref[0].abs().max().item()
    print(f'\n  agreement vs base (max|y| {y_max:.4f}; bf16 carries ~0.4% per rounding)')
    print(f'  {"variant":<20} {"max|dy|/max|y|":>15} {"rel L2":>10} '
          f'{"max|dx.grad|":>14} {"max|dw.grad|":>14}')
    for name, dy, rel, dgx, dgw in rows:
        print(f'  {name:<20} {dy / y_max:>15.2e} {rel:>10.2e} {dgx:>14.3e} {dgw:>14.3e}')


# ---------------------------------------------------------------------------
# model: the whole HAX1 step
# ---------------------------------------------------------------------------

def build_hax(args, fused=False, disable_recompute=False):
    import yaml

    from infra.models import build_model
    recipe = yaml.safe_load(pathlib.Path(args.model_recipe).read_text())
    margs = dict(recipe['model_args'])
    margs.update(la_fused=fused, la_disable_recompute=disable_recompute,
                 chunk_size=args.chunk)   # fused: a FUSIONS spec
    torch.manual_seed(0)
    model = build_model(recipe['model_name'], margs)
    return model.to(device='cuda', dtype=torch.bfloat16), margs['vocab_size']


def bench_model(args):
    '''One variant at a time: three 340M models plus their activation
    graphs do not share a 40GB card, and the peak-memory column is only
    honest with one model resident.'''
    import gc

    tokens = args.batch * args.seq
    print(f'\n== model ({args.model_recipe}) tokens {tokens} fwd+bwd, '
          f'compile={args.compile} ==')
    print(f'  {"variant":<20} {"ms":>9} {"tok/s":>10} {"MFU":>7} {"peak GB":>9} {"vs base":>8}')

    ref_ms = None
    for name, kw in (('base', dict(fused=False)),
                     ('conv', dict(fused='conv')),
                     ('l2norm', dict(fused='l2norm')),
                     ('gate', dict(fused='gate')),
                     ('norm_gate', dict(fused='norm_gate')),
                     ('conv,gate,beta', dict(fused='conv,gate,beta')),
                     ('all', dict(fused=True))):
        model, vocab = build_hax(args, **kw)
        model.train()
        torch.manual_seed(2)
        tok = torch.randint(0, vocab, (args.batch, args.seq), device='cuda')

        def step(t, m=model):
            m(t).float().pow(2).mean().backward()
            for p in m.parameters():
                p.grad = None

        fn = torch.compile(step) if args.compile else step
        n = sum(p.numel() for p in model.parameters())
        # the training loop's MFU numerator
        flops_per_tok = 6 * n + model.attn_flops_per_token(args.seq)
        r = bench(fn, (tok,), warmup=args.warmup, iters=max(args.iters // 2, 5), name=name)
        if r.failed:
            print(f'  {name:<20} {"FAILED":>9}   {r.error}')
        else:
            ref_ms = ref_ms if ref_ms is not None else r.mean_ms
            toks = tokens / (r.mean_ms / 1e3)
            gb = (r.peak_mem or 0) / 2**30
            print(f'  {name:<20} {r.mean_ms:>9.1f} {toks:>10,.0f} '
                  f'{flops_per_tok * toks / 312e12:>7.3f} {gb:>9.1f} '
                  f'{ref_ms / r.mean_ms:>7.2f}x')
        del model, fn, step, tok
        gc.collect()
        torch.cuda.empty_cache()


def profile_model(args):
    '''Where a HAX1 step's GPU time actually goes, compiled (--compile)
    or not. The one number that says whether more kernel work on the
    linear-attention path is worth doing: what share of the step the
    chunk kernels hold.'''
    from torch.profiler import ProfilerActivity, profile
    model, vocab = build_hax(args, fused=args.fused)
    model.train()
    torch.manual_seed(2)
    tok = torch.randint(0, vocab, (args.batch, args.seq), device='cuda')

    def step(t):
        model(t).float().pow(2).mean().backward()
        for p in model.parameters():
            p.grad = None

    fn = torch.compile(step) if args.compile else step
    for _ in range(5):                      # includes the compile
        fn(tok)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(3):
            fn(tok)
        torch.cuda.synchronize()
    print(f'\n== profile (fused={args.fused!r}, compile={args.compile}, '
          f'batch {args.batch}) top CUDA kernels ==')
    ev = prof.key_averages()
    print(ev.table(sort_by='self_cuda_time_total', row_limit=30))
    total = sum(e.self_device_time_total for e in ev)
    def share(pred):
        return sum(e.self_device_time_total for e in ev if pred(e.key)) / max(total, 1)
    kda = share(lambda n: any(t in n.lower() for t in (
        'chunk_kda', 'chunk_gated_delta', 'chunk_gla', 'wy_fast', 'solve_tril',
        'cumsum', 'l2norm', 'recompute_w_u', 'delta_rule')))
    gemm = share(lambda n: any(t in n.lower() for t in (
        'gemm', 'cutlass', 'sm80_', 'ampere_', 'nn_align', 'nt_align', 'tn_align')))
    print(f'  linear-attention kernels: {kda:6.1%} of GPU time')
    print(f'  GEMM kernels:             {gemm:6.1%}')
    print(f'  everything else:          {1 - kda - gemm:6.1%}')


# ---------------------------------------------------------------------------
# sweep: read back what raven_gpudev_sweep.sbatch measured
# ---------------------------------------------------------------------------

def read_sweep(args):
    '''Tabulate the last tokens_per_s / mfu / max_mem_gb / loss_ce of
    every run under --runs. The compiled, sharded training step is the
    measurement that settles a kernel-path question; this is how to read
    it back.'''
    import glob

    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    dirs = sorted(glob.glob(f'{args.runs}/*/tb'))
    if not dirs:
        print(f'[sweep] no runs under {args.runs}')
        return
    print(f'\n== sweep ({args.runs}) ==')
    print(f'  {"run":<28}{"tok/s":>10}{"MFU":>8}{"peak GB":>9}{"loss_ce":>10}')
    for d in dirs:
        events = sorted(glob.glob(d + '/events*'))
        if not events:
            continue
        ea = EventAccumulator(events[0])
        ea.Reload()

        def last(tag):
            try:
                return ea.Scalars(tag)[-1].value
            except Exception:  # noqa: BLE001 - a run may have died early
                return float('nan')

        run = pathlib.Path(d).parent.name
        print(f'  {run:<28}{last("tokens_per_s"):>10,.0f}{last("mfu"):>8.3f}'
              f'{last("max_mem_gb"):>9.2f}{last("loss_ce"):>10.4f}')


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--which', choices=['parts', 'layer', 'model', 'profile', 'sweep', 'all'],
                   default='all')
    p.add_argument('--runs', default='runs', help='sweep: the directory of run dirs to read')
    p.add_argument('--shape', choices=['pure', 'parallel', 'both'], default='both')
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--seq', type=int, default=2048)
    p.add_argument('--conv', type=int, default=4)
    p.add_argument('--chunk', type=int, default=64)
    p.add_argument('--gate', choices=['vector', 'scalar', 'none'], default='vector')
    p.add_argument('--no-delta', action='store_true')
    p.add_argument('--compile', action='store_true')
    p.add_argument('--fused', default=False,
                   help="profile / single-run modes: the FUSIONS spec ('gate', 'conv,gate', ...)")
    p.add_argument('--model-recipe', default='recipe/model/hax1_340M.yaml')
    p.add_argument('--warmup', type=int, default=5)
    p.add_argument('--iters', type=int, default=20)
    args = p.parse_args()

    if args.which == 'sweep':          # reads run dirs, needs no GPU
        read_sweep(args)
        return

    assert torch.cuda.is_available(), 'this bench is about A100 kernels'
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print(f'{torch.cuda.get_device_name(0)} | torch {torch.__version__} | fla {HAS_FLA}')
    try:
        import fla
        print(f'fla {fla.__version__}')
    except ImportError:
        pass

    shapes = ['pure', 'parallel'] if args.shape == 'both' else [args.shape]
    t0 = time.perf_counter()
    if args.which in ('parts', 'all'):
        for s in shapes:
            bench_parts(args, s)
    if args.which in ('layer', 'all'):
        for s in shapes:
            bench_layer(args, s)
    if args.which in ('model', 'all'):
        bench_model(args)
    if args.which == 'profile':
        profile_model(args)
    print(f'\n[bench] {time.perf_counter() - t0:.1f}s total')


if __name__ == '__main__':
    with contextlib.suppress(KeyboardInterrupt):
        main()
