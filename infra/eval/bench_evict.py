# Throughput / peak-memory bench for the eviction-only training step at
# the real geometry (one GPU, one micro-batch; the trainer multiplies by
# world x accum). The stateless compiled forward is benched alongside as
# the reference the managed path is measured against.
#
#   python -m infra.eval.bench_evict --model_recipe recipe/model/sax1_340M.yaml \
#       --batch 8 --budget 128 --block 256 --compile [--ckpt_act] [--plain]

import argparse
import time

import torch
import yaml

from ..components.evict import EvictCfg, Health, stream_hidden
from ..components.losses import make_ce
from ..models import build_model
from ..train.main import _YamlLoader


def bench(step, steps):
    step(); step()                       # warmup (incl. compile)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(steps):
        step()
    torch.cuda.synchronize()
    return time.perf_counter() - t0, torch.cuda.max_memory_allocated() / 2**30


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_recipe', default='recipe/model/sax1_340M.yaml')
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--seq', type=int, default=2048)
    ap.add_argument('--budget', type=int, default=128)
    ap.add_argument('--block', type=int, default=256)
    ap.add_argument('--ring_window', type=int, default=32)
    ap.add_argument('--score', default='lin')
    ap.add_argument('--manage_every', type=int, default=1)
    ap.add_argument('--steps', type=int, default=5)
    ap.add_argument('--compile', action='store_true')
    ap.add_argument('--ckpt_act', action='store_true',
                    help='activation checkpointing per (layer, block) cell')
    ap.add_argument('--plain', action='store_true',
                    help='also bench the stateless compiled forward')
    ap.add_argument('--ce_impl', default='liger')
    args = ap.parse_args()

    device = torch.device('cuda')
    raw = yaml.load(open(args.model_recipe, encoding='utf-8'), _YamlLoader)
    model = build_model(raw['model_name'], raw['model_args']).to(device)
    model.train()
    ce_fn = make_ce(args.ce_impl, z_loss=1e-4, compute_dtype=torch.bfloat16)
    x = torch.randint(0, raw['model_args']['vocab_size'],
                      (args.batch, args.seq), device=device)
    y = torch.roll(x, -1, dims=1)
    n_tok = args.steps * args.batch * args.seq

    cfg = EvictCfg(block_len=args.block, budget=args.budget,
                   ring_window=args.ring_window, score=args.score,
                   manage_every=args.manage_every, compile_cell=args.compile)

    def managed_step():
        for p in model.parameters():
            p.grad = None
        with torch.autocast('cuda', dtype=torch.bfloat16):
            h = stream_hidden(model, x, cfg, manage=True,
                              use_checkpoint=args.ckpt_act, health=Health())
            loss = ce_fn(h, model.head.weight, y)
        loss.backward()
        return loss

    dt, mem = bench(managed_step, args.steps)
    print(f'BENCH evict block={args.block} budget={args.budget} score={args.score} '
          f'manage_every={args.manage_every} compile={args.compile} '
          f'ckpt_act={args.ckpt_act} micro={args.batch}: '
          f'{n_tok} tok / {dt:.2f}s = {n_tok/dt/1e6:.4f} Mtok/s per GPU | '
          f'peak {mem:.1f} GiB', flush=True)

    if args.plain:
        if args.compile and hasattr(model, 'compile_blocks'):
            model.compile_blocks()

        def plain_step():
            for p in model.parameters():
                p.grad = None
            with torch.autocast('cuda', dtype=torch.bfloat16):
                h = model(x, return_hidden=True)
                loss = ce_fn(h, model.head.weight, y)
            loss.backward()
            return loss

        dt, mem = bench(plain_step, args.steps)
        print(f'BENCH plain compile={args.compile} micro={args.batch}: '
              f'{n_tok} tok / {dt:.2f}s = {n_tok/dt/1e6:.4f} Mtok/s per GPU | '
              f'peak {mem:.1f} GiB', flush=True)


if __name__ == '__main__':
    main()
