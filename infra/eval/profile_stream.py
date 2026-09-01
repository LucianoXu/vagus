# Throughput-sprint step 1: profile one managed training step and
# attribute time to phases (readout / scoring+manage / measure_stats /
# transport / rest = projections+ffn+backward+python launch).
#
# Phase markers are injected by monkeypatching the unified module's
# functions with record_function wrappers — zero production-code churn.
#
#   python -m infra.eval.profile_stream --ckpt runs/.../ckpt-....pt \
#       [--steps 3] [--batch 8] [--budget 512] [--block 256]

import argparse
from functools import wraps

import torch
from torch.profiler import ProfilerActivity, profile, record_function

import infra.components.unified as U
from infra.components.unified import Health, ManageCfg, stream_hidden
from infra.eval.budget_ppl import load_model


def _mark(name, fn):
    @wraps(fn)
    def wrapped(*a, **k):
        with record_function(name):
            return fn(*a, **k)
    return wrapped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--steps', type=int, default=3)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--budget', type=int, default=512)
    ap.add_argument('--block', type=int, default=256)
    ap.add_argument('--seq', type=int, default=2048)
    args = ap.parse_args()

    device = torch.device('cuda')
    model, _ = load_model(args.ckpt, device)
    model.train()
    mcfg = ManageCfg(block_len=args.block, budget=args.budget)

    U._manage = _mark('PHASE_scoring_manage', U._manage)
    U._measure_stats = _mark('PHASE_measure_stats', U._measure_stats)
    U._combined_readout = _mark('PHASE_readout', U._combined_readout)
    U._transport = _mark('PHASE_transport', U._transport)

    x = torch.randint(0, 32000, (args.batch, args.seq), device=device)
    y = torch.roll(x, -1, dims=1)

    def step():
        for p in model.parameters():
            p.grad = None
        with torch.autocast('cuda', dtype=torch.bfloat16):
            h = stream_hidden(model, x, mcfg, manage=True,
                              use_checkpoint=True, health=Health())
            loss = torch.nn.functional.cross_entropy(
                model.head(h).float().transpose(1, 2), y)
        loss.backward()
        return loss

    step()                                   # warmup
    torch.cuda.synchronize()

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 with_stack=False) as prof:
        for _ in range(args.steps):
            with record_function('TRAIN_STEP'):
                step()
        torch.cuda.synchronize()

    ka = prof.key_averages()
    print(ka.table(sort_by='cuda_time_total', row_limit=40))
    print('\n=== phase attribution (of TRAIN_STEP wall) ===')
    tot_cpu = {e.key: e.cpu_time_total for e in ka}
    step_t = tot_cpu.get('TRAIN_STEP', 1)
    for k in ['PHASE_readout', 'PHASE_scoring_manage', 'PHASE_measure_stats',
              'PHASE_transport']:
        e = next((r for r in ka if r.key == k), None)
        if e:
            print(f'{k:24s} cpu_total {e.cpu_time_total/1e3:10.1f} ms '
                  f'({100*e.cpu_time_total/step_t:5.1f}% of step) '
                  f'cuda_total {e.device_time_total/1e3:10.1f} ms  '
                  f'calls {e.count}')
    print(f'{"TRAIN_STEP":24s} cpu_total {step_t/1e3:10.1f} ms')
    n_tok = args.steps * args.batch * args.seq
    wall = step_t / 1e6 * 1  # cpu_total of TRAIN_STEP across steps, s
    print(f'\napprox tokens/step-wall: {n_tok} tok / {wall:.2f}s '
          f'= {n_tok/max(wall,1e-9)/1e6:.4f} Mtok/s (single GPU, micro={args.batch})')


if __name__ == '__main__':
    main()
