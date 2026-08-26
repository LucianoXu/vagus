'''
Multi-process (DDP-style) benchmarking. Launch under torchrun; every rank
runs the same bench locally, a barrier aligns ranks before the timed region,
and results gather to rank 0 for an imbalance report.

    torchrun --nproc_per_node 4 your_script.py

The measured fn may contain collectives (all_reduce etc.) — then per-rank
times include waiting on the slowest rank, which is exactly the number that
matters for training throughput. Rank imbalance in the report separates
"everyone is slow" from "one straggler drags the group".
'''

from typing import Callable, Optional

import torch.distributed as dist

from ..utils import infer_device
from .core import BenchResult, bench


def dist_bench(
    fn: Callable,
    args: tuple = (),
    kwargs: Optional[dict] = None,
    *,
    name: Optional[str] = None,
    warmup: int = 10,
    iters: int = 50,
) -> Optional[dict[int, BenchResult]]:
    '''Barrier-aligned bench on every rank; returns {rank: BenchResult} on
    rank 0 and None elsewhere.'''
    assert dist.is_initialized(), 'call torch.distributed.init_process_group first'
    rank, world = dist.get_rank(), dist.get_world_size()
    device = infer_device(args, kwargs or {})

    # align ranks so a late-starting process doesn't read as a straggler
    if device.type == 'cuda':
        dist.barrier(device_ids=[device.index or 0])
    else:
        dist.barrier()

    result = bench(fn, args, kwargs, name=name, device=device,
                   warmup=warmup, iters=iters)

    gathered: list = [None] * world
    dist.all_gather_object(gathered, result)
    if rank != 0:
        return None
    return {i: r for i, r in enumerate(gathered)}


def render_dist_table(results: dict[int, BenchResult]) -> str:
    ok = {i: r for i, r in results.items() if not r.failed}
    lines = [f'[meter] world_size={len(results)}']
    header = ['rank', 'device', 'mean', 'std', 'min', 'peak mem']
    rows = [header]
    for i, r in results.items():
        if r.failed:
            rows.append([str(i), r.device, f'FAILED: {r.error}', '', '', ''])
            continue
        mem = 'n/a' if r.peak_mem is None else f'{r.peak_mem / 2**20:.1f} MiB'
        rows.append([str(i), r.device, f'{r.mean_ms:.3f} ms',
                     f'{r.std_ms:.3f}', f'{r.min_ms:.3f}', mem])
    widths = [max(len(row[c]) for row in rows) for c in range(len(header))]
    for j, row in enumerate(rows):
        lines.append('  '.join(c.ljust(w) for c, w in zip(row, widths)).rstrip())
        if j == 0:
            lines.append('  '.join('-' * w for w in widths))

    if len(ok) >= 2:
        means = [r.mean_ms for r in ok.values()]
        worst, best = max(means), min(means)
        lines.append(f'group mean {sum(means) / len(means):.3f} ms | '
                     f'slowest/fastest rank = {worst / best:.2f}x '
                     f'({worst:.3f} / {best:.3f} ms)')
    return '\n'.join(lines)
