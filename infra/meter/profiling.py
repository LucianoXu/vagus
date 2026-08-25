'''
Where-does-the-time-go layer: run each variant under torch.profiler over the
same harness (variants dict + shared inputs) that compare() uses, print the
top ops, and export a Chrome trace per variant (open in chrome://tracing or
https://ui.perfetto.dev).

Backend notes: CUDA gets kernel-level activity; on MPS torch.profiler only
records CPU-side op dispatch (use torch.mps.profiler + Instruments for
Metal-level detail); CPU is fully supported.
'''

import pathlib
from typing import Callable, Optional

from torch.profiler import ProfilerActivity, profile

from .core import synchronize, infer_device


def profile_variants(
    variants: dict[str, Callable],
    args: tuple = (),
    kwargs: Optional[dict] = None,
    *,
    steps: int = 5,
    warmup: int = 3,
    out_dir: str = 'profile-output',
    row_limit: int = 12,
) -> dict[str, pathlib.Path]:
    '''Profile every variant; returns {name: trace_path}. A variant that
    raises is skipped with a note.'''
    kwargs = kwargs or {}
    device = infer_device(args, kwargs)

    activities = [ProfilerActivity.CPU]
    if device.type == 'cuda':
        activities.append(ProfilerActivity.CUDA)
    sort_by = ('self_cuda_time_total' if device.type == 'cuda'
               else 'self_cpu_time_total')

    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    traces: dict[str, pathlib.Path] = {}
    for name, fn in variants.items():
        try:
            for _ in range(warmup):
                fn(*args, **kwargs)
            synchronize(device)

            with profile(activities=activities, profile_memory=True,
                         record_shapes=True) as prof:
                for _ in range(steps):
                    fn(*args, **kwargs)
                    synchronize(device)
        except Exception as e:  # noqa: BLE001 - keep profiling the others
            print(f'[meter] profile "{name}" failed: {type(e).__name__}: {e}')
            continue

        path = out / f'{name}.trace.json'
        prof.export_chrome_trace(str(path))
        traces[name] = path
        print(f'\n[meter] profile "{name}" ({steps} steps on {device}) '
              f'— trace: {path}')
        print(prof.key_averages().table(sort_by=sort_by, row_limit=row_limit))
    return traces
