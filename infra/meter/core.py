'''
Generic performance meter: wall-clock time and peak memory for callables
on cpu / cuda / mps. All device quirks are isolated in this file.

Timing is perf_counter around ``fn() + synchronize``, per iteration, so the
same code path works on every backend and yields a distribution.

Memory:
  - cuda: true peak via reset_peak_memory_stats / max_memory_allocated.
  - mps:  sample after every iteration; transient in-kernel peaks are invisible (approximate).
  - cpu:  ru_maxrss delta, process-wide and monotonic (very approximate).
'''

import gc
import resource
import sys
import time
from dataclasses import dataclass
from typing import Callable, Optional

import torch


@dataclass
class BenchResult:
    name: str
    device: str
    iters: int
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    first_call_ms: float        # includes lazy costs (e.g. torch.compile)
    peak_mem: Optional[int]     # bytes; None = unavailable
    mem_exact: bool             # True only when the backend reports a true peak
    error: Optional[str] = None

    @property
    def failed(self) -> bool:
        return self.error is not None


def synchronize(device: torch.device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    elif device.type == 'mps':
        torch.mps.synchronize()


def _empty_cache(device: torch.device):
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    elif device.type == 'mps':
        empty = getattr(torch.mps, 'empty_cache', None)
        if empty is not None:
            empty()


def infer_device(args, kwargs) -> torch.device:
    for v in list(args) + list((kwargs or {}).values()):
        if isinstance(v, torch.Tensor):
            return v.device
    return torch.device('cpu')


class _MemProbe:
    def __init__(self, device: torch.device):
        self.device = device
        self._baseline = 0
        self._peak = 0

    def _current(self) -> Optional[int]:
        if self.device.type == 'mps':
            return torch.mps.current_allocated_memory()
        if self.device.type == 'cpu':
            # ru_maxrss is bytes on macOS, kilobytes on Linux
            factor = 1 if sys.platform == 'darwin' else 1024
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * factor
        return None

    def start(self):
        if self.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(self.device)
            self._baseline = torch.cuda.max_memory_allocated(self.device)
        else:
            self._baseline = self._current() or 0
            self._peak = self._baseline

    def sample(self):
        # cuda tracks its own peak; others need polling between iterations
        if self.device.type != 'cuda':
            cur = self._current()
            if cur is not None:
                self._peak = max(self._peak, cur)

    def stop(self) -> tuple[Optional[int], bool]:
        if self.device.type == 'cuda':
            peak = torch.cuda.max_memory_allocated(self.device)
            return max(0, peak - self._baseline), True
        cur = self._current()
        if cur is None:
            return None, False
        return max(0, max(self._peak, cur) - self._baseline), False


def _cuda_timed_loop(run, device: torch.device, iters: int) -> list[float]:
    '''CUDA timing: prefer triton's do_bench (event timing + L2 flush +
    adaptive repeat count — ``iters`` is advisory there); otherwise a manual
    CUDA-event loop that flushes L2 between iterations the same way.'''
    try:
        from triton.testing import do_bench
        return list(do_bench(run, return_mode='all'))
    except ImportError:
        pass

    cache = torch.empty(256 * 1024 * 1024, dtype=torch.int8, device=device)
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        cache.zero_()
        starts[i].record()
        run()
        ends[i].record()
    torch.cuda.synchronize(device)
    return [s.elapsed_time(e) for s, e in zip(starts, ends)]


def _cuda_peak_mem(run, device: torch.device, probes: int = 3) -> int:
    '''Exact peak-memory delta, in a pass separate from timing so the
    L2-flush buffer never pollutes the number.'''
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    base = torch.cuda.memory_allocated(device)
    for _ in range(probes):
        run()
    torch.cuda.synchronize(device)
    return max(0, torch.cuda.max_memory_allocated(device) - base)


def bench(
    fn: Callable,
    args: tuple = (),
    kwargs: Optional[dict] = None,
    *,
    name: Optional[str] = None,
    device: Optional[torch.device] = None,
    warmup: int = 10,
    iters: int = 50,
) -> BenchResult:
    '''Measure fn(*args, **kwargs): warmup runs, then timed runs with a
    device synchronize inside every measurement window. On CUDA, timing is
    event-based with L2-cache flushing (via triton.testing.do_bench when
    triton is installed) and peak memory is exact.'''
    kwargs = kwargs or {}
    device = device or infer_device(args, kwargs)
    label = name or getattr(fn, '__name__', repr(fn))

    def run():
        return fn(*args, **kwargs)

    def run_once():
        t0 = time.perf_counter()
        out = run()
        synchronize(device)
        return (time.perf_counter() - t0) * 1e3, out

    try:
        gc.collect()
        _empty_cache(device)
        synchronize(device)

        first_call_ms, _ = run_once()
        for _ in range(max(0, warmup - 1)):
            run_once()

        if device.type == 'cuda':
            times = _cuda_timed_loop(run, device, iters)
            peak_mem, mem_exact = _cuda_peak_mem(run, device), True
        else:
            probe = _MemProbe(device)
            probe.start()
            times = []
            for _ in range(iters):
                dt, out = run_once()
                times.append(dt)
                # sample while the iteration's output is still alive, otherwise
                # backends without a peak counter always read the baseline
                probe.sample()
                del out
            peak_mem, mem_exact = probe.stop()
    except Exception as e:  # noqa: BLE001 - a failed variant must not kill the run
        return BenchResult(label, str(device), 0, 0, 0, 0, 0, 0,
                           None, False, error=f'{type(e).__name__}: {e}')

    mean = sum(times) / len(times)
    std = (sum((t - mean) ** 2 for t in times) / len(times)) ** 0.5
    return BenchResult(
        name=label, device=str(device), iters=len(times),
        mean_ms=mean, std_ms=std, min_ms=min(times), max_ms=max(times),
        first_call_ms=first_call_ms, peak_mem=peak_mem, mem_exact=mem_exact,
    )
