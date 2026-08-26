'''
Run several implementations of the same computation over identical inputs,
check them against a reference variant, and report speed / memory / accuracy
side by side.
'''

from typing import Callable, Optional

import torch

from ..utils import infer_device
from .core import BenchResult, bench


def _fmt_mem(nbytes: Optional[int], exact: bool) -> str:
    if nbytes is None:
        return 'n/a'
    size = float(nbytes)
    for unit in ('B', 'KiB', 'MiB', 'GiB'):
        if size < 1024 or unit == 'GiB':
            break
        size /= 1024
    return f'{size:.1f} {unit}' + ('' if exact else ' ~')


def _max_diff(a, b) -> Optional[float]:
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        if a.shape != b.shape:
            return float('inf')
        return (a.float() - b.float()).abs().max().item()
    return None


def compare(
    variants: dict[str, Callable],
    args: tuple = (),
    kwargs: Optional[dict] = None,
    *,
    ref: Optional[str] = None,
    warmup: int = 10,
    iters: int = 50,
    check: bool = True,
    verbose: bool = True,
) -> dict[str, BenchResult]:
    '''Benchmark every variant on the same args/kwargs.

    ``ref`` names the reference variant (default: the first one); other
    variants report speedup and max output deviation relative to it.
    Variants must not mutate their inputs. A variant that raises is
    reported as failed instead of aborting the run.
    '''
    kwargs = kwargs or {}
    ref = ref or next(iter(variants))
    device = infer_device(args, kwargs)

    results: dict[str, BenchResult] = {}
    outputs: dict[str, object] = {}

    for name, fn in variants.items():
        # bench first so first_call_ms captures lazy costs (torch.compile);
        # the correctness pass afterwards reuses the warmed-up fn
        results[name] = bench(fn, args, kwargs, name=name,
                              device=device, warmup=warmup, iters=iters)
        if check and not results[name].failed:
            try:
                with torch.no_grad():
                    outputs[name] = fn(*args, **kwargs)
            except Exception:  # noqa: BLE001 - already reported by bench()
                outputs[name] = None

    if verbose:
        print(render_table(results, outputs if check else None, ref=ref))
    return results


def render_table(
    results: dict[str, BenchResult],
    outputs: Optional[dict] = None,
    *,
    ref: str,
) -> str:
    ref_res = results.get(ref)
    ref_out = (outputs or {}).get(ref)

    header = ['variant', 'mean', 'std', 'min', 'speedup',
              'first call', 'peak mem', 'max|Δ| vs ref']
    rows = [header]
    for name, r in results.items():
        if r.failed:
            rows.append([name, f'FAILED: {r.error}', '', '', '', '', '', ''])
            continue
        speedup = ('1.00x (ref)' if name == ref else
                   f'{ref_res.mean_ms / r.mean_ms:.2f}x'
                   if ref_res and not ref_res.failed else 'n/a')
        diff = _max_diff(ref_out, (outputs or {}).get(name)) if outputs else None
        rows.append([
            name,
            f'{r.mean_ms:.3f} ms',
            f'{r.std_ms:.3f}',
            f'{r.min_ms:.3f}',
            speedup,
            f'{r.first_call_ms:.1f} ms',
            _fmt_mem(r.peak_mem, r.mem_exact),
            ('ref' if name == ref else
             f'{diff:.2e}' if diff is not None else '-'),
        ])

    widths = [max(len(row[i]) for row in rows) for i in range(len(header))]
    lines = []
    for i, row in enumerate(rows):
        lines.append('  '.join(c.ljust(w) for c, w in zip(row, widths)).rstrip())
        if i == 0:
            lines.append('  '.join('-' * w for w in widths))

    dev = next(iter(results.values())).device if results else '?'
    note = ('  (~ = approximate: this backend has no true peak-memory counter)'
            if any(not r.mem_exact and r.peak_mem is not None
                   for r in results.values()) else '')
    return f'[meter] device={dev}{note}\n' + '\n'.join(lines)
