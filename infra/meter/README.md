# meter

Backend-adaptive performance measurement for torch code: wall-clock time and
peak memory on `cpu` / `cuda` / `mps`, plus side-by-side comparison of several
implementations of the same computation (with a correctness check against a
reference variant).

## Usage

```python
from infra.meter import bench, compare

# measure one callable
r = bench(fn, (x,), warmup=10, iters=50)
print(r.mean_ms, r.peak_mem)

# race implementations over identical inputs; prints a table
compare({
    'eager': module,
    'compiled': torch.compile(module),
    'triton': my_triton_fn,          # include only when available
}, (x,))
```

Rules for variants: they must not mutate their inputs, and the caller decides
which variants exist on the current machine (see `examples/bench_rope.py` for
the availability-gating pattern). A variant that raises is reported as
`FAILED` instead of aborting the run.

## Example: RoPE (eager vs compile vs triton)

```sh
python infra/meter/examples/bench_rope.py --device mps --seq 4096
python infra/meter/examples/bench_rope.py --device cuda --dtype bf16   # on a GPU box
```

On this Mac (MPS, no triton) the triton column is skipped automatically; sync
the vault to a CUDA machine to get all three.

## Profiling (`--profile`)

`compare` answers "how fast"; `profile_variants` answers "where does the time
go" over the same variants/inputs harness: per-op tables printed to stdout and
a Chrome trace per variant (open in chrome://tracing or ui.perfetto.dev),
written to `runs/profiles/` (git-ignored).

```sh
python infra/meter/examples/bench_rope.py --device mps --profile
```

## Distributed (multi-process / multi-GPU)

`dist.dist_bench` runs the same bench on every rank with a barrier aligning
the start, gathers results to rank 0, and reports per-rank stats plus a
slowest/fastest imbalance ratio. Works with gloo on CPU (testable on this
Mac) and nccl on a GPU box.

```sh
# gloo/cpu on macOS — pin loopback + IPv4 or the rendezvous hangs on IPv6 reverse-DNS
GLOO_SOCKET_IFNAME=lo0 torchrun --nproc_per_node 2 \
    --master_addr 127.0.0.1 --master_port 29517 infra/meter/examples/bench_dist.py

torchrun --nproc_per_node 8 infra/meter/examples/bench_dist.py --nccl   # GPU box
```

## Measurement notes

- **Timing**: on CUDA, event-based with an L2-cache flush between iterations —
  via `triton.testing.do_bench` when triton is installed (its adaptive repeat
  count makes `iters` advisory), else a manual CUDA-event loop with the same
  flushing. On mps/cpu, `perf_counter` around `fn() + synchronize`.
  `first_call_ms` is measured before warmup and captures lazy costs — for
  `torch.compile` that is the compile time.
- **Memory**: CUDA reports a true peak (`max_memory_allocated`), measured in a
  pass separate from timing so the L2-flush buffer never pollutes it. MPS only
  exposes `current_allocated_memory`, so the probe samples between iterations
  while the output is still alive — transient in-kernel peaks are invisible.
  CPU falls back to `ru_maxrss`, which is process-wide and monotonic. The
  table marks non-exact numbers with `~`.

## Layout

- `core.py` — `bench()` + all device quirks (sync, timing, memory probes)
- `compare.py` — `compare()` variant racing + table rendering
- `profiling.py` — `profile_variants()` torch.profiler integration
- `dist.py` — `dist_bench()` barrier-aligned per-rank measurement
- `examples/bench_rope.py` — the RoPE comparison (`--profile` supported)
- `examples/bench_dist.py` — distributed toy train-step example
- `examples/rope_triton.py` — triton RoPE kernel (import-guarded, CUDA only)
