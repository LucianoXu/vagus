'''
meter — a small, backend-adaptive performance measurement toolkit.

    from infra.meter import bench, compare

``bench`` measures one callable (speed + peak memory);
``compare`` races several implementations of the same computation over
identical inputs and reports speed / memory / accuracy side by side.
Works on cpu / cuda / mps; unavailable backends should simply be left
out of the variants dict by the caller (see examples/bench_rope.py).
'''

from .core import BenchResult, bench, synchronize
from .compare import compare, render_table
from .profiling import profile_variants

__all__ = ['BenchResult', 'bench', 'compare', 'profile_variants',
           'render_table', 'synchronize']
# .dist (dist_bench, render_dist_table) is imported explicitly by callers so
# that plain single-process use never touches torch.distributed

