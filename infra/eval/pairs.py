# Paired comparison of two subjects on the same items: mean difference
# with a bootstrap interval. Kept out of the tasks — a task reports
# per-item numbers, the runner decides what to compare with what.

import numpy as np


def paired(ref: list[float], other: list[float], n_boot: int = 2000, seed: int = 0) -> dict:
    '''other - ref per item; mean, percentile-bootstrap 95% CI of the
    mean, and the fraction of items where other > ref.'''
    a, b = np.asarray(ref, dtype=np.float64), np.asarray(other, dtype=np.float64)
    assert a.shape == b.shape and a.ndim == 1 and len(a) > 0
    d = b - a
    n = len(d)
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, n, (n_boot, n))].mean(axis=1)
    return {
        'n': n,
        'mean_diff': float(d.mean()),
        'ci95': [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))],
        'frac_positive': float((d > 0).mean()),
    }
