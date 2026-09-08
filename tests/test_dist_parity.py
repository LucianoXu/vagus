# Opt-in wrapper around tests/dist/parity_check.py (4 processes, ~2 min
# on a laptop CPU): DDP == FSDP2 == HSDP parity, exact sharded Muon,
# train() with checkpoint + resume under HSDP. Enable with
# VAGUS_DIST_TESTS=1; the raven smoke job runs the script directly
# under NCCL.

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(not os.environ.get('VAGUS_DIST_TESTS'),
                    reason='opt-in (VAGUS_DIST_TESTS=1): 4-process torchrun parity check, ~2 min')
def test_parallel_regimes_agree(tmp_path):
    cmd = [sys.executable, '-m', 'torch.distributed.run', '--nproc_per_node', '4',
           '-m', 'tests.dist.parity_check', '--out', str(tmp_path)]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=1800)
    tail = '\n'.join((proc.stdout + proc.stderr).splitlines()[-40:])
    assert proc.returncode == 0, tail
    assert '[parity] ALL OK' in proc.stdout, tail
