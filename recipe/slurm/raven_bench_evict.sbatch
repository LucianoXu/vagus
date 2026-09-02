#!/bin/bash -l
# Evict-only throughput/memory bench + test suite — Raven, one A100
# share. Runs the GPU test suite first (gates 0/1 on the real device),
# then the bench grid given as arguments (each line one bench call).
#   sbatch -J bench-evict recipe/slurm/raven_bench_evict.sbatch
#SBATCH --job-name=bench-evict
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=125000
#SBATCH --time=01:30:00
#SBATCH --output=runs/slurm-%x-%j.out
#SBATCH --error=runs/slurm-%x-%j.err

set -uo pipefail
cd "$HOME/work/vagus"

module purge
module load python-waterboa/2025.06
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Job $SLURM_JOB_ID on $(hostname) | $(nvidia-smi -L | head -1)"
echo "=== tests ==="
python -m pytest tests/test_evict.py -q 2>&1 | tail -15

echo "=== bench grid ==="
B=infra.eval.bench_evict
python -m $B --batch 8 --budget 128 --block 256 --compile --plain
python -m $B --batch 8 --budget 128 --block 256 --compile --ckpt_act
python -m $B --batch 8 --budget 128 --block 256
python -m $B --batch 8 --budget 128 --block 512 --compile
python -m $B --batch 8 --budget 512 --block 256 --compile
python -m $B --batch 16 --budget 128 --block 256 --compile
