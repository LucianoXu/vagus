#!/bin/bash -l
# LAX smoke on one A100 (gpudev, minutes): fla-vs-torch kernel tests,
# then the mixer benchmark at the LAX1-340M layer shape. Run before the
# first pilot on any new fla / torch / driver combination.
#   submit from the repo root: sbatch recipe/slurm/raven_smoke_lax.sbatch
#SBATCH --job-name=lax-smoke
#SBATCH --partition=gpudev
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=120000
#SBATCH --time=00:15:00
#SBATCH --output=runs/slurm-%x-%j.out
#SBATCH --error=runs/slurm-%x-%j.err

set -uo pipefail
cd "$HOME/work/vagus"

module purge
module load python-waterboa/2025.06
source .venv/bin/activate
export OMP_NUM_THREADS=18

echo "Job $SLURM_JOB_ID on $(hostname) | $(nvidia-smi -L | head -1)"
python -c "import torch, fla; print('torch', torch.__version__, '| fla', fla.__version__)" || exit 1

python -m pytest tests/test_linear_attention.py tests/test_gdn_lm.py -q -x || exit 1
python -m infra.meter.examples.bench_linear_attention --device cuda --dtype bf16 --which train --batch 16
python -m infra.meter.examples.bench_linear_attention --device cuda --dtype bf16 --which prefill --batch 16
python -m infra.meter.examples.bench_linear_attention --device cuda --dtype bf16 --which decode --batch 16 --seq 2048
