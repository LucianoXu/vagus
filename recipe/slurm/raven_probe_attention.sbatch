#!/bin/bash -l
# Attention health probe on saved checkpoints — one A100, minutes.
# Submit from the repo root with the checkpoints as arguments:
#   sbatch [--export=ALL,SEQ=2048] recipe/slurm/raven_probe_attention.sbatch \
#     runs/<run>/ckpt-00005000.pt runs/<run>/ckpt-00010000.pt
# Read-only: loads checkpoints, touches no run directory.
#SBATCH --job-name=attnprobe
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

echo "Job $SLURM_JOB_ID on $(hostname) | $(nvidia-smi -L | head -1) | checkpoints $*"
python -m infra.meter.examples.probe_attention_health "$@" \
    --seq "${SEQ:-4096}" --skip 64 --json runs/attn-probe-$SLURM_JOB_ID.json
