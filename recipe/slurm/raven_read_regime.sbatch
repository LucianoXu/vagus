#!/bin/bash -l
# Readout-regime probe (infra/meter/examples/read_regime.py) on saved
# checkpoints — one A100, minutes. Where do a trained mixer's reads sit
# on the head RMSNorm's curve, and what read_floor would move the knee
# to the median read? Calibrates LAX5 from LAX1; re-run on LAX5 to see
# whether the floor is still in use.
#   sbatch recipe/slurm/raven_read_regime.sbatch \
#     LAX1=runs/lax1-340M-fwe15B-c9af2f44/model-final.pt [LABEL=PATH ...]
# Read-only: loads checkpoints, writes runs/read-regime-<job>.json.
#SBATCH --job-name=readregime
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

echo "Job $SLURM_JOB_ID on $(hostname) | $(nvidia-smi -L | head -1) | $(git rev-parse --short HEAD) | $*"
CKPTS=()
for c in "$@"; do CKPTS+=(--ckpt "$c"); done
python -u -m infra.meter.examples.read_regime "${CKPTS[@]}" \
    --data data/tokenized/fineweb-edu-100BT-mistral32k \
    --out runs/read-regime-$SLURM_JOB_ID.json
