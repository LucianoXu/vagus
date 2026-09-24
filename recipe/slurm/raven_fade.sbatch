#!/bin/bash -l
# The fade diagnostic (infra/consolidate/fade.py) — one A100 of gpu1.
# Does scaling the carried memory by c read as a weaker memory, at the
# prediction (all layers / one layer) and at each layer's readout?
# Runs in the checkout it was submitted from.
#   sbatch recipe/slurm/raven_fade.sbatch runs/gen-eval/lax1-model-final.pt LAX1
#SBATCH --job-name=fade
#SBATCH --partition=gpu1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=125000
#SBATCH --time=02:00:00
#SBATCH --output=runs/slurm-%x-%j.out
#SBATCH --error=runs/slurm-%x-%j.err

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
CKPT="$1"; LABEL="${2:-subject}"; shift 2 || shift $#

module purge
module load python-waterboa/2025.06
source .venv/bin/activate
export OMP_NUM_THREADS=18

echo "Job $SLURM_JOB_ID on $(hostname) | $(nvidia-smi -L | head -1) | $(pwd) $(git rev-parse --short HEAD) | $CKPT $LABEL $*"
python -u -m infra.consolidate.fade --ckpt "$CKPT" --label "$LABEL" \
    --data data/tokenized/fineweb-edu-100BT-mistral32k \
    --out "runs/eval/fade-$LABEL-$SLURM_JOB_ID.json" "$@"
