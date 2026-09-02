#!/bin/bash -l
# Evict-only budget-PPL evaluation (infra/eval/evict_ppl.py) — Raven,
# one A100 share of a node. All script arguments pass through:
#   sbatch -J eval-evict recipe/slurm/raven_evict_ppl.sbatch \
#     --ckpt runs/sax2-340M-fwe15B-79629bc6/model-final.pt \
#     --data_dir data/tokenized/fineweb-edu-100BT-mistral32k \
#     --out runs/eval/evict_frozen.json --budgets 32 64 128 256 512
#SBATCH --job-name=evict-ppl
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=125000
#SBATCH --time=06:00:00
#SBATCH --output=runs/slurm-%x-%j.out
#SBATCH --error=runs/slurm-%x-%j.err

set -uo pipefail
cd "$HOME/work/vagus"

module purge
module load python-waterboa/2025.06
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Job $SLURM_JOB_ID on $(hostname) | $(nvidia-smi -L | head -1) | args: $*"
python -m infra.eval.evict_ppl "$@"
