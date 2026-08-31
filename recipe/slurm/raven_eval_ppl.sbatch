#!/bin/bash -l
# Budget-PPL evaluation (infra/eval/budget_ppl.py) — Raven, single GPU
# of one node. All script arguments pass through:
#   sbatch -J eval-frozen recipe/slurm/raven_eval_ppl.sbatch \
#     --ckpt runs/sax2-340M-fwe15B-79629bc6/model-final.pt \
#     --data_dir data/tokenized/fineweb-edu-100BT-mistral32k \
#     --out runs/eval/frozen_sax2.json
#SBATCH --job-name=budget-ppl
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --gres=gpu:a100:4
#SBATCH --mem=120000
#SBATCH --time=06:00:00
#SBATCH --output=runs/slurm-%x-%j.out
#SBATCH --error=runs/slurm-%x-%j.err

set -uo pipefail
cd "$HOME/work/vagus"

module purge
module load python-waterboa/2025.06
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Job $SLURM_JOB_ID on $(hostname) | args: $*"
python -m infra.eval.budget_ppl "$@"
