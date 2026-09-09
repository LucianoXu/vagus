#!/bin/bash -l
# Any eval recipe on one A100 of the shared gpu1 partition. Arguments:
# the recipe, then key=value EvalConfig overrides. Runs in the checkout
# it was submitted from (SLURM_SUBMIT_DIR), so a side checkout such as
# ~/work/vagus-mem-consolidate evaluates its own code and runs/.
#   sbatch -J probe recipe/slurm/raven_eval.sbatch recipe/eval/consolidate_lax1_probe.yaml
#   sbatch -J replay recipe/slurm/raven_eval.sbatch recipe/eval/consolidate_lax1_replay.yaml
#SBATCH --job-name=eval
#SBATCH --partition=gpu1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=125000
#SBATCH --time=06:00:00
#SBATCH --output=runs/slurm-%x-%j.out
#SBATCH --error=runs/slurm-%x-%j.err

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
CONFIG="$1"; shift

module purge
module load python-waterboa/2025.06
source .venv/bin/activate
export OMP_NUM_THREADS=18
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Job $SLURM_JOB_ID on $(hostname) | $(nvidia-smi -L | head -1) | $(pwd) $(git rev-parse --short HEAD) | $CONFIG $*"
python -u -m infra.eval.main "$CONFIG" "$@"
