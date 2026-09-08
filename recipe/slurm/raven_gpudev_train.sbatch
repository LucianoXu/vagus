#!/bin/bash -l
# Any train recipe for a few minutes on one gpudev node (4xA100, 15 min
# cap): throughput / memory / compile checks of a configuration before
# it goes to the gpu partition. Arguments: the recipe, then key=value
# TrainConfig overrides (world_size must be 4). Example — the HAX1 smoke
# recipe on one node under DDP vs FSDP without resharding:
#   sbatch -J hax1ddp recipe/slurm/raven_gpudev_train.sbatch recipe/train/hax1_340M_smoke.yaml \
#     run_name=hax1-smoke-ddp world_size=4 grad_accum_steps=8 shard_size=1 train_tokens=15728640
#   sbatch -J hax1nrs recipe/slurm/raven_gpudev_train.sbatch recipe/train/hax1_340M_smoke.yaml \
#     run_name=hax1-smoke-nrs world_size=4 grad_accum_steps=8 fsdp_reshard_after_forward=false train_tokens=15728640
# Not an experiment: the run dir lands under runs/ and should be deleted.
#SBATCH --job-name=gpudev-train
#SBATCH --partition=gpudev
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:a100:4
#SBATCH --mem=500000
#SBATCH --time=00:15:00
#SBATCH --output=runs/slurm-%x-%j.out
#SBATCH --error=runs/slurm-%x-%j.err

set -uo pipefail
cd "$HOME/work/vagus"
CONFIG="$1"; shift

module purge
module load python-waterboa/2025.06
source .venv/bin/activate
export OMP_NUM_THREADS=18
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Job $SLURM_JOB_ID on $(hostname) | $(nvidia-smi -L | head -1) | $CONFIG $*"
python -m torch.distributed.run --nproc_per_node 4 --shutdown_timeout 600 -m infra.train.main "$CONFIG" "$@"
