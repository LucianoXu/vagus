#!/bin/bash -l
# Tier-1 continued-training arms (unified / plain) — Raven, 1 node x 4xA100.
# Submit from the repo root with the train recipe as the argument:
#   sbatch -J uct-s43 recipe/slurm/raven_managed.sbatch recipe/train/unified_ct2B_m512_s43.yaml
# Resubmitting the same script+config resumes from the latest recent ckpt.
#SBATCH --job-name=managed-ct
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:a100:4
#SBATCH --mem=500000
#SBATCH --time=24:00:00
#SBATCH --output=runs/slurm-%x-%j.out
#SBATCH --error=runs/slurm-%x-%j.err
#SBATCH --signal=B:SIGUSR1@900

set -uo pipefail
cd "$HOME/work/vagus"

CONFIG="${1:?train recipe yaml required}"

module purge
module load python-waterboa/2025.06
source .venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=18

echo "Job $SLURM_JOB_ID on $(hostname) | $(nvidia-smi -L | head -1) | config $CONFIG"

setsid torchrun --nproc_per_node 4 -m infra.train.managed "$CONFIG" &
PID=$!
trap "echo 'walltime soon: forwarding SIGTERM to training'; kill -TERM -- -$PID" USR1
wait $PID
status=$?
echo "training exited with status $status"
exit $status
