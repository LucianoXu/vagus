#!/bin/bash -l
# SAX1-340M on FineWeb-Edu — Raven, 1 node x 4xA100.
# Submit from the repo root; an optional argument picks the train recipe:
#   main run:  sbatch recipe/slurm/raven_sax1_340M.sbatch
#   lr pilots: sbatch -J sax1pilot17 -t 03:00:00 \
#                recipe/slurm/raven_sax1_340M.sbatch \
#                recipe/train/sax1_340M_pilot_lr17.yaml
# Resubmitting the same script+config resumes from the latest recent
# checkpoint.
#SBATCH --job-name=sax1-340M
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:a100:4
#SBATCH --mem=500000
#SBATCH --time=24:00:00
#SBATCH --output=runs/slurm-%x-%j.out
#SBATCH --error=runs/slurm-%x-%j.err
# 900s before walltime, SIGUSR1 to this batch shell; we forward SIGTERM to
# the training process group -> every rank sets the stop flag -> rank 0
# writes a recent checkpoint and all exit cleanly.
#SBATCH --signal=B:SIGUSR1@900

set -uo pipefail
cd "$HOME/work/vagus"

CONFIG="${1:-recipe/train/sax1_340M_fwe15B.yaml}"

module purge
module load python-waterboa/2025.06
source .venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=18

echo "Job $SLURM_JOB_ID on $(hostname) | $(nvidia-smi -L | head -1) | config $CONFIG"

setsid torchrun --nproc_per_node 4 -m infra.train.main "$CONFIG" &
PID=$!
trap "echo 'walltime soon: forwarding SIGTERM to training'; kill -TERM -- -$PID" USR1
wait $PID
status=$?
echo "training exited with status $status"
exit $status
