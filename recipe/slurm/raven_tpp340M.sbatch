#!/bin/bash -l
# Transformer++ 340M baseline on FineWeb-Edu 15B — Raven, 1 node x 4xA100.
#   submit from the repo root: sbatch recipe/slurm/raven_tpp340M.sbatch
# Resubmitting the same script resumes from the latest recent checkpoint.
#SBATCH --job-name=tpp340M
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

module purge
module load python-waterboa/2025.06
source .venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=18

echo "Job $SLURM_JOB_ID on $(hostname) | $(nvidia-smi -L | head -1)"

setsid torchrun --nproc_per_node 4 -m infra.train.main \
    recipe/train/tpp_340M_fwe15B.yaml &
PID=$!
trap "echo 'walltime soon: forwarding SIGTERM to training'; kill -TERM -- -$PID" USR1
wait $PID
status=$?
# A trapped signal interrupts `wait` with 128+sig while training is still
# running. Without waiting again the batch script exits here and SLURM
# tears the step down before rank 0 has written the recent checkpoint
# (job 30073697: ended at 23:44:44 with exit 138, recent.pt.tmp truncated,
# resume fell back to recent-prev.pt). Keep waiting for the real exit.
while [ "$status" -gt 128 ] && kill -0 "$PID" 2>/dev/null; do
    wait $PID
    status=$?
done
echo "training exited with status $status"
exit $status
