#!/bin/bash -l
# LAX2-340M (KDA with an Isotropic Kalman write gain) on FineWeb-Edu —
# Raven, 1 node x 4xA100, the same layout LAX1 ran on.
# Submit from the repo root; an optional argument picks the train recipe,
# further key=value arguments override top-level TrainConfig fields:
#   smoke FIRST: sbatch -J lax2smoke -t 00:30:00 \
#                  recipe/slurm/raven_lax2_340M.sbatch \
#                  recipe/train/lax2_340M_smoke.yaml
#     Read the smoke's gate probe before spending 12 h: if beta is still
#     pinned near 1 the Kalman gain is inert and the run is pointless.
#   main run:    sbatch recipe/slurm/raven_lax2_340M.sbatch
# No pilot ladder: the mixer's width and depth are LAX1's, so the LAX1
# ladder's lr 1.75e-3 carries over.
# Resubmitting the same script+config resumes from the latest recent
# checkpoint (the run dir is <run_name>-<commit8>); a resubmit from a
# newer checkout continues the original directory with run_dir=...
#SBATCH --job-name=lax2-340M
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

CONFIG="${1:-recipe/train/lax2_340M_fwe15B.yaml}"
shift $(( $# > 0 ? 1 : 0 ))

module purge
module load python-waterboa/2025.06
source .venv/bin/activate
# the training path needs the fla Triton kernels (one-time, login node):
#   pip install fla-core   # import name `fla` (kernels only)
# without it la_impl=auto silently runs the torch chunk reference
python -c "import fla" 2>/dev/null || echo "WARNING: fla not installed; torch reference path (slow)"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=18

echo "Job $SLURM_JOB_ID on $(hostname) | $(nvidia-smi -L | head -1) | config $CONFIG $*"

# --shutdown_timeout: torchrun also receives the group SIGTERM and would
# SIGKILL the workers 30 s later by default, before rank 0 has finished
# writing the recent checkpoint
setsid torchrun --nproc_per_node 4 --shutdown_timeout 900 -m infra.train.main "$CONFIG" "$@" &
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
