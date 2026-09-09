#!/bin/bash -l
# HAX2-1.3B (parallel-hybrid KDA + gated NoPE softmax) on FineWeb-Edu —
# Raven, 3 nodes x 4xA100 under HSDP (shard inside the node, replicate
# across). Submit from the repo root; an optional argument picks the
# train recipe, further key=value arguments override top-level
# TrainConfig fields:
#   main run:  sbatch recipe/slurm/raven_hax2_1.3B.sbatch
#   smoke:     sbatch -J hax2smoke4 -t 00:40:00 recipe/slurm/raven_hax2_1.3B.sbatch \
#                recipe/train/hax2_1.3B_smoke.yaml
# Resubmitting the same script+config resumes from the latest recent
# checkpoint (the run dir is <run_name>-<commit8>; pass run_dir=... to
# continue a run across code commits). Walltime chaining: queue the
# continuations up front, each after the previous one, all with the
# same run_dir=; a link whose run is already complete (model-final.pt
# in the run dir) exits at once instead of holding the nodes:
#   J1=$(sbatch --parsable recipe/slurm/raven_hax2_1.3B.sbatch)
#   J2=$(sbatch --parsable --dependency=afterany:$J1 recipe/slurm/raven_hax2_1.3B.sbatch \
#          recipe/train/hax2_1.3B_fwe60B.yaml run_dir=runs/hax2-1.3B-fwe60B-<commit8>)
#   ... (one link per expected walltime, ~24 h each)
#SBATCH --job-name=hax2-1.3B
#SBATCH --partition=gpu
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:a100:4
#SBATCH --mem=500000
#SBATCH --time=24:00:00
#SBATCH --output=runs/slurm-%x-%j.out
#SBATCH --error=runs/slurm-%x-%j.err
# 900s before walltime, SIGUSR1 to this batch shell; we forward SIGTERM
# through srun to every node's torchrun -> its workers -> every rank sets
# the stop flag -> rank 0 writes a recent checkpoint and all exit cleanly.
#SBATCH --signal=B:SIGUSR1@900

set -uo pipefail
cd "$HOME/work/vagus"

CONFIG="${1:-recipe/train/hax2_1.3B_fwe60B.yaml}"
shift $(( $# > 0 ? 1 : 0 ))

module purge
module load python-waterboa/2025.06
source .venv/bin/activate
python -c "import fla" 2>/dev/null || echo "WARNING: fla not installed; torch reference path (slow)"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=18
# rendezvous on the first allocated node; a job-unique port
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=$(( 20000 + SLURM_JOB_ID % 20000 ))
# NCCL picks the IB transport itself; INIT/NET logging is the receipt
# (look for "NET/IB" in the .err), cheap enough to keep on
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET

echo "Job $SLURM_JOB_ID on $SLURM_JOB_NODELIST | master $MASTER_ADDR:$MASTER_PORT | config $CONFIG $*"

# chain guard: a continuation link whose run already finished releases
# the allocation immediately (run_dir=... is how links address the run)
for arg in "$@"; do
    case "$arg" in run_dir=*)
        if [ -f "${arg#run_dir=}/model-final.pt" ]; then
            echo "run ${arg#run_dir=} is complete (model-final.pt present); nothing to do"
            exit 0
        fi;;
    esac
done

srun --kill-on-bad-exit=1 recipe/slurm/torchrun_node.sh "$CONFIG" "$@" &
PID=$!
trap "echo 'walltime soon: forwarding SIGTERM to training'; kill -TERM $PID" USR1
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
