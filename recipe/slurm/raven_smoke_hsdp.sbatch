#!/bin/bash -l
# HSDP / HAX1 smoke on one gpudev node (4xA100, minutes). Three stages:
#   1. the CPU-side unit tests (parallel mixer, hybrid model, kernels);
#   2. tests/dist/parity_check under NCCL on 4 GPUs: sharded Muon ==
#      dense Muon exactly, DDP == FSDP == HSDP(2x2) loss/param traces,
#      and a train() run with checkpoint + resume;
#   3. HAX1-340M for 30 steps under FSDP2 on the node (shard_size 4,
#      no replication): compile, memory at micro 8, step time, the
#      340M-size checkpoint gather + slim export.
# Run before the first 2-node job on any new torch / fla / driver combo:
#   submit from the repo root: sbatch recipe/slurm/raven_smoke_hsdp.sbatch
#SBATCH --job-name=hsdp-smoke
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

module purge
module load python-waterboa/2025.06
source .venv/bin/activate
export OMP_NUM_THREADS=18
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Job $SLURM_JOB_ID on $(hostname) | $(nvidia-smi -L | head -1)"
python -c "import torch, fla; print('torch', torch.__version__, '| fla', fla.__version__)" || exit 1

echo "=== 1. unit tests"
python -m pytest tests -q -x || exit 1

echo "=== 2. parity under NCCL (4 GPUs)"
python -m torch.distributed.run --nproc_per_node 4 -m tests.dist.parity_check || exit 1

echo "=== 3. HAX1-340M, 30 steps, FSDP2 on one node"
SMOKE_DIR="runs/hax1-340M-smoke1n-$SLURM_JOB_ID"
python -m torch.distributed.run --nproc_per_node 4 --shutdown_timeout 600 -m infra.train.main \
    recipe/train/hax1_340M_smoke.yaml \
    run_name=hax1-340M-smoke1n run_dir="$SMOKE_DIR" \
    world_size=4 grad_accum_steps=8 train_tokens=15728640 || exit 1
ls -la "$SMOKE_DIR"
echo "smoke OK; remove $SMOKE_DIR when done reading it"
