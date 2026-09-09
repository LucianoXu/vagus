#!/bin/bash -l
# One gpudev slot, one recipe, one config key swept over several values:
# short training runs whose only difference is that key, so their
# tokens_per_s / mfu / max_mem_gb are comparable. The measurement that
# settles a kernel-path question — a microbenchmark and a compiled
# training step do not agree, and this is the one that counts.
#
#   sbatch recipe/slurm/raven_gpudev_sweep.sbatch \
#     recipe/train/hax1_340M_smoke.yaml model_args.la_fused false conv gate
#
# Arguments: the recipe, the override key, then its values. Fixed
# overrides for one node (world 4) are baked in below; the run name is
# kb-sw-<value>. Budget ~2.5 min per value at 20 steps of HAX1-340M, so
# five fit in the 15-minute cap. Read the results with
#   python -m infra.meter.examples.bench_kda_kernel --which sweep runs/
# or straight off the TensorBoard scalars.
#
# Not an experiment: the run dirs land under runs/ and should be deleted.
#SBATCH --job-name=gpudev-sweep
#SBATCH --partition=gpudev
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:a100:4
#SBATCH --mem=400000
#SBATCH --time=00:15:00
#SBATCH --output=runs/slurm-%x-%j.out
#SBATCH --error=runs/slurm-%x-%j.err

set -uo pipefail
cd "$HOME/work/vagus"

CONFIG="$1"; KEY="$2"; shift 2

module purge
module load python-waterboa/2025.06
source .venv/bin/activate
export OMP_NUM_THREADS=18
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Job $SLURM_JOB_ID on $(hostname) | $(nvidia-smi -L | head -1)"
echo "sweep $CONFIG over $KEY: $*"

# The run-to-run spread of this measurement is a few percent while the
# within-run tokens_per_s is flat to 0.5%, so a value has to be repeated
# — list it more than once, interleaved with its control — before a
# difference of that size means anything. Each repeat gets its own run
# name (i is the position in the argument list) so none is overwritten.
i=0
for VALUE in "$@"; do
    i=$((i + 1))
    NAME=$(echo "$VALUE" | tr ',' '-')-$i
    echo "=== $KEY=$VALUE -> kb-sw-$NAME"
    rm -rf "runs/kb-sw-$NAME-"*
    python -u -m torch.distributed.run --nproc_per_node 4 --shutdown_timeout 120 \
        -m infra.train.main "$CONFIG" \
        world_size=4 grad_accum_steps=8 shard_size=4 train_tokens=10485760 \
        log_interval=2 permanent_ckpt_interval=null recent_ckpt_minutes=100000 \
        "run_name=kb-sw-$NAME" "$KEY=$VALUE"
done
