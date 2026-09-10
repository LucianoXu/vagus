#!/bin/bash
# Phase-B: multi-hop replay curricula on LAX1, gap 256, 32 items, at the
# operating point phase A picked (env: LR, MIX, SCHED, WARM, RK; defaults
# below are placeholders — set them from the phase-A table). Cells: how
# the memory is removed (scale / layers_topdown / layers_bottomup / svd),
# how many hops, fixed vs chained teacher; the one-hop cell is the
# reference and is submitted too so the comparison shares a commit.
#   LR=5e-5 MIX=0.5 bash recipe/slurm/submit_consolidate_curricula.sh [--dry-run]
set -euo pipefail
cd "$(dirname "$0")/../.."
RECIPE=recipe/eval/consolidate_lax1_replay_gap256_s64_lr5e5.yaml
LR=${LR:-5e-5}; MIX=${MIX:-0.5}; SCHED=${SCHED:-const}; WARM=${WARM:-0}; RK=${RK:-0}; STEPS=${STEPS:-64}
DRY=${1:-}

submit() {   # name hops degrade teacher
    local name=$1 hops=$2 how=$3 teacher=$4
    local cmd=(sbatch -J "$name" recipe/slurm/raven_eval.sbatch "$RECIPE"
               "eval_name=consolidate-gap256-cur-$name-340M"
               "tasks.0.args.sleep.lr=$LR" "tasks.0.args.sleep.lm_mix=$MIX"
               "tasks.0.args.sleep.lr_schedule=$SCHED" "tasks.0.args.sleep.warmup=$WARM"
               "tasks.0.args.sleep.retain_kl=$RK" "tasks.0.args.sleep.steps=$STEPS"
               "tasks.0.args.sleep.hops=[$hops]" "tasks.0.args.sleep.degrade=$how"
               "tasks.0.args.sleep.teacher=$teacher")
    if [ "$DRY" = --dry-run ]; then echo "${cmd[*]}"; else "${cmd[@]}"; fi
}

submit onehop "0" scale fixed
for how in scale layers_topdown layers_bottomup svd; do
    submit "$how-2fix"   "0.5, 0"                 "$how" fixed
    submit "$how-2chain" "0.5, 0"                 "$how" chain
    submit "$how-4fix"   "0.75, 0.5, 0.25, 0"     "$how" fixed
done
