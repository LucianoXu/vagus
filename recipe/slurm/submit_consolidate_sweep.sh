#!/bin/bash
# Phase-A sweep of the replay_kl consolidation on LAX1, gap 256, 64 steps,
# 32 items (the reading of consolidate_lax1_replay_gap256_s64_lr5e5.yaml,
# which is the (lr 5e-5, lm_mix 0.5, const, retain_kl 0) cell and is not
# resubmitted): peak lr x LM mix, lr schedule, and the retain-KL anchor.
# One 1-GPU job per cell through raven_eval.sbatch; the cell is written
# into eval_name, so each record lands as registry/eval/<eval_name>-<commit>.json.
#   bash recipe/slurm/submit_consolidate_sweep.sh            # submit
#   bash recipe/slurm/submit_consolidate_sweep.sh --dry-run  # print the commands
set -euo pipefail
cd "$(dirname "$0")/../.."
RECIPE=recipe/eval/consolidate_lax1_replay_gap256_s64_lr5e5.yaml
DRY=${1:-}

submit() {   # name lr lm_mix schedule warmup retain_kl
    local name=$1 lr=$2 mix=$3 sched=$4 warm=$5 rk=$6
    local cmd=(sbatch -J "$name" recipe/slurm/raven_eval.sbatch "$RECIPE"
               "eval_name=consolidate-gap256-$name-340M"
               "tasks.0.args.sleep.lr=$lr" "tasks.0.args.sleep.lm_mix=$mix"
               "tasks.0.args.sleep.lr_schedule=$sched" "tasks.0.args.sleep.warmup=$warm"
               "tasks.0.args.sleep.retain_kl=$rk")
    if [ "$DRY" = --dry-run ]; then echo "${cmd[*]}"; else "${cmd[@]}"; fi
}

# A. peak lr x LM mix (const schedule, no retain term); (5e-5, 0.5) exists
for lr in 2e-5 5e-5 1e-4; do
    for mix in 0 0.5 2; do
        [ "$lr" = 5e-5 ] && [ "$mix" = 0.5 ] && continue
        submit "lr$lr-mix$mix" "$lr" "$mix" const 0 0
    done
done
# B. schedules at lm_mix 0.5
submit "lr5e-5-mix0.5-cos" 5e-5 0.5 cosine 4 0
submit "lr1e-4-mix0.5-cos" 1e-4 0.5 cosine 4 0
submit "lr1e-4-mix0.5-lin" 1e-4 0.5 linear 4 0
# C. retain-KL anchor (lr 5e-5 const unless noted)
submit "lr5e-5-mix0-rk0.5" 5e-5 0 const 0 0.5
submit "lr5e-5-mix0-rk2" 5e-5 0 const 0 2
submit "lr5e-5-mix0.5-rk1" 5e-5 0.5 const 0 1
submit "lr1e-4-mix0.5-cos-rk1" 1e-4 0.5 cosine 4 1
