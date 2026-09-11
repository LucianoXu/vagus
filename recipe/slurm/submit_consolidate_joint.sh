#!/bin/bash
# Joint consolidation (the memory as a variable) on LAX1, gap 256, 64
# steps, 32 items, at the phase-A operating point (lr 5e-5, no LM mix,
# retain_kl 0.5). Cells: fixed vs re-anchored teacher, rows vs free memory
# parametrisation, penalty weight lam, memory lr; mem_lr 0 is the control
# (memory frozen at m: one hop plus a consistency term). The one-hop
# reference is consolidate-gap256-lr5e-5-mix0-rk0.5 (ratio 0.165).
#   bash recipe/slurm/submit_consolidate_joint.sh [--dry-run]
set -euo pipefail
cd "$(dirname "$0")/../.."
RECIPE=recipe/eval/consolidate_lax1_replay_gap256_s64_lr5e5.yaml
DRY=${1:-}
B=(tasks.0.args.sleep.lr=5e-5 tasks.0.args.sleep.lm_mix=0 tasks.0.args.sleep.retain_kl=0.5)

submit() {   # name mode mem_param lam mem_lr [extra overrides...]
    local name=$1 mode=$2 how=$3 lam=$4 mlr=$5; shift 5
    local cmd=(sbatch -J "$name" recipe/slurm/raven_eval.sbatch "$RECIPE"
               "eval_name=consolidate-gap256-joint-$name-340M" "${B[@]}"
               "tasks.0.args.sleep.mode=$mode" "tasks.0.args.sleep.mem_param=$how"
               "tasks.0.args.sleep.lam=$lam" "tasks.0.args.sleep.mem_lr=$mlr" "$@")
    if [ "$DRY" = --dry-run ]; then echo "${cmd[*]}"; else "${cmd[@]}"; fi
}

submit jf-rows-l1-m0.2    joint_fixed    rows 1   0.2  tasks.0.args.sleep.layer_probe=true
submit jf-rows-l1-m0.5    joint_fixed    rows 1   0.5
submit jf-rows-l1-m0      joint_fixed    rows 1   0        # control: memory frozen
submit jf-rows-l0.3-m0.2  joint_fixed    rows 0.3 0.2
submit jf-rows-l3-m0.2    joint_fixed    rows 3   0.2
submit jf-free-l1-m1e-3   joint_fixed    free 1   1e-3 tasks.0.args.sleep.layer_probe=true
submit jf-free-l1-m1e-2   joint_fixed    free 1   1e-2
submit jr-rows-l1-m0.2    joint_reanchor rows 1   0.2
submit jr-free-l1-m1e-3   joint_reanchor free 1   1e-3
