#!/bin/bash
# Per-node launcher for multi-node training: one srun task per node runs
# this, and torchrun starts the 4 GPU workers of that node. The batch
# script exports MASTER_ADDR / MASTER_PORT (rendezvous on the first node)
# and the environment (module, venv) that srun propagates.
#
#   srun --kill-on-bad-exit=1 recipe/slurm/torchrun_node.sh <config> [key=value ...]
#
# --shutdown_timeout: on SIGTERM (walltime forwarding) the elastic agent
# re-signals the workers and would SIGKILL them after 30 s by default;
# the workers need the time to write the recent checkpoint first.
cd "$HOME/work/vagus" || exit 1
exec python -m torch.distributed.run \
    --nnodes "$SLURM_NNODES" --nproc_per_node 4 --node_rank "$SLURM_NODEID" \
    --rdzv_backend c10d --rdzv_endpoint "$MASTER_ADDR:$MASTER_PORT" --rdzv_id "$SLURM_JOB_ID" \
    --shutdown_timeout 900 \
    -m infra.train.main "$@"
