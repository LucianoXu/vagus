#!/bin/bash -l
# FineWeb-Edu tokenization on one Raven CPU node.
#   submit from the repo root: sbatch recipe/slurm/raven_tokenize.sbatch <sample> <count>
#   e.g. sbatch recipe/slurm/raven_tokenize.sbatch 100BT 140
# Source parquet must already be on disk — compute nodes have no internet,
# so `python -m infra.dataset download` runs on a login node first.
# Idempotent: resubmitting skips finished shards and resumes.
#SBATCH --job-name=tokenize
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=144
#SBATCH --mem=240000
#SBATCH --time=24:00:00
#SBATCH --output=runs/slurm-%x-%j.out
#SBATCH --error=runs/slurm-%x-%j.err

set -euo pipefail
cd "$HOME/work/vagus"

module purge
module load python-waterboa/2025.06
source .venv/bin/activate

echo "Job $SLURM_JOB_ID on $(hostname) | sample=$1 count=$2"
python -m infra.dataset tokenize --sample "${1:?sample}" --count "${2:?count}"
