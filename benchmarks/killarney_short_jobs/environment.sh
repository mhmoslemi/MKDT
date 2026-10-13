#!/bin/bash
set -euo pipefail

: "${SLURM_JOB_ID:?This worker must run through sbatch.}"

export PROJECT_ROOT
PROJECT_ROOT=$(readlink -f /home/mmoslem3/scratch/MKDT)
export DATA_ROOT=/home/mmoslem3/scratch/data
export MKDT_MULTI_BATCH="$PROJECT_ROOT/benchmarks/mkdt_multidataset/results/batch_20261010T160046Z_Y6dqZS"
export MKDT_CODE_ROOT="$MKDT_MULTI_BATCH/code"
export MKDT_SYNTH_ROOT="$MKDT_MULTI_BATCH/synthetic"
export TARGET_CACHE_ROOT="$MKDT_MULTI_BATCH/targets"

[[ -d "$MKDT_CODE_ROOT" ]] || { echo "Missing code snapshot: $MKDT_CODE_ROOT" >&2; exit 1; }
[[ -f /home/mmoslem3/ENV/bin/activate ]] || {
    echo "Missing virtual environment: /home/mmoslem3/ENV/bin/activate" >&2
    exit 1
}

cd "$MKDT_CODE_ROOT"
source /home/mmoslem3/ENV/bin/activate

export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export HF_HOME=/home/mmoslem3/scratch/hf_cache
export XDG_CACHE_HOME=/home/mmoslem3/scratch/cache
export WANDB_MODE=offline
