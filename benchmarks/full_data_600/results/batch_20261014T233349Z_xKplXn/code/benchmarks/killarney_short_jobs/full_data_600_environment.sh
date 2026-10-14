#!/bin/bash
set -euo pipefail

: "${SLURM_JOB_ID:?This worker must run through sbatch.}"
: "${PROJECT_ROOT:?}"
: "${REDO600_BATCH:?}"
: "${REDO600_CODE_ROOT:?}"
: "${TARGET_CACHE_ROOT:?}"

module load StdEnv/2023 python/3.11.5
source "${VENV_ACTIVATE:-$HOME/ENV/bin/activate}"

export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export HF_HOME="$SCRATCH/hf_cache"
export XDG_CACHE_HOME="$SCRATCH/cache"
export WANDB_MODE=offline
