#!/bin/bash
set -euo pipefail

: "${SLURM_JOB_ID:?Use sbatch, not bash, to execute downstream workers.}"
: "${PROJECT_ROOT:?}"
: "${DOWNSTREAM_BATCH:?}"
: "${DOWNSTREAM_CODE_ROOT:?}"
: "${TARGET_CACHE_ROOT:?}"

cd "$DOWNSTREAM_CODE_ROOT"
module --force purge
module load StdEnv/2023 python/3.11.5
source "${VENV_ACTIVATE:-$HOME/ENV/bin/activate}"

export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export HF_HOME="$SCRATCH/hf_cache"
export XDG_CACHE_HOME="$SCRATCH/cache"
export TORCH_HOME="$SCRATCH/torch_cache"
export TMPDIR="${SLURM_TMPDIR:-/tmp}"
