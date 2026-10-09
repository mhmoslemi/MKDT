#!/bin/bash
set -euo pipefail
: "${SLURM_JOB_ID:?Use sbatch, not bash, to execute benchmark jobs.}"
: "${PROJECT_ROOT:?}"
: "${MAIN_RUN_ROOT:?}"
: "${TARGET_CACHE:?}"
: "${DATASET:?}"
cd "${MAIN_CODE_ROOT:-$PROJECT_ROOT}"
SELECTION_DIR="${SELECTION_DIR:-$MAIN_RUN_ROOT/selections}"
VENV_ACTIVATE="${VENV_ACTIVATE:-$HOME/ENV/bin/activate}"
module load StdEnv/2023 python/3.11.5
source "$VENV_ACTIVATE"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export HF_HOME="$SCRATCH/hf_cache"
export XDG_CACHE_HOME="$SCRATCH/cache"
