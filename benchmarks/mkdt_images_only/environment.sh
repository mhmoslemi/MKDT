#!/bin/bash
set -euo pipefail
: "${SLURM_JOB_ID:?Use Slurm for benchmark execution.}"
: "${PROJECT_ROOT:?}"
: "${MKDT_BATCH_ROOT:?}"

cd "${MKDT_CODE_ROOT:-$PROJECT_ROOT}"
module load StdEnv/2023 python/3.11.5
source "${VENV_ACTIVATE:-$HOME/ENV/bin/activate}"

export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export HF_HOME="$SCRATCH/hf_cache" XDG_CACHE_HOME="$SCRATCH/cache"
export DATA_ROOT="${DATA_ROOT:-$SCRATCH/data}"
export TARGET_CACHE_ROOT="${TARGET_CACHE_ROOT:-$PROJECT_ROOT/benchmarks/downstream/results/batch_20261009T043709Z_zC8EPD/targets}"
export MKDT_IMAGE_ROOT="${MKDT_IMAGE_ROOT:-$PROJECT_ROOT/logged_files/CIFAR10}"
