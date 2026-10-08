#!/bin/bash
set -euo pipefail
: "${SLURM_JOB_ID:?Use Slurm for benchmark execution.}"
: "${PROJECT_ROOT:?}"
cd "${DOWNSTREAM_CODE_ROOT:-$PROJECT_ROOT}"
DATA_ROOT="${DATA_ROOT:-$SCRATCH/data}"
DOWNSTREAM_RUN_ROOT="${DOWNSTREAM_RUN_ROOT:-$PROJECT_ROOT/benchmarks/downstream/results/manual}"
SOURCE_CIFAR10_BATCH="${SOURCE_CIFAR10_BATCH:-$PROJECT_ROOT/benchmarks/cifar10/results/batch_20261008T163904Z_ELW6hy}"
TARGET_DATASETS="${TARGET_DATASETS:-CIFAR100,Aircraft,CUB2011,Dogs,Flowers,TinyImageNet}"
module load StdEnv/2023 python/3.11.5
source "${VENV_ACTIVATE:-$HOME/ENV/bin/activate}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export HF_HOME="$SCRATCH/hf_cache" XDG_CACHE_HOME="$SCRATCH/cache"
