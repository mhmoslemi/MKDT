#!/bin/bash
# Source only from an allocated Slurm job.
set -euo pipefail
: "${SLURM_JOB_ID:?Use sbatch, not bash, to execute benchmark jobs.}"
: "${PROJECT_ROOT:?}"
cd "${BENCH_CODE_ROOT:-$PROJECT_ROOT}"
DATA_ROOT="${DATA_ROOT:-$SCRATCH/data}"
BENCH_RUN_ROOT="${BENCH_RUN_ROOT:-$PROJECT_ROOT/benchmarks/cifar10/results/manual}"
SELECTION_DIR="${SELECTION_DIR:-$BENCH_RUN_ROOT/selections}"
VENV_ACTIVATE="${VENV_ACTIVATE:-$HOME/ENV/bin/activate}"
module load StdEnv/2023 python/3.11.5
source "$VENV_ACTIVATE"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export HF_HOME="$SCRATCH/hf_cache"
export XDG_CACHE_HOME="$SCRATCH/cache"
