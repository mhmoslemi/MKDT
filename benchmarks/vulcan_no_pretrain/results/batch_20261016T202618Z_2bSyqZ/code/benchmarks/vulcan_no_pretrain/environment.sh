#!/bin/bash
set -euo pipefail

: "${SLURM_JOB_ID:?Run this worker through sbatch, not on the login node.}"
: "${PROJECT_ROOT:?}"
: "${NO_PRETRAIN_BATCH:?}"
: "${NO_PRETRAIN_CODE_ROOT:?}"
: "${TARGET_CACHE_ROOT:?}"

cd "$NO_PRETRAIN_CODE_ROOT"
module --force purge
module load StdEnv/2023 python/3.11.5
source "${VENV_ACTIVATE:-$HOME/ENV/bin/activate}"

export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export HF_HOME="$SCRATCH/hf_cache"
export XDG_CACHE_HOME="$SCRATCH/cache"
export TORCH_HOME="$SCRATCH/torch_cache"
export TMPDIR="${SLURM_TMPDIR:-/tmp}"
