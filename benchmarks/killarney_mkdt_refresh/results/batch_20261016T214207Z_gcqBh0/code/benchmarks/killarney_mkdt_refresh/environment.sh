#!/bin/bash
set -euo pipefail

: "${SLURM_JOB_ID:?Run this worker through sbatch, not on the login node.}"
: "${PROJECT_ROOT:?}"
: "${MKDT_REFRESH_BATCH:?}"
: "${MKDT_REFRESH_CODE_ROOT:?}"
: "${MKDT_SYNTH_ROOT:?}"
: "${TARGET_CACHE_ROOT:?}"

cd "$MKDT_REFRESH_CODE_ROOT"
source "${VENV_ACTIVATE:-/home/mmoslem3/ENV/bin/activate}"

export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export HF_HOME="${SCRATCH:-/scratch/mmoslem3}/hf_cache"
export XDG_CACHE_HOME="${SCRATCH:-/scratch/mmoslem3}/cache"
export TORCH_HOME="${SCRATCH:-/scratch/mmoslem3}/torch_cache"
export TMPDIR="${SLURM_TMPDIR:-/tmp}"
export WANDB_MODE=offline
