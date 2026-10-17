#!/bin/bash
set -euo pipefail
: "${SLURM_JOB_ID:?Submit this script through Slurm}"
scripts=${1:?absolute path to slurm/random_eval}
module --force purge
module load StdEnv/2023 python/3.11.5 nodejs/20.16.0
source "${BENCH_VENV:-$HOME/ENV}/bin/activate"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export HF_HOME="$SLURM_TMPDIR/hf" XDG_CACHE_HOME="$SLURM_TMPDIR/cache" TMPDIR="$SLURM_TMPDIR"
node "$scripts/test_plan.js"
python "$scripts/test_worker.py"
