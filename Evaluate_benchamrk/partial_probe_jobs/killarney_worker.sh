#!/bin/bash
set -euo pipefail
: "${SLURM_JOB_ID:?Submit through sbatch}"
bench=$1 manifest=$2
module --force purge
module load StdEnv/2023 python/3.11.5 nodejs/20.16.0
source /home/mmoslem3/ENV/bin/activate
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export XDG_CACHE_HOME="$bench/partial_probe_jobs/cache/xdg"
export TORCH_HOME="$bench/partial_probe_jobs/cache/torch" HF_HOME="$bench/partial_probe_jobs/cache/hf"
export TMPDIR="${SLURM_TMPDIR:?}"
cd "$bench"
exec python -u partial_probe_jobs/worker.py "$manifest" /scratch/mmoslem3/data
