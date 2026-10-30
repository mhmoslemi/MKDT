#!/bin/bash
set -euo pipefail
: "${SLURM_JOB_ID:?Submit through sbatch}"
[[ $# == 1 ]] || exit 2
module --force purge
module load StdEnv/2023 python/3.11.5
source /home/mmoslem3/ENV/bin/activate
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export XDG_CACHE_HOME="${SLURM_TMPDIR:?}/cache"
export TORCH_HOME="$XDG_CACHE_HOME/torch" HF_HOME="$XDG_CACHE_HOME/hf" MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
export TMPDIR="$SLURM_TMPDIR"
exec python -u Evaluate_benchamrk/iic_init/distill_cifar100.py "$1"
