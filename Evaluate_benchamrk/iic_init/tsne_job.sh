#!/bin/bash
set -euo pipefail
: "${SLURM_JOB_ID:?Submit through sbatch}"
[[ $# == 2 ]] || exit 2
module --force purge
module load StdEnv/2023 python/3.11.5
source /home/mmoslem3/ENV/bin/activate
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK" NUMEXPR_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export XDG_CACHE_HOME="${SLURM_TMPDIR:?}/cache"
export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib" TORCH_HOME="$XDG_CACHE_HOME/torch" HF_HOME="$XDG_CACHE_HOME/hf"
export TMPDIR="$SLURM_TMPDIR"
exec python -u Evaluate_benchamrk/iic_init/tsne_initialization.py "$1" "$2"
