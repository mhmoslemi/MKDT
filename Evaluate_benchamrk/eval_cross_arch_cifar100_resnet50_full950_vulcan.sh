#!/bin/bash
set -euo pipefail
: "${SLURM_JOB_ID:?Submit this probe-only script with sbatch}"
[[ $# == 2 ]]
module --force purge
module load StdEnv/2023 python/3.11.5
source /home/mmoslem3/ENV/bin/activate
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export HF_HOME=/scratch/mmoslem3/MKDT/Evaluate_benchamrk/cross_arch_ablation/cache/hf
export XDG_CACHE_HOME=/scratch/mmoslem3/MKDT/Evaluate_benchamrk/cross_arch_ablation/cache/xdg
export TORCH_HOME=/scratch/mmoslem3/MKDT/Evaluate_benchamrk/cross_arch_ablation/cache/torch
export TMPDIR="${SLURM_TMPDIR:?}"
cd /scratch/mmoslem3/MKDT/Evaluate_benchamrk
exec python -u cross_arch_ablation/probe_full950.py "$1" "$2"
