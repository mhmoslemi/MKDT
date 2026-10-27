#!/bin/bash
set -euo pipefail
bench=$1 venv=$2 data=$3
module --force purge
module load StdEnv/2023 python/3.11.5
source "$venv/bin/activate"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export HF_HOME="$bench/cross_arch_ablation/cache/hf" XDG_CACHE_HOME="$bench/cross_arch_ablation/cache/xdg"
export TORCH_HOME="$bench/cross_arch_ablation/cache/torch" TMPDIR="${SLURM_TMPDIR:?}"
cd "$bench"
exec python -u cross_arch_ablation/calibrate.py "$data"
