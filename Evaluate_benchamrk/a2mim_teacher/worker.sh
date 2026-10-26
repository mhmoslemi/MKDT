#!/bin/bash
set -euo pipefail
: "${SLURM_JOB_ID:?Submit using submit_a2mim_teachers_vulcan.sh}"
bench=$1 dataset=$2 venv=$3 data=$4
module --force purge
module load StdEnv/2023 gcc/12.3 cuda/12.6 python/3.11.5
source "$venv/bin/activate"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export XDG_CACHE_HOME="$bench/a2mim_teacher/cache" TORCH_HOME="$bench/a2mim_teacher/cache/torch"
export TMPDIR="${SLURM_TMPDIR:?}"
cd "$bench"
python -u a2mim_teacher/check.py
exec python -u a2mim_teacher/train.py --dataset "$dataset" --data-root "$data" \
    --epochs 300 --batch-size 256 --seed 0
