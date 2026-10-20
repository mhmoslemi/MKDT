#!/bin/bash
# CIFAR-100 MKDT downstream evaluation inside an existing Killarney GPU allocation.
set -euo pipefail
: "${SLURM_JOB_ID:?Run this on the GPU node you allocated, not the login node}"
[[ ${SLURM_CLUSTER_NAME:-${CC_CLUSTER:-}} == killarney ]] || {
    echo 'Run this launcher on Killarney.' >&2; exit 2;
}
bench_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
module spider python/3.11.5 >/dev/null 2>&1
module load StdEnv/2023 python/3.11.5
source "${BENCH_VENV:-$HOME/ENV}/bin/activate"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export HF_HOME="$bench_dir/.eval_cache/hf" XDG_CACHE_HOME="$bench_dir/.eval_cache/xdg"
export TORCH_HOME="$bench_dir/.eval_cache/torch"
exec python "$bench_dir/eval_jobs/interactive_mkdt_split.py" --source CIFAR100 --scope downstream "$@"
