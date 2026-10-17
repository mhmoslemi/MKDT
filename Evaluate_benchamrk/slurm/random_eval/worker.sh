#!/bin/bash
set -euo pipefail
: "${SLURM_JOB_ID:?Run inside a Slurm allocation}"
batch=$(cd -- "${1:?batch}" && pwd -P)
task=${2:?task}
cd "$batch"
sha256sum --check --quiet bundle.sha256
source "$batch/environment.sh"
module --force purge
module load StdEnv/2023 python/3.11.5
source "$BENCH_VENV/bin/activate"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export HF_HOME="$batch/cache/hf" XDG_CACHE_HOME="$batch/cache/xdg" TORCH_HOME="$batch/cache/torch"
export TMPDIR="${SLURM_TMPDIR:?}"
exec python -u "$batch/worker.py" "$batch" "$task"
