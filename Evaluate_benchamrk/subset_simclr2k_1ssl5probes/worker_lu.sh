#!/bin/bash
set -euo pipefail
[[ $# == 5 ]] || exit 2
bench=$1 manifest=$2 venv=$3 data=$4 worker_hash=$5
: "${SLURM_JOB_ID:?Submit through Slurm}"
[[ ${SLURM_JOB_ACCOUNT:-} == aip-yiweilu ]] || exit 2
[[ $(sha256sum "$bench/subset_simclr2k_1ssl5probes/worker_lu.py" | cut -d ' ' -f 1) == "$worker_hash" ]] || exit 2
module --force purge
module load StdEnv/2023 python/3.11.5 nodejs/20.16.0
source "$venv/bin/activate"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export HF_HOME="$bench/subset_simclr2k_1ssl5probes/cache/hf" XDG_CACHE_HOME="$bench/subset_simclr2k_1ssl5probes/cache/xdg"
export TORCH_HOME="$bench/subset_simclr2k_1ssl5probes/cache/torch" TMPDIR="${SLURM_TMPDIR:?}"
cd "$bench"
exec python -u subset_simclr2k_1ssl5probes/worker_lu.py "$manifest" "$data"
