#!/bin/bash
set -euo pipefail
: "${SLURM_JOB_ID:?Run this worker through Slurm}"
: "${BATCH:?}"
[[ ${SLURM_CLUSTER_NAME:-} == vulcan ]] || { echo 'Expected Vulcan' >&2; exit 2; }
module --force purge
module load StdEnv/2023 python/3.11.5
source /home/mmoslem3/ENV/bin/activate
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export HF_HOME="$BATCH/cache/hf" XDG_CACHE_HOME="$BATCH/cache/xdg" TORCH_HOME="$BATCH/cache/torch"
export TMPDIR="${SLURM_TMPDIR:?Missing node-local temporary directory}"
export BENCH_DATA_ROOT=/scratch/mmoslem3/data
cd "$BATCH/code"
sha256sum --check --quiet "$BATCH/code.sha256"
printf 'JOB=%s ARRAY_TASK=%s CLUSTER=%s ACCOUNT=%s HOST=%s\n' \
    "$SLURM_JOB_ID" "${SLURM_ARRAY_TASK_ID:-none}" "$SLURM_CLUSTER_NAME" \
    "${SLURM_JOB_ACCOUNT:-unknown}" "$(hostname)"

