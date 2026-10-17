#!/bin/bash
set -euo pipefail
: "${SLURM_JOB_ID:?Use Slurm}"
: "${BATCH:?}"
[[ ${SLURM_CLUSTER_NAME:-} == vulcan && ${SLURM_CPUS_PER_TASK:-} == 1 ]] || exit 2
module --force purge
module load StdEnv/2023 python/3.11.5
source /home/mmoslem3/ENV/bin/activate
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export HF_HOME="$BATCH/cache/hf" XDG_CACHE_HOME="$BATCH/cache/xdg" TORCH_HOME="$BATCH/cache/torch"
export TMPDIR="${SLURM_TMPDIR:?}"
cd "$BATCH/code"
sha256sum --check --quiet "$BATCH/code.sha256"
(cd "$BATCH" && sha256sum --check --quiet bundle.sha256)
printf 'JOB=%s CLUSTER=%s ACCOUNT=%s HOST=%s CPU=1 RAM=6G\n' \
    "$SLURM_JOB_ID" "$SLURM_CLUSTER_NAME" "$SLURM_JOB_ACCOUNT" "$(hostname)"

