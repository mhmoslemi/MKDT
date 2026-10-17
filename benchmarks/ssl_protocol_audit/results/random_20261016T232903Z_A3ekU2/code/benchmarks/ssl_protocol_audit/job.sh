#!/bin/bash
set -euo pipefail
: "${SLURM_JOB_ID:?Run through Slurm}"
: "${AUDIT_CODE:?}"
: "${AUDIT_OUTPUT:?}"
: "${AUDIT_KIND:?}"
: "${ENCODER_DIR:?}"
: "${TARGET_CACHE:?}"
: "${VENV_ACTIVATE:?}"
module load StdEnv/2023 python/3.11.5
source "$VENV_ACTIVATE"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export HF_HOME="$SCRATCH/hf_cache" XDG_CACHE_HOME="$SCRATCH/cache"
cd "$AUDIT_CODE"
srun --unbuffered python benchmarks/audit_cifar100_barlow.py \
    --kind "$AUDIT_KIND" --encoder-dir "$ENCODER_DIR" --target-cache "$TARGET_CACHE" \
    --epoch-config OT-SSL-DD/training_epochs.json --output "$AUDIT_OUTPUT" --runs 3
