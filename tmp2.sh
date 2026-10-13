#!/bin/bash
set -euo pipefail

: "${SLURM_JOB_ID:?Run this inside a GPU salloc allocation.}"

export PROJECT_ROOT=/scratch/mmoslem3/MKDT
export FULL_BATCH="$PROJECT_ROOT/benchmarks/full_data_refresh/results/batch_20261010T160040Z_73NiHv"
export FULL_CODE_ROOT="$FULL_BATCH/code"
export TARGET_CACHE_ROOT="$FULL_BATCH/targets"
export FULL_SSL_LOG_EVERY=6

queued_array=1599080
config=full_CIFAR100_ResNet18_barlowtwins
encoder_root="$FULL_BATCH/runs/$config/encoders"

if [[ $# -gt 1 ]]; then
    echo "Usage: bash tmp2.sh [seed 0-9]" >&2
    exit 2
fi

if [[ $# -eq 1 ]]; then
    seed=$1
else
    seed=
    for candidate in {0..9}; do
        checkpoint=$(printf '%s/shard_%d_%d/seed_%03d.pt' "$encoder_root" "$candidate" "$candidate" "$candidate")
        if [[ ! -f "$checkpoint" ]]; then
            seed=$candidate
            break
        fi
    done
    if [[ -z "$seed" ]]; then
        echo "All CIFAR100 seeds 0-9 are already complete."
        exit 0
    fi
fi

[[ "$seed" =~ ^[0-9]$ ]] || { echo "Seed must be one digit from 0 through 9." >&2; exit 2; }
shard="$encoder_root/shard_${seed}_${seed}"
checkpoint=$(printf '%s/seed_%03d.pt' "$shard" "$seed")

if [[ -f "$checkpoint" ]]; then
    echo "CIFAR100 seed $seed is already complete; canceling only its queued duplicate."
    scancel "${queued_array}_${seed}" 2>/dev/null || true
    exit 0
fi
if [[ -e "$shard" ]]; then
    echo "Incomplete shard already exists at $shard; move it aside before retrying." >&2
    exit 1
fi

export SLURM_ARRAY_TASK_ID=$seed
log="$FULL_BATCH/logs/fsp_C100_R18_bt-salloc-${SLURM_JOB_ID}_${seed}.out"
echo "Starting CIFAR100 seed $seed; log: $log"
echo "PROGRESS: 0%"
bash "$PROJECT_ROOT/benchmarks/full_data_refresh/seed_array_job.sh" \
    CIFAR100 ResNet18 barlowtwins 600 "$config" 2>&1 | tee "$log"

[[ -f "$checkpoint" ]] || { echo "Missing expected checkpoint: $checkpoint" >&2; exit 1; }
scancel "${queued_array}_${seed}" 2>/dev/null || true
echo "CIFAR100 seed $seed completed; queued duplicate ${queued_array}_${seed} canceled."
