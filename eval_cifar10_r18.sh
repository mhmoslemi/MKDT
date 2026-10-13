#!/bin/bash
set -euo pipefail

: "${SLURM_JOB_ID:?Run this inside a GPU salloc allocation.}"
export PROJECT_ROOT=/scratch/mmoslem3/MKDT
export FULL_BATCH="$PROJECT_ROOT/benchmarks/full_data_refresh/results/batch_20261010T160040Z_73NiHv"
export FULL_CODE_ROOT="$FULL_BATCH/code"
export TARGET_CACHE_ROOT="$FULL_BATCH/targets"

config=full_CIFAR10_ResNet18_barlowtwins
encoder_root="$FULL_BATCH/runs/$config/encoders"
for seed in {0..9}; do
    checkpoint=$(printf '%s/shard_%d_%d/seed_%03d.pt' "$encoder_root" "$seed" "$seed" "$seed")
    [[ -f "$checkpoint" ]] || { echo "Missing encoder checkpoint: $checkpoint" >&2; exit 1; }
done

log="$FULL_BATCH/logs/fsp_C10_R18_bt_eval-salloc-${SLURM_JOB_ID}.out"
echo "Starting CIFAR10 ResNet18 full-data evaluation; log: $log"
bash "$PROJECT_ROOT/benchmarks/full_data_refresh/job.sh" \
    evaluate CIFAR10 ResNet18 barlowtwins 600 "$config" \
    CIFAR10,CIFAR100,TinyImageNet,Aircraft,CUB2011,Dogs,Flowers 2>&1 | tee "$log"
echo "CIFAR10 ResNet18 evaluation completed."
