#!/bin/bash
set -euo pipefail

: "${SLURM_JOB_ID:?Run this inside a GPU salloc allocation.}"
export PROJECT_ROOT=/scratch/mmoslem3/MKDT
export MKDT_MULTI_BATCH="$PROJECT_ROOT/benchmarks/mkdt_multidataset/results/batch_20261010T160046Z_Y6dqZS"
export MKDT_CODE_ROOT="$MKDT_MULTI_BATCH/code"
export MKDT_SYNTH_ROOT="$MKDT_MULTI_BATCH/synthetic"
export TARGET_CACHE_ROOT="$MKDT_MULTI_BATCH/targets"
export DATA_ROOT=/scratch/mmoslem3/data

output_dir="$MKDT_SYNTH_ROOT/work/init/CIFAR10/VGG11"
log="$MKDT_MULTI_BATCH/logs/mkdt_sort_CIFAR10_VGG11-salloc-${SLURM_JOB_ID}.out"
echo "Starting CIFAR10 VGG11 sorting; log: $log"
bash "$PROJECT_ROOT/benchmarks/mkdt_multidataset/synthesis_job.sh" \
    sort CIFAR10 VGG11 2>&1 | tee "$log"
for size in 1 2 5; do
    output="$output_dir/CIFAR10_barlow_twins_${size}_high_loss_indices.pkl"
    [[ -f "$output" ]] || { echo "Missing sorting output: $output" >&2; exit 1; }
done
for downstream_job in 1599502 1599503 1599504; do
    scontrol update JobId="$downstream_job" Dependency=
    scontrol release "$downstream_job"
done
echo "CIFAR10 VGG11 sorting completed; downstream distillation jobs released."
