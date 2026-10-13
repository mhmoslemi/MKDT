#!/bin/bash
set -euo pipefail

: "${SLURM_JOB_ID:?Run this inside a GPU salloc allocation.}"
export PROJECT_ROOT=/scratch/mmoslem3/MKDT
export MKDT_MULTI_BATCH="$PROJECT_ROOT/benchmarks/mkdt_multidataset/results/batch_20261010T160046Z_Y6dqZS"
export MKDT_CODE_ROOT="$MKDT_MULTI_BATCH/code"
export MKDT_SYNTH_ROOT="$MKDT_MULTI_BATCH/synthetic"
export TARGET_CACHE_ROOT="$MKDT_MULTI_BATCH/targets"
export DATA_ROOT=/scratch/mmoslem3/data

start=60
end=79
destination="$MKDT_SYNTH_ROOT/work/buffers/CIFAR100/VGG11"
log="$MKDT_MULTI_BATCH/logs/mkdt_exp_CIFAR100_VGG11_${start}-salloc-${SLURM_JOB_ID}.out"

echo "Starting CIFAR100 VGG11 experts $start-$end; log: $log"
bash "$PROJECT_ROOT/benchmarks/mkdt_multidataset/synthesis_job.sh" \
    experts CIFAR100 VGG11 "$start" 2>&1 | tee "$log"
for index in $(seq "$start" "$end"); do
    [[ -f "$destination/replay_buffer_${index}.pt" ]] || { echo "Missing expert $index" >&2; exit 1; }
done
all_complete=true
for index in $(seq 0 99); do [[ -f "$destination/replay_buffer_${index}.pt" ]] || all_complete=false; done
if $all_complete; then
    scontrol update JobId=1599510 Dependency=
    scontrol release 1599510
fi
echo "Expert shard $start-$end completed."
