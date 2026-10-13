#!/bin/bash
set -euo pipefail

expected_allocation=1626662
if [[ "${SLURM_JOB_ID:-}" != "$expected_allocation" ]]; then
    echo "Run this inside Slurm allocation $expected_allocation." >&2
    exit 2
fi

export PROJECT_ROOT=/scratch/mmoslem3/MKDT
export MKDT_MULTI_BATCH="$PROJECT_ROOT/benchmarks/mkdt_multidataset/results/batch_20261010T160046Z_Y6dqZS"
export MKDT_CODE_ROOT="$MKDT_MULTI_BATCH/code"
export MKDT_SYNTH_ROOT="$MKDT_MULTI_BATCH/synthetic"
export TARGET_CACHE_ROOT="$MKDT_MULTI_BATCH/targets"
export DATA_ROOT=/scratch/mmoslem3/data

job_script="$PROJECT_ROOT/benchmarks/mkdt_multidataset/benchmark_job.sh"
log_dir="$MKDT_MULTI_BATCH/logs"

run_eval() {
    dataset=$1
    name="mkev_${dataset}_ConvNet_ResNet18_5_barlowtwins"
    log="$log_dir/${name}-salloc-${SLURM_JOB_ID}.out"
    echo "Starting $name; log: $log"
    bash "$job_script" evaluate "$dataset" ConvNet ResNet18 5 barlowtwins 6000 2>&1 | tee "$log"
}

cd "$PROJECT_ROOT"
run_eval CIFAR10
run_eval CIFAR100
echo "Both evaluations completed successfully."
