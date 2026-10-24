#!/bin/bash
set -euo pipefail
bench_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
output="$bench_dir/ours_eval_results_simclr2k_barlow10k_2ssl3probes_downstream"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}" OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
mkdir -p "$output/logs"
python -u "$bench_dir/eval_jobs/interactive_ours_downstream.py" CIFAR100 barlowtwins --output "$output" "$@" 2>&1 | tee -a "$output/logs/CIFAR100_barlowtwins.log"
