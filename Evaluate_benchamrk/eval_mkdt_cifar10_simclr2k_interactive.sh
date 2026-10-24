#!/bin/bash
set -euo pipefail
: "${SLURM_JOB_ID:?Run inside your existing GPU allocation}"
bench_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
output="$bench_dir/mkdt_eval_results_simclr2k_2ssl3probes"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p "$output/logs"
python -u "$bench_dir/eval_jobs/interactive_mkdt_2k.py" CIFAR10 --output "$output" "$@" 2>&1 |
    tee -a "$output/logs/CIFAR10.log" | awk -f "$bench_dir/eval_jobs/ssl_console.awk"
