#!/bin/bash
set -euo pipefail
bench_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}" OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
mkdir -p "$bench_dir/ours_eval_results_interactive_2ssl3probes/logs"
python -u "$bench_dir/eval_jobs/interactive_ours_main.py" TinyImageNet "$@" 2>&1 | \
    tee -a "$bench_dir/ours_eval_results_interactive_2ssl3probes/logs/TinyImageNet${OURS_EVAL_LOG_SUFFIX:-}.log" | \
    awk -f "$bench_dir/eval_jobs/ssl_console.awk"
