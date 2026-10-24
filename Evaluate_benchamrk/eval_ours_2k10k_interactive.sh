#!/bin/bash
set -euo pipefail
trap 'exit 130' INT
trap 'exit 143' TERM
bench_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
output="$bench_dir/ours_eval_results_simclr2k_barlow10k_2ssl3probes_main"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}" OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
mkdir -p "$output/logs"
status=0

if [[ "$#" == 0 ]]; then
    set -- CIFAR10 CIFAR100 TinyImageNet
fi
for dataset in "$@"; do
    case "$dataset" in
        CIFAR10|CIFAR100|TinyImageNet) ;;
        *) echo "Unknown dataset: $dataset" >&2; exit 2 ;;
    esac
    if python -u "$bench_dir/eval_jobs/interactive_ours_2k10k.py" "$dataset" --output "$output" 2>&1 | \
        tee -a "$output/logs/$dataset.log" | \
        awk -f "$bench_dir/eval_jobs/ssl_console.awk"; then
        continue
    else
        result=$?
        if [[ "$result" == 130 || "$result" == 143 ]]; then
            exit "$result"
        fi
        status=1
    fi
done
exit "$status"
