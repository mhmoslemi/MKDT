#!/bin/bash
# CIFAR100 Ours/MKDT main only; skip jobs needing >=1 hour, including the margin.
set -euo pipefail
bench_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
export CIFAR_FINISH_MANIFEST="$bench_dir/cifar_finish_jobs/cifar100_tasks.tsv"
export CIFAR_FINISH_WORKER="$bench_dir/cifar_finish_jobs/worker_cifar100.py"
export CIFAR_FINISH_MAX_SECONDS=3599
exec bash "$bench_dir/submit_cifar_finish_lu_killarney.sh" "$@"
