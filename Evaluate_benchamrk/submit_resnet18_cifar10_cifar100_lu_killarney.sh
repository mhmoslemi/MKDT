#!/bin/bash
# ResNet18 BN only, main tables only. Existing submitter excludes kn101.
set -euo pipefail
bench_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
export CIFAR_FINISH_MODEL=ResNet18

# CIFAR10 Ours/MKDT and remaining CIFAR100 Random probes/training.
export CIFAR_FINISH_MANIFEST="$bench_dir/cifar_finish_jobs/tasks.tsv"
export CIFAR_FINISH_WORKER="$bench_dir/cifar_finish_jobs/worker.py"
export CIFAR_FINISH_MAX_SECONDS=0
bash "$bench_dir/submit_cifar_finish_lu_killarney.sh" "$@"

# CIFAR100 Ours/MKDT retain the requested under-one-hour limit, margin included.
export CIFAR_FINISH_MANIFEST="$bench_dir/cifar_finish_jobs/cifar100_tasks.tsv"
export CIFAR_FINISH_WORKER="$bench_dir/cifar_finish_jobs/worker_cifar100.py"
export CIFAR_FINISH_MAX_SECONDS=3599
exec bash "$bench_dir/submit_cifar_finish_lu_killarney.sh" "$@"
