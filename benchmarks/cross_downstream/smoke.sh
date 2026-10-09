#!/bin/bash
#SBATCH --account=aip-boyuwang
#SBATCH --job-name=cross_smoke
#SBATCH --time=0-00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --gres=gpu:l40s:1
#SBATCH --output=benchmarks/cross_downstream/smoke-%j.out

set -euo pipefail
PROJECT_ROOT="${SLURM_SUBMIT_DIR:?Submit from the MKDT project directory}"
export PROJECT_ROOT CROSS_RUN_ROOT="$SLURM_TMPDIR/cross-smoke"
source "$PROJECT_ROOT/benchmarks/cross_downstream/environment.sh"
mkdir -p "$CROSS_RUN_ROOT"
srun --unbuffered python3 benchmarks/benchmark_cross_dataset.py pretrain \
    --source CIFAR100 --source-cache "$CIFAR100_SOURCE_CACHE" \
    --selection-dir "$CIFAR100_SELECTION_DIR" --output "$CROSS_RUN_ROOT/encoder" \
    --runs 1 --method random --ssl-method simclr --subset-percentage 1 --ssl-epochs 1
srun --unbuffered python3 benchmarks/benchmark_cross_dataset.py evaluate \
    --source CIFAR100 --target TinyImageNet --target-cache "$TINYIMAGENET_SOURCE_CACHE" \
    --encoder-dir "$CROSS_RUN_ROOT/encoder" --output "$CROSS_RUN_ROOT/evaluation" \
    --runs 1 --method random --ssl-method simclr --subset-percentage 1 \
    --ssl-epochs 1 --label-percentage 1 --probe-epochs 1
