#!/bin/bash
#SBATCH --account=aip-boyuwang
#SBATCH --job-name=bench_prepare_kmeans
#SBATCH --time=0-01:05:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=15G
#SBATCH --gres=gpu:l40s:1
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the MKDT project directory}}"
source "$PROJECT_ROOT/benchmarks/cifar10/environment.sh"
mkdir -p "$SELECTION_DIR"
srun --unbuffered python3 OT-SSL-DD/benchmark_cifar10.py prepare \
    --data-path "$DATA_ROOT" --output "$SELECTION_DIR" --runs 15 --seed-start 0
