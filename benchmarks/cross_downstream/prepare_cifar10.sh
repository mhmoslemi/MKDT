#!/bin/bash
#SBATCH --account=aip-boyuwang
#SBATCH --job-name=xpack_cifar10
#SBATCH --time=0-00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the MKDT project directory}}"
source "$PROJECT_ROOT/benchmarks/cross_downstream/environment.sh"
mkdir -p "$CROSS_RUN_ROOT/targets"
srun --unbuffered python3 benchmarks/benchmark_cross_dataset.py prepare-target \
    --device cpu --data-path "$DATA_ROOT" --target CIFAR10 \
    --output "$CROSS_RUN_ROOT/targets/CIFAR10.pt"
