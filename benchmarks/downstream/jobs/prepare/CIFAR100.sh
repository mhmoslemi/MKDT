#!/bin/bash
#SBATCH --account=aip-boyuwang
#SBATCH --job-name=ds_pack_CIFAR100
#SBATCH --time=0-01:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=15G
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the MKDT project directory}}"
source "$PROJECT_ROOT/benchmarks/downstream/environment.sh"
mkdir -p "$DOWNSTREAM_RUN_ROOT/targets"
srun --unbuffered python3 benchmarks/benchmark_downstream.py prepare-target \
    --device cpu --data-path "$DATA_ROOT" --target CIFAR100 \
    --output "$DOWNSTREAM_RUN_ROOT/targets/CIFAR100.pt"
