#!/bin/bash
#SBATCH --account=aip-boyuwang
#SBATCH --job-name=tiny_prepare_kmeans
#SBATCH --time=0-00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --gres=gpu:l40s:1
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the MKDT project directory}}"
source "$PROJECT_ROOT/benchmarks/main/environment.sh"
mkdir -p "$MAIN_RUN_ROOT"
srun --unbuffered python3 benchmarks/benchmark_dataset.py prepare \
    --dataset TinyImageNet --target-cache "$TARGET_CACHE" \
    --output "$SELECTION_DIR" --runs 10 --seed-start 0
