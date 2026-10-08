#!/bin/bash
#SBATCH --account=aip-boyuwang
#SBATCH --job-name=ds_selections
#SBATCH --time=0-01:05:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=15G
#SBATCH --gres=gpu:l40s:1
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the MKDT project directory}}"
source "$PROJECT_ROOT/benchmarks/downstream/environment.sh"
python3 benchmarks/benchmark_downstream.py prepare-selections \
    --data-path "$DATA_ROOT" --output "$DOWNSTREAM_RUN_ROOT/selections" \
    --source-batch "$SOURCE_CIFAR10_BATCH" --runs 15
