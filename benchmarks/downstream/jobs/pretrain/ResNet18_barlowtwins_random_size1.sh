#!/bin/bash
#SBATCH --account=aip-boyuwang
#SBATCH --job-name=ds_pre_ResNet18_barlowtwins_random_size1
#SBATCH --time=0-01:40:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=15G
#SBATCH --gres=gpu:l40s:1
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the MKDT project directory}}"
source "$PROJECT_ROOT/benchmarks/downstream/environment.sh"
mkdir -p "$DOWNSTREAM_RUN_ROOT/encoders"
srun --unbuffered python3 benchmarks/benchmark_downstream.py pretrain \
    --data-path "$DATA_ROOT" --output "$DOWNSTREAM_RUN_ROOT/encoders/ResNet18_barlowtwins_random_size1" \
    --selection-dir "$DOWNSTREAM_RUN_ROOT/selections" --runs 15 \
    --model ResNet18 --ssl-method barlowtwins --method random \
    --subset-percentage 1 --ssl-epochs 1200
