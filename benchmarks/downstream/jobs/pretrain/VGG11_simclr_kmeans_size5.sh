#!/bin/bash
#SBATCH --account=aip-boyuwang
#SBATCH --job-name=ds_pre_VGG11_simclr_kmeans_size5
#SBATCH --time=0-01:25:00
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
    --data-path "$DATA_ROOT" --output "$DOWNSTREAM_RUN_ROOT/encoders/VGG11_simclr_kmeans_size5" \
    --selection-dir "$DOWNSTREAM_RUN_ROOT/selections" --runs 15 \
    --model VGG11 --ssl-method simclr --method kmeans \
    --subset-percentage 5 --ssl-epochs 500
