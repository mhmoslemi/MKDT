#!/bin/bash
#SBATCH --account=aip-boyuwang
#SBATCH --job-name=ds_Flowers_ResNet18_barlowtwins_kmeans_size1_lbl1
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
mkdir -p "$DOWNSTREAM_RUN_ROOT/evaluations/Flowers_ResNet18_barlowtwins_kmeans_size1_lbl1"
srun --unbuffered python3 OT-SSL-DD/benchmark_downstream.py evaluate \
    --output "$DOWNSTREAM_RUN_ROOT/evaluations/Flowers_ResNet18_barlowtwins_kmeans_size1_lbl1/job_$SLURM_JOB_ID" \
    --target Flowers --target-cache "$DOWNSTREAM_RUN_ROOT/targets/Flowers.pt" \
    --encoder-dir "$DOWNSTREAM_RUN_ROOT/encoders/ResNet18_barlowtwins_kmeans_size1" \
    --model ResNet18 --method kmeans --ssl-method barlowtwins \
    --subset-percentage 1 --ssl-epochs 1200 \
    --label-percentage 1 --label-policy exact --probe-updates 400 --runs 15
