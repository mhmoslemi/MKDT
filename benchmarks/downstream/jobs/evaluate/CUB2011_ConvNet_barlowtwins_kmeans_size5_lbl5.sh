#!/bin/bash
#SBATCH --account=aip-boyuwang
#SBATCH --job-name=ds_CUB2011_ConvNet_barlowtwins_kmeans_size5_lbl5
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
mkdir -p "$DOWNSTREAM_RUN_ROOT/evaluations/CUB2011_ConvNet_barlowtwins_kmeans_size5_lbl5"
srun --unbuffered python3 OT-SSL-DD/benchmark_downstream.py evaluate \
    --output "$DOWNSTREAM_RUN_ROOT/evaluations/CUB2011_ConvNet_barlowtwins_kmeans_size5_lbl5/job_$SLURM_JOB_ID" \
    --target CUB2011 --target-cache "$DOWNSTREAM_RUN_ROOT/targets/CUB2011.pt" \
    --encoder-dir "$DOWNSTREAM_RUN_ROOT/encoders/ConvNet_barlowtwins_kmeans_size5" \
    --model ConvNet --method kmeans --ssl-method barlowtwins \
    --subset-percentage 5 --ssl-epochs 240 \
    --label-percentage 5 --label-policy exact --probe-updates 400 --runs 15
