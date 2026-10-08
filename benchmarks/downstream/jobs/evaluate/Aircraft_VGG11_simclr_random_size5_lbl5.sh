#!/bin/bash
#SBATCH --account=aip-boyuwang
#SBATCH --job-name=ds_Aircraft_VGG11_simclr_random_size5_lbl5
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
mkdir -p "$DOWNSTREAM_RUN_ROOT/evaluations/Aircraft_VGG11_simclr_random_size5_lbl5"
srun --unbuffered python3 benchmarks/benchmark_downstream.py evaluate \
    --output "$DOWNSTREAM_RUN_ROOT/evaluations/Aircraft_VGG11_simclr_random_size5_lbl5/job_$SLURM_JOB_ID" \
    --target Aircraft --target-cache "$DOWNSTREAM_RUN_ROOT/targets/Aircraft.pt" \
    --encoder-dir "$DOWNSTREAM_RUN_ROOT/encoders/VGG11_simclr_random_size5" \
    --model VGG11 --method random --ssl-method simclr \
    --subset-percentage 5 --ssl-epochs 500 \
    --label-percentage 5 --label-policy exact --probe-epochs 100 --runs 15
