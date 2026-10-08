#!/bin/bash
#SBATCH --account=aip-boyuwang
#SBATCH --job-name=bench_ResNet18_barlowtwins_kmeans_size1_lbl1
#SBATCH --time=0-01:40:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=15G
#SBATCH --gres=gpu:l40s:1
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the MKDT project directory}}"
source "$PROJECT_ROOT/benchmarks/cifar10/environment.sh"
CONFIG_ID=ResNet18_barlowtwins_kmeans_size1_lbl1
# Measured runtime estimate for all 15 repetitions: 37.4 minutes.
# The time limit includes one additional hour, rounded up to five minutes.
OUTPUT="$BENCH_RUN_ROOT/$CONFIG_ID/job_$SLURM_JOB_ID"
mkdir -p "$BENCH_RUN_ROOT/$CONFIG_ID"
srun --unbuffered python3 benchmarks/benchmark_cifar10.py run \
    --data-path "$DATA_ROOT" --output "$OUTPUT" \
    --selection-dir "$SELECTION_DIR" --runs 15 --seed-start 0 \
    --method kmeans --model ResNet18 --ssl-method barlowtwins \
    --subset-percentage 1 --label-percentage 1 \
    --ssl-epochs 1200 --probe-epochs 200
