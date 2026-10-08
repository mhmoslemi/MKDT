#!/bin/bash
#SBATCH --account=aip-boyuwang
#SBATCH --job-name=bench_VGG11_simclr_random_size1_lbl5
#SBATCH --time=0-01:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=15G
#SBATCH --gres=gpu:l40s:1
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the MKDT project directory}}"
source "$PROJECT_ROOT/benchmarks/cifar10/environment.sh"
CONFIG_ID=VGG11_simclr_random_size1_lbl5
# Measured runtime estimate for all 15 repetitions: 12.0 minutes.
# The time limit includes one additional hour, rounded up to five minutes.
OUTPUT="$BENCH_RUN_ROOT/$CONFIG_ID/job_$SLURM_JOB_ID"
mkdir -p "$BENCH_RUN_ROOT/$CONFIG_ID"
srun --unbuffered python3 benchmarks/benchmark_cifar10.py run \
    --data-path "$DATA_ROOT" --output "$OUTPUT" \
    --selection-dir "$SELECTION_DIR" --runs 15 --seed-start 0 \
    --method random --model VGG11 --ssl-method simclr \
    --subset-percentage 1 --label-percentage 5 \
    --ssl-epochs 1200 --probe-epochs 100
