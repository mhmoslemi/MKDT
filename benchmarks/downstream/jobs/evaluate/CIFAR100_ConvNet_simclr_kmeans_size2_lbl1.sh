#!/bin/bash
#SBATCH --account=aip-boyuwang
#SBATCH --job-name=ds_CIFAR100_ConvNet_simclr_kmeans_size2_lbl1
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
mkdir -p "$DOWNSTREAM_RUN_ROOT/evaluations/CIFAR100_ConvNet_simclr_kmeans_size2_lbl1"
srun --unbuffered python3 benchmarks/benchmark_downstream.py evaluate \
    --output "$DOWNSTREAM_RUN_ROOT/evaluations/CIFAR100_ConvNet_simclr_kmeans_size2_lbl1/job_$SLURM_JOB_ID" \
    --target CIFAR100 --target-cache "$DOWNSTREAM_RUN_ROOT/targets/CIFAR100.pt" \
    --encoder-dir "$DOWNSTREAM_RUN_ROOT/encoders/ConvNet_simclr_kmeans_size2" \
    --model ConvNet --method kmeans --ssl-method simclr \
    --subset-percentage 2 --ssl-epochs 800 \
    --label-percentage 1 --label-policy exact --probe-epochs 200 --runs 15
