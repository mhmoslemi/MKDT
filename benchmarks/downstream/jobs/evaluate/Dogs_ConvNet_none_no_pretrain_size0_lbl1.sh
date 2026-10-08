#!/bin/bash
#SBATCH --account=aip-yiweilu
#SBATCH --job-name=ds_Dogs_ConvNet_none_no_pretrain_size0_lbl1
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
mkdir -p "$DOWNSTREAM_RUN_ROOT/evaluations/Dogs_ConvNet_none_no_pretrain_size0_lbl1"
srun --unbuffered python3 benchmarks/benchmark_downstream.py evaluate \
    --output "$DOWNSTREAM_RUN_ROOT/evaluations/Dogs_ConvNet_none_no_pretrain_size0_lbl1/job_$SLURM_JOB_ID" \
    --target Dogs --target-cache "$DOWNSTREAM_RUN_ROOT/targets/Dogs.pt" \
    --encoder-dir "$DOWNSTREAM_RUN_ROOT/encoders/none" \
    --model ConvNet --method no_pretrain --ssl-method none \
    --subset-percentage 0 --ssl-epochs 0 \
    --label-percentage 1 --label-policy exact --probe-epochs 200 --runs 15
