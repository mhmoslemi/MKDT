#!/bin/bash
#SBATCH --account=aip-yiweilu
#SBATCH --job-name=synmain_cifar100_ConvNet_simclr2
#SBATCH --time=0-00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --gres=gpu:l40s:1
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
source "$PROJECT_ROOT/benchmarks/synthetic_new/environment.sh"
srun --unbuffered python3 benchmarks/benchmark_synthetic_new.py evaluate \
    --synthetic-root "$SYNTHETIC_INPUT_ROOT" --source CIFAR100 --target CIFAR100 \
    --target-cache "$TARGET_CACHE_ROOT/CIFAR100.pt" \
    --encoder-dir "$SYNTH_BATCH_ROOT/encoders/cifar100_ConvNet_simclr_ours_size2" \
    --output "$SYNTH_BATCH_ROOT/evaluations/cifar100_ConvNet_simclr_ours_size2__CIFAR100_lbl1/job_$SLURM_JOB_ID" \
    --model ConvNet --ssl-method simclr --subset-percentage 2 \
    --ssl-epochs 4000 --label-percentage 1 \
    --probe-epochs 500
srun --unbuffered python3 benchmarks/benchmark_synthetic_new.py evaluate \
    --synthetic-root "$SYNTHETIC_INPUT_ROOT" --source CIFAR100 --target CIFAR100 \
    --target-cache "$TARGET_CACHE_ROOT/CIFAR100.pt" \
    --encoder-dir "$SYNTH_BATCH_ROOT/encoders/cifar100_ConvNet_simclr_ours_size2" \
    --output "$SYNTH_BATCH_ROOT/evaluations/cifar100_ConvNet_simclr_ours_size2__CIFAR100_lbl5/job_$SLURM_JOB_ID" \
    --model ConvNet --ssl-method simclr --subset-percentage 2 \
    --ssl-epochs 4000 --label-percentage 5 \
    --probe-epochs 300
