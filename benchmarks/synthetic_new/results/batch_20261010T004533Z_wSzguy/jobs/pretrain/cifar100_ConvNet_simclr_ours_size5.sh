#!/bin/bash
#SBATCH --account=aip-boyuwang
#SBATCH --job-name=synpre_cifar100_ConvNet_simclr5
#SBATCH --time=0-01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --gres=gpu:l40s:1
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
source "$PROJECT_ROOT/benchmarks/synthetic_new/environment.sh"
srun --unbuffered python3 benchmarks/benchmark_synthetic_new.py pretrain \
    --synthetic-root "$SYNTHETIC_INPUT_ROOT" --source CIFAR100 \
    --output "$SYNTH_BATCH_ROOT/encoders/cifar100_ConvNet_simclr_ours_size5" --model ConvNet \
    --ssl-method simclr --subset-percentage 5 --ssl-epochs 3000
