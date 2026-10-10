#!/bin/bash
#SBATCH --account=aip-yiweilu
#SBATCH --job-name=synpre_cifar100_ConvNet_barlowtwins2
#SBATCH --time=0-02:00:00
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
    --output "$SYNTH_BATCH_ROOT/encoders/cifar100_ConvNet_barlowtwins_ours_size2" --model ConvNet \
    --ssl-method barlowtwins --subset-percentage 2 --ssl-epochs 8000
