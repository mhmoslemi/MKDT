#!/bin/bash
#SBATCH --account=aip-boyuwang
#SBATCH --job-name=synmain_cifar10_VGG11_barlowtwins2
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
    --synthetic-root "$SYNTHETIC_INPUT_ROOT" --source CIFAR10 --target CIFAR10 \
    --target-cache "$CIFAR10_CACHE" \
    --encoder-dir "$SYNTH_BATCH_ROOT/encoders/cifar10_VGG11_barlowtwins_ours_size2" \
    --output "$SYNTH_BATCH_ROOT/evaluations/cifar10_VGG11_barlowtwins_ours_size2__CIFAR10_lbl1/job_$SLURM_JOB_ID" \
    --model VGG11 --ssl-method barlowtwins --subset-percentage 2 \
    --ssl-epochs 8000 --label-percentage 1 \
    --probe-epochs 500
srun --unbuffered python3 benchmarks/benchmark_synthetic_new.py evaluate \
    --synthetic-root "$SYNTHETIC_INPUT_ROOT" --source CIFAR10 --target CIFAR10 \
    --target-cache "$CIFAR10_CACHE" \
    --encoder-dir "$SYNTH_BATCH_ROOT/encoders/cifar10_VGG11_barlowtwins_ours_size2" \
    --output "$SYNTH_BATCH_ROOT/evaluations/cifar10_VGG11_barlowtwins_ours_size2__CIFAR10_lbl5/job_$SLURM_JOB_ID" \
    --model VGG11 --ssl-method barlowtwins --subset-percentage 2 \
    --ssl-epochs 8000 --label-percentage 5 \
    --probe-epochs 300
