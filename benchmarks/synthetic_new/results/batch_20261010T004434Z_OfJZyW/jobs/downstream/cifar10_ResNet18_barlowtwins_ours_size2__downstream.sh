#!/bin/bash
#SBATCH --account=aip-boyuwang
#SBATCH --job-name=syndown_cifar10_barlowtwins2
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
    --synthetic-root "$SYNTHETIC_INPUT_ROOT" --source CIFAR10 --target CIFAR100 \
    --target-cache "$TARGET_CACHE_ROOT/CIFAR100.pt" \
    --encoder-dir "$SYNTH_BATCH_ROOT/encoders/cifar10_ResNet18_barlowtwins_ours_size2" \
    --output "$SYNTH_BATCH_ROOT/evaluations/cifar10_ResNet18_barlowtwins_ours_size2__CIFAR100_lbl1/job_$SLURM_JOB_ID" \
    --model ResNet18 --ssl-method barlowtwins --subset-percentage 2 \
    --ssl-epochs 8000 --label-percentage 1 \
    --probe-epochs 500
srun --unbuffered python3 benchmarks/benchmark_synthetic_new.py evaluate \
    --synthetic-root "$SYNTHETIC_INPUT_ROOT" --source CIFAR10 --target CIFAR100 \
    --target-cache "$TARGET_CACHE_ROOT/CIFAR100.pt" \
    --encoder-dir "$SYNTH_BATCH_ROOT/encoders/cifar10_ResNet18_barlowtwins_ours_size2" \
    --output "$SYNTH_BATCH_ROOT/evaluations/cifar10_ResNet18_barlowtwins_ours_size2__CIFAR100_lbl5/job_$SLURM_JOB_ID" \
    --model ResNet18 --ssl-method barlowtwins --subset-percentage 2 \
    --ssl-epochs 8000 --label-percentage 5 \
    --probe-epochs 300
srun --unbuffered python3 benchmarks/benchmark_synthetic_new.py evaluate \
    --synthetic-root "$SYNTHETIC_INPUT_ROOT" --source CIFAR10 --target Aircraft \
    --target-cache "$TARGET_CACHE_ROOT/Aircraft.pt" \
    --encoder-dir "$SYNTH_BATCH_ROOT/encoders/cifar10_ResNet18_barlowtwins_ours_size2" \
    --output "$SYNTH_BATCH_ROOT/evaluations/cifar10_ResNet18_barlowtwins_ours_size2__Aircraft_lbl1/job_$SLURM_JOB_ID" \
    --model ResNet18 --ssl-method barlowtwins --subset-percentage 2 \
    --ssl-epochs 8000 --label-percentage 1 \
    --probe-epochs 500
srun --unbuffered python3 benchmarks/benchmark_synthetic_new.py evaluate \
    --synthetic-root "$SYNTHETIC_INPUT_ROOT" --source CIFAR10 --target Aircraft \
    --target-cache "$TARGET_CACHE_ROOT/Aircraft.pt" \
    --encoder-dir "$SYNTH_BATCH_ROOT/encoders/cifar10_ResNet18_barlowtwins_ours_size2" \
    --output "$SYNTH_BATCH_ROOT/evaluations/cifar10_ResNet18_barlowtwins_ours_size2__Aircraft_lbl5/job_$SLURM_JOB_ID" \
    --model ResNet18 --ssl-method barlowtwins --subset-percentage 2 \
    --ssl-epochs 8000 --label-percentage 5 \
    --probe-epochs 300
srun --unbuffered python3 benchmarks/benchmark_synthetic_new.py evaluate \
    --synthetic-root "$SYNTHETIC_INPUT_ROOT" --source CIFAR10 --target CUB2011 \
    --target-cache "$TARGET_CACHE_ROOT/CUB2011.pt" \
    --encoder-dir "$SYNTH_BATCH_ROOT/encoders/cifar10_ResNet18_barlowtwins_ours_size2" \
    --output "$SYNTH_BATCH_ROOT/evaluations/cifar10_ResNet18_barlowtwins_ours_size2__CUB2011_lbl1/job_$SLURM_JOB_ID" \
    --model ResNet18 --ssl-method barlowtwins --subset-percentage 2 \
    --ssl-epochs 8000 --label-percentage 1 \
    --probe-epochs 500
srun --unbuffered python3 benchmarks/benchmark_synthetic_new.py evaluate \
    --synthetic-root "$SYNTHETIC_INPUT_ROOT" --source CIFAR10 --target CUB2011 \
    --target-cache "$TARGET_CACHE_ROOT/CUB2011.pt" \
    --encoder-dir "$SYNTH_BATCH_ROOT/encoders/cifar10_ResNet18_barlowtwins_ours_size2" \
    --output "$SYNTH_BATCH_ROOT/evaluations/cifar10_ResNet18_barlowtwins_ours_size2__CUB2011_lbl5/job_$SLURM_JOB_ID" \
    --model ResNet18 --ssl-method barlowtwins --subset-percentage 2 \
    --ssl-epochs 8000 --label-percentage 5 \
    --probe-epochs 300
srun --unbuffered python3 benchmarks/benchmark_synthetic_new.py evaluate \
    --synthetic-root "$SYNTHETIC_INPUT_ROOT" --source CIFAR10 --target Dogs \
    --target-cache "$TARGET_CACHE_ROOT/Dogs.pt" \
    --encoder-dir "$SYNTH_BATCH_ROOT/encoders/cifar10_ResNet18_barlowtwins_ours_size2" \
    --output "$SYNTH_BATCH_ROOT/evaluations/cifar10_ResNet18_barlowtwins_ours_size2__Dogs_lbl1/job_$SLURM_JOB_ID" \
    --model ResNet18 --ssl-method barlowtwins --subset-percentage 2 \
    --ssl-epochs 8000 --label-percentage 1 \
    --probe-epochs 500
srun --unbuffered python3 benchmarks/benchmark_synthetic_new.py evaluate \
    --synthetic-root "$SYNTHETIC_INPUT_ROOT" --source CIFAR10 --target Dogs \
    --target-cache "$TARGET_CACHE_ROOT/Dogs.pt" \
    --encoder-dir "$SYNTH_BATCH_ROOT/encoders/cifar10_ResNet18_barlowtwins_ours_size2" \
    --output "$SYNTH_BATCH_ROOT/evaluations/cifar10_ResNet18_barlowtwins_ours_size2__Dogs_lbl5/job_$SLURM_JOB_ID" \
    --model ResNet18 --ssl-method barlowtwins --subset-percentage 2 \
    --ssl-epochs 8000 --label-percentage 5 \
    --probe-epochs 300
srun --unbuffered python3 benchmarks/benchmark_synthetic_new.py evaluate \
    --synthetic-root "$SYNTHETIC_INPUT_ROOT" --source CIFAR10 --target Flowers \
    --target-cache "$TARGET_CACHE_ROOT/Flowers.pt" \
    --encoder-dir "$SYNTH_BATCH_ROOT/encoders/cifar10_ResNet18_barlowtwins_ours_size2" \
    --output "$SYNTH_BATCH_ROOT/evaluations/cifar10_ResNet18_barlowtwins_ours_size2__Flowers_lbl1/job_$SLURM_JOB_ID" \
    --model ResNet18 --ssl-method barlowtwins --subset-percentage 2 \
    --ssl-epochs 8000 --label-percentage 1 \
    --probe-epochs 500
srun --unbuffered python3 benchmarks/benchmark_synthetic_new.py evaluate \
    --synthetic-root "$SYNTHETIC_INPUT_ROOT" --source CIFAR10 --target Flowers \
    --target-cache "$TARGET_CACHE_ROOT/Flowers.pt" \
    --encoder-dir "$SYNTH_BATCH_ROOT/encoders/cifar10_ResNet18_barlowtwins_ours_size2" \
    --output "$SYNTH_BATCH_ROOT/evaluations/cifar10_ResNet18_barlowtwins_ours_size2__Flowers_lbl5/job_$SLURM_JOB_ID" \
    --model ResNet18 --ssl-method barlowtwins --subset-percentage 2 \
    --ssl-epochs 8000 --label-percentage 5 \
    --probe-epochs 300
srun --unbuffered python3 benchmarks/benchmark_synthetic_new.py evaluate \
    --synthetic-root "$SYNTHETIC_INPUT_ROOT" --source CIFAR10 --target TinyImageNet \
    --target-cache "$TARGET_CACHE_ROOT/TinyImageNet.pt" \
    --encoder-dir "$SYNTH_BATCH_ROOT/encoders/cifar10_ResNet18_barlowtwins_ours_size2" \
    --output "$SYNTH_BATCH_ROOT/evaluations/cifar10_ResNet18_barlowtwins_ours_size2__TinyImageNet_lbl1/job_$SLURM_JOB_ID" \
    --model ResNet18 --ssl-method barlowtwins --subset-percentage 2 \
    --ssl-epochs 8000 --label-percentage 1 \
    --probe-epochs 500
srun --unbuffered python3 benchmarks/benchmark_synthetic_new.py evaluate \
    --synthetic-root "$SYNTHETIC_INPUT_ROOT" --source CIFAR10 --target TinyImageNet \
    --target-cache "$TARGET_CACHE_ROOT/TinyImageNet.pt" \
    --encoder-dir "$SYNTH_BATCH_ROOT/encoders/cifar10_ResNet18_barlowtwins_ours_size2" \
    --output "$SYNTH_BATCH_ROOT/evaluations/cifar10_ResNet18_barlowtwins_ours_size2__TinyImageNet_lbl5/job_$SLURM_JOB_ID" \
    --model ResNet18 --ssl-method barlowtwins --subset-percentage 2 \
    --ssl-epochs 8000 --label-percentage 5 \
    --probe-epochs 300
