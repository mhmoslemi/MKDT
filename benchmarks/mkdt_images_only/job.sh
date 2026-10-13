#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=15G
#SBATCH --gres=gpu:l40s:1
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
[[ $# -eq 5 ]] || { echo "Usage: $0 <pretrain|evaluate> <ConvNet|VGG11|ResNet18> <1|2|5> <simclr|barlowtwins> <epochs>" >&2; exit 2; }
phase=$1
model=$2
size=$3
ssl=$4
epochs=$5

source "$PROJECT_ROOT/benchmarks/mkdt_images_only/environment.sh"
config="${model}_size${size}_${ssl}"
encoder="$MKDT_BATCH_ROOT/runs/$config/encoders"

case "$size" in
    1) image_path="${MKDT_IMAGE_1:-$MKDT_IMAGE_ROOT/CIFAR10_1pct/images_5000.pt}" ;;
    2) image_path="${MKDT_IMAGE_2:-$MKDT_IMAGE_ROOT/CIFAR10_2pct/images_5000.pt}" ;;
    5) image_path="${MKDT_IMAGE_5:-$MKDT_IMAGE_ROOT/CIFAR10_5pct/images_5000.pt}" ;;
    *) echo "Invalid subset size: $size" >&2; exit 2 ;;
esac

case "$phase" in
    pretrain)
        [[ -f "$image_path" ]] || { echo "Missing MKDT image tensor: $image_path" >&2; exit 1; }
        srun --unbuffered python3 benchmarks/benchmark_mkdt_images.py pretrain \
            --images "$image_path" --output "$encoder" \
            --model "$model" \
            --ssl-method "$ssl" --subset-percentage "$size" \
            --ssl-epochs "$epochs" --runs 10
        ;;
    evaluate)
        [[ -f "$encoder/summary.json" ]] || { echo "Missing completed encoder summary: $encoder/summary.json" >&2; exit 1; }
        srun --unbuffered python3 benchmarks/benchmark_mkdt_images.py evaluate-all \
            --encoder-dir "$encoder" --output "$MKDT_BATCH_ROOT/runs/$config" \
            --data-path "$DATA_ROOT" --target-cache-root "$TARGET_CACHE_ROOT" \
            --model "$model" \
            --ssl-method "$ssl" --subset-percentage "$size" \
            --ssl-epochs "$epochs" --runs 10
        ;;
    *) echo "Invalid phase: $phase" >&2; exit 2 ;;
esac
