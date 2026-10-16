#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --gres=gpu:l40s:1
#SBATCH --no-requeue

set -euo pipefail

if (( $# != 9 )); then
    echo "Usage: $0 <pretrain|evaluate> <source> <generator> <model> <size> <ssl> <epochs> <main|all> <config>" >&2
    exit 2
fi

source "$MKDT_REFRESH_CODE_ROOT/benchmarks/killarney_mkdt_refresh/environment.sh"

phase=$1
source_dataset=$2
generator=$3
model=$4
size=$5
ssl=$6
epochs=$7
scope=$8
config=$9

case "$source_dataset" in
    CIFAR10) mkdt_dataset=CIFAR10; final=5000 ;;
    CIFAR100) mkdt_dataset=CIFAR100; final=5000 ;;
    TinyImageNet) mkdt_dataset=Tiny; final=10000 ;;
    *) echo "Invalid source: $source_dataset" >&2; exit 2 ;;
esac
case "$generator:$model" in
    ConvNet:ConvNet|VGG11:VGG11|ConvNet:ResNet18) ;;
    *) echo "Invalid generator/backbone mapping: $generator -> $model" >&2; exit 2 ;;
esac
[[ $size == 1 || $size == 2 ]] || { echo "Invalid MKDT size: $size" >&2; exit 2; }
[[ $ssl == simclr || $ssl == barlowtwins ]] || { echo "Invalid SSL objective: $ssl" >&2; exit 2; }
[[ $epochs == 10000 ]] || { echo "Current 1%/2% MKDT protocol requires 10000 SSL epochs" >&2; exit 2; }
[[ $scope == main || $scope == all ]] || { echo "Invalid evaluation scope: $scope" >&2; exit 2; }

expected_config="${source_dataset}_${generator}_${model}_size${size}_${ssl}_pt10000"
[[ $config == "$expected_config" ]] || {
    echo "Config mismatch: found $config, expected $expected_config" >&2
    exit 2
}

images="$MKDT_SYNTH_ROOT/generated/$generator/$source_dataset/${mkdt_dataset}_${size}pct/images_${final}.pt"
encoder="$MKDT_REFRESH_BATCH/runs/$config/encoders"

case "$phase" in
    pretrain)
        [[ -s $images ]] || { echo "Missing synthetic images: $images" >&2; exit 1; }
        srun --unbuffered python3 benchmarks/benchmark_mkdt_refresh.py pretrain \
            --source "$source_dataset" \
            --generator "$generator" \
            --images "$images" \
            --output "$encoder" \
            --model "$model" \
            --ssl-method "$ssl" \
            --subset-percentage "$size" \
            --ssl-epochs "$epochs" \
            --seed-start 0 \
            --runs 1 \
            --ssl-log-every 250
        ;;
    evaluate)
        [[ -s $encoder/seed_000.pt ]] || {
            echo "Missing saved SSL encoder: $encoder/seed_000.pt" >&2
            exit 1
        }
        extra=()
        [[ $scope == all ]] && extra+=(--all-targets)
        srun --unbuffered python3 benchmarks/benchmark_mkdt_refresh.py evaluate \
            --source "$source_dataset" \
            --generator "$generator" \
            --encoder-dir "$encoder" \
            --target-cache-root "$TARGET_CACHE_ROOT" \
            --output "$MKDT_REFRESH_BATCH/runs/$config" \
            --model "$model" \
            --ssl-method "$ssl" \
            --subset-percentage "$size" \
            --ssl-epochs "$epochs" \
            --runs 15 \
            --reuse-encoder-seed 0 \
            "${extra[@]}"
        ;;
    *)
        echo "Invalid phase: $phase" >&2
        exit 2
        ;;
esac
