#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=15G
#SBATCH --gres=gpu:l40s:1
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
[[ $# -eq 7 ]] || { echo "Usage: $0 <pretrain|evaluate> <source> <generator> <model> <size> <ssl> <epochs>" >&2; exit 2; }
source "$PROJECT_ROOT/benchmarks/mkdt_multidataset/environment.sh"
phase=$1
source_dataset=$2
generator=$3
model=$4
size=$5
ssl=$6
epochs=$7
case "$source_dataset" in
    CIFAR10) mkdt_dataset=CIFAR10; final=5000 ;;
    CIFAR100) mkdt_dataset=CIFAR100; final=5000 ;;
    TinyImageNet) mkdt_dataset=Tiny; final=10000 ;;
    *) echo "Invalid source: $source_dataset" >&2; exit 2 ;;
esac
images="$MKDT_SYNTH_ROOT/generated/$generator/$source_dataset/${mkdt_dataset}_${size}pct/images_${final}.pt"
config="${source_dataset}_${generator}_${model}_size${size}_${ssl}"
encoder="$MKDT_MULTI_BATCH/runs/$config/encoders"
case "$phase" in
    pretrain)
        [[ -f $images ]] || { echo "Missing synthetic images: $images" >&2; exit 1; }
        srun --unbuffered python3 benchmarks/benchmark_mkdt_multidataset.py pretrain \
            --source "$source_dataset" --generator "$generator" \
            --images "$images" --output "$encoder" \
            --model "$model" --ssl-method "$ssl" \
            --subset-percentage "$size" --ssl-epochs "$epochs"
        ;;
    evaluate)
        [[ -f "$encoder/summary.json" ]] || { echo "Missing encoder summary: $encoder/summary.json" >&2; exit 1; }
        srun --unbuffered python3 benchmarks/benchmark_mkdt_multidataset.py evaluate \
            --source "$source_dataset" --generator "$generator" \
            --encoder-dir "$encoder" \
            --target-cache-root "$TARGET_CACHE_ROOT" \
            --output "$MKDT_MULTI_BATCH/runs/$config" \
            --model "$model" --ssl-method "$ssl" \
            --subset-percentage "$size" --ssl-epochs "$epochs"
        ;;
    *) echo "Invalid phase: $phase" >&2; exit 2 ;;
esac
