#!/bin/bash
set -euo pipefail

[[ $# -eq 7 ]] || {
    echo "Usage: $0 <prepare-pretrain|pretrain|finalize-pretrain> <source> <generator> <model> <size> <ssl> <epochs>" >&2
    exit 2
}
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
common=(
    --source "$source_dataset" --generator "$generator"
    --output "$encoder" --model "$model" --ssl-method "$ssl"
    --subset-percentage "$size" --ssl-epochs "$epochs"
)

case "$phase" in
    prepare-pretrain)
        srun --unbuffered python3 benchmarks/benchmark_mkdt_multidataset.py \
            prepare-pretrain "${common[@]}" --images "$images" --device cpu
        ;;
    pretrain)
        : "${SLURM_ARRAY_TASK_ID:?Pretraining must run as a Slurm array task.}"
        srun --unbuffered python3 benchmarks/benchmark_mkdt_multidataset.py \
            pretrain "${common[@]}" --images "$images" \
            --seed-start "$SLURM_ARRAY_TASK_ID" --runs 1
        ;;
    finalize-pretrain)
        srun --unbuffered python3 benchmarks/benchmark_mkdt_multidataset.py \
            finalize-pretrain "${common[@]}" --device cpu
        ;;
    *) echo "Invalid phase: $phase" >&2; exit 2 ;;
esac
