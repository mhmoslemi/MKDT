#!/bin/bash
set -euo pipefail

usage() {
    echo "Usage: $0 <pretrain|evaluate> <source> <model> <ssl-method> [seed-start runs]" >&2
    exit 2
}

(( $# >= 4 )) || usage

: "${REDO600_CODE_ROOT:?}"
source "$REDO600_CODE_ROOT/benchmarks/killarney_short_jobs/full_data_600_environment.sh"

task=$1
source_dataset=$2
model=$3
ssl_method=$4
shift 4

case "$source_dataset:$model:$ssl_method" in
    CIFAR10:ConvNet:simclr|CIFAR10:VGG11:simclr|CIFAR10:ResNet18:simclr|\
    CIFAR100:ResNet18:simclr|CIFAR100:ResNet18:barlowtwins|\
    TinyImageNet:ResNet18:simclr|TinyImageNet:ResNet18:barlowtwins) ;;
    *)
        echo "Unsupported 600-epoch configuration: $source_dataset $model $ssl_method" >&2
        exit 2
        ;;
esac

case "$source_dataset" in
    CIFAR10) source_prefix=cifar10 ;;
    CIFAR100) source_prefix=cifar100 ;;
    TinyImageNet) source_prefix=tinyimagenet ;;
    *) exit 2 ;;
esac

identifier="${source_prefix}_${model}_${ssl_method}_full_size100_600ep"
encoder_root="$REDO600_BATCH/encoders/$identifier"
program=benchmarks/benchmark_full_data_600.py

pretrain() {
    (( $# == 2 )) || usage
    local seed_start=$1
    local runs=$2
    local final_seed output

    [[ $seed_start =~ ^[0-9]+$ && $runs =~ ^[0-9]+$ ]] \
        && (( runs > 0 && seed_start + runs <= 10 )) || {
        echo "Invalid seed shard: seed_start=$seed_start runs=$runs" >&2
        exit 2
    }
    final_seed=$((seed_start + runs - 1))
    output="$encoder_root/shard_${seed_start}_${final_seed}"
    [[ ! -e $output ]] || {
        echo "Refusing to overwrite existing encoder shard: $output" >&2
        exit 1
    }
    [[ -s $TARGET_CACHE_ROOT/$source_dataset.pt ]] || {
        echo "Missing source cache: $TARGET_CACHE_ROOT/$source_dataset.pt" >&2
        exit 1
    }

    mkdir -p "$encoder_root"
    cd "$REDO600_CODE_ROOT"
    srun --unbuffered python3 "$program" pretrain \
        --source "$source_dataset" \
        --source-cache "$TARGET_CACHE_ROOT/$source_dataset.pt" \
        --output "$output" \
        --model "$model" \
        --ssl-method "$ssl_method" \
        --ssl-epochs 600 \
        --seed-start "$seed_start" \
        --runs "$runs" \
        --ssl-log-every 100
}

check_checkpoints() {
    local seed checkpoint matches
    for (( seed=0; seed<10; seed++ )); do
        printf -v checkpoint 'seed_%03d.pt' "$seed"
        matches=("$encoder_root"/shard_*/"$checkpoint")
        (( ${#matches[@]} == 1 )) && [[ -s ${matches[0]} ]] || {
            echo "Expected exactly one checkpoint for seed $seed under $encoder_root" >&2
            exit 1
        }
    done
}

evaluate() {
    (( $# == 0 )) || usage
    local target label probe output

    shopt -s nullglob
    check_checkpoints
    cd "$REDO600_CODE_ROOT"
    for target in CIFAR10 CIFAR100 Aircraft CUB2011 Dogs Flowers TinyImageNet; do
        [[ -s $TARGET_CACHE_ROOT/$target.pt ]] || {
            echo "Missing target cache: $TARGET_CACHE_ROOT/$target.pt" >&2
            exit 1
        }
        for label in 1 5; do
            if [[ $label == 1 ]]; then
                probe=500
            else
                probe=300
            fi
            output="$REDO600_BATCH/evaluations/${identifier}__${target}_lbl${label}/job_$SLURM_JOB_ID"
            mkdir -p "$(dirname "$output")"
            srun --unbuffered python3 "$program" evaluate \
                --source "$source_dataset" \
                --encoder-dir "$encoder_root" \
                --target "$target" \
                --target-cache "$TARGET_CACHE_ROOT/$target.pt" \
                --output "$output" \
                --model "$model" \
                --ssl-method "$ssl_method" \
                --ssl-epochs 600 \
                --label-percentage "$label" \
                --probe-epochs "$probe" \
                --runs 10
        done
    done
}

case "$task" in
    pretrain) pretrain "$@" ;;
    evaluate) evaluate "$@" ;;
    *) usage ;;
esac
