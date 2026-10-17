#!/bin/bash
set -euo pipefail

usage() {
    echo "Usage: $0 <pretrain|evaluate> <CIFAR10|CIFAR100> <ConvNet|VGG11|ResNet18> <simclr|barlowtwins> [seed]" >&2
    exit 2
}

(( $# >= 4 )) || usage

: "${MAINLONG_CODE_ROOT:?}"
source "$MAINLONG_CODE_ROOT/benchmarks/killarney_short_jobs/full_data_main_long_environment.sh"

task=$1
source_dataset=$2
model=$3
ssl_method=$4
shift 4

case "$source_dataset" in
    CIFAR10) source_prefix=cifar10 ;;
    CIFAR100) source_prefix=cifar100 ;;
    *) echo "Unsupported source dataset: $source_dataset" >&2; exit 2 ;;
esac

case "$model" in
    ConvNet|VGG11|ResNet18) ;;
    *) echo "Unsupported model: $model" >&2; exit 2 ;;
esac

case "$ssl_method" in
    simclr) ssl_epochs=1000 ;;
    barlowtwins) ssl_epochs=1500 ;;
    *) echo "Unsupported SSL method: $ssl_method" >&2; exit 2 ;;
esac

identifier="${source_prefix}_${model}_${ssl_method}_full_size100_${ssl_epochs}ep"
encoder_root="$MAINLONG_BATCH/encoders/$identifier"
program=benchmarks/benchmark_full_data_main_long.py

pretrain() {
    (( $# <= 1 )) || usage
    local seed output

    if (( $# == 1 )); then
        seed=$1
    else
        : "${SLURM_ARRAY_TASK_ID:?Pretraining requires an array task ID or an explicit seed.}"
        seed=$SLURM_ARRAY_TASK_ID
    fi
    [[ $seed =~ ^[0-9]+$ ]] && (( seed >= 0 && seed < 10 )) || {
        echo "Invalid encoder seed: $seed" >&2
        exit 2
    }

    output="$encoder_root/shard_${seed}_${seed}"
    if [[ -e $output ]]; then
        # A failed job may leave a config-only directory. Preserve it and put
        # an explicit manual retry in a separate immutable shard.
        output="${output}_retry_${SLURM_JOB_ID}"
    fi
    [[ -s $TARGET_CACHE_ROOT/$source_dataset.pt ]] || {
        echo "Missing source cache: $TARGET_CACHE_ROOT/$source_dataset.pt" >&2
        exit 1
    }

    mkdir -p "$encoder_root"
    cd "$MAINLONG_CODE_ROOT"
    srun --unbuffered python3 "$program" pretrain \
        --source "$source_dataset" \
        --source-cache "$TARGET_CACHE_ROOT/$source_dataset.pt" \
        --output "$output" \
        --model "$model" \
        --ssl-method "$ssl_method" \
        --ssl-epochs "$ssl_epochs" \
        --seed-start "$seed" \
        --runs 1 \
        --ssl-log-every 100
}

check_checkpoints() {
    local seed checkpoint matches
    shopt -s nullglob
    for (( seed=0; seed<10; seed++ )); do
        printf -v checkpoint 'seed_%03d.pt' "$seed"
        matches=("$encoder_root"/shard_*/"$checkpoint")
        (( ${#matches[@]} == 1 )) && [[ -s ${matches[0]} ]] || {
            echo "Expected exactly one checkpoint for seed $seed under $encoder_root; found ${#matches[@]}" >&2
            exit 1
        }
    done
}

evaluate() {
    (( $# == 0 )) || usage
    local label probe output

    check_checkpoints
    [[ -s $TARGET_CACHE_ROOT/$source_dataset.pt ]] || {
        echo "Missing target cache: $TARGET_CACHE_ROOT/$source_dataset.pt" >&2
        exit 1
    }

    cd "$MAINLONG_CODE_ROOT"
    for label in 1 5; do
        probe=$(jq -er --arg target "$source_dataset" --arg label "$label" \
            '.evaluation_by_dataset[$target].linear_probe_epochs_by_label_percentage[$label]
             | select(type == "number" and . > 0 and . == floor)' \
            "$MAINLONG_CODE_ROOT/OT-SSL-DD/training_epochs.json")
        output="$MAINLONG_BATCH/evaluations/${identifier}__${source_dataset}_lbl${label}/job_$SLURM_JOB_ID"
        mkdir -p "$(dirname "$output")"
        srun --unbuffered python3 "$program" evaluate \
            --source "$source_dataset" \
            --encoder-dir "$encoder_root" \
            --target "$source_dataset" \
            --target-cache "$TARGET_CACHE_ROOT/$source_dataset.pt" \
            --output "$output" \
            --model "$model" \
            --ssl-method "$ssl_method" \
            --ssl-epochs "$ssl_epochs" \
            --label-percentage "$label" \
            --probe-epochs "$probe" \
            --runs 10
    done
}

case "$task" in
    pretrain) pretrain "$@" ;;
    evaluate) evaluate "$@" ;;
    *) usage ;;
esac
