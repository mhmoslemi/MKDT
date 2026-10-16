#!/bin/bash
set -euo pipefail

usage() {
    echo "Usage: $0 <CIFAR10|CIFAR100> <ConvNet|VGG11|ResNet18> <simclr|barlowtwins>" >&2
    exit 2
}

(( $# == 3 )) || usage

: "${MAINLONG_CODE_ROOT:?}"
source "$MAINLONG_CODE_ROOT/benchmarks/killarney_short_jobs/full_data_main_long_environment.sh"

source_dataset=$1
model=$2
ssl_method=$3

case "$source_dataset" in
    CIFAR10)
        source_prefix=cifar10
        targets=(CIFAR100 Aircraft CUB2011 Dogs Flowers TinyImageNet)
        ;;
    CIFAR100)
        source_prefix=cifar100
        targets=(TinyImageNet CIFAR10 Aircraft CUB2011 Dogs Flowers)
        ;;
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

expected_protocol=full_data_main_simclr1000_barlow1500_v1
identifier="${source_prefix}_${model}_${ssl_method}_full_size100_${ssl_epochs}ep"
encoder_root="$MAINLONG_BATCH/encoders/$identifier"
program=benchmarks/benchmark_full_data_main_long.py

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

evaluation_complete() {
    local target=$1
    local label=$2
    local probe=$3
    local eval_dir="$MAINLONG_BATCH/evaluations/${identifier}__${target}_lbl${label}"
    local job_dir summary
    local valid=0
    local seed_files=()

    shopt -s nullglob
    for job_dir in "$eval_dir"/job_*; do
        summary="$job_dir/summary.json"
        [[ -s $summary ]] || continue
        grep -Eq '"protocol"[[:space:]]*:[[:space:]]*"'"$expected_protocol"'"' "$summary" || continue
        grep -Eq '"source"[[:space:]]*:[[:space:]]*"'"$source_dataset"'"' "$summary" || continue
        grep -Eq '"target"[[:space:]]*:[[:space:]]*"'"$target"'"' "$summary" || continue
        grep -Eq '"model"[[:space:]]*:[[:space:]]*"'"$model"'"' "$summary" || continue
        grep -Eq '"ssl_method"[[:space:]]*:[[:space:]]*"'"$ssl_method"'"' "$summary" || continue
        grep -Eq '"ssl_epochs"[[:space:]]*:[[:space:]]*'"$ssl_epochs"'([,[:space:]]|$)' "$summary" || continue
        grep -Eq '"label_percentage"[[:space:]]*:[[:space:]]*'"$label"'([,[:space:]]|$)' "$summary" || continue
        grep -Eq '"probe_epochs"[[:space:]]*:[[:space:]]*'"$probe"'([,[:space:]]|$)' "$summary" || continue
        grep -Eq '"complete"[[:space:]]*:[[:space:]]*true' "$summary" || continue
        grep -Eq '"completed_runs"[[:space:]]*:[[:space:]]*10([,[:space:]]|$)' "$summary" || continue
        grep -Eq '"expected_runs"[[:space:]]*:[[:space:]]*10([,[:space:]]|$)' "$summary" || continue
        seed_files=("$job_dir"/seed_*.json)
        (( ${#seed_files[@]} == 10 )) || continue
        valid=$((valid + 1))
    done

    (( valid <= 1 )) || {
        echo "Duplicate complete evaluations for $identifier target=$target labels=$label%: $valid" >&2
        exit 1
    }
    (( valid == 1 ))
}

check_checkpoints
cd "$MAINLONG_CODE_ROOT"

for target in "${targets[@]}"; do
    [[ -s $TARGET_CACHE_ROOT/$target.pt ]] || {
        echo "Missing target cache: $TARGET_CACHE_ROOT/$target.pt" >&2
        exit 1
    }
    for label in 1 5; do
        if [[ $label == 1 ]]; then probe=500; else probe=300; fi
        if evaluation_complete "$target" "$label" "$probe"; then
            printf 'SKIP complete source=%s target=%s model=%s ssl=%s labels=%s%%\n' \
                "$source_dataset" "$target" "$model" "$ssl_method" "$label"
            continue
        fi

        output="$MAINLONG_BATCH/evaluations/${identifier}__${target}_lbl${label}/job_$SLURM_JOB_ID"
        mkdir -p "$(dirname "$output")"
        srun --unbuffered python3 "$program" evaluate \
            --source "$source_dataset" \
            --encoder-dir "$encoder_root" \
            --target "$target" \
            --target-cache "$TARGET_CACHE_ROOT/$target.pt" \
            --output "$output" \
            --model "$model" \
            --ssl-method "$ssl_method" \
            --ssl-epochs "$ssl_epochs" \
            --label-percentage "$label" \
            --probe-epochs "$probe" \
            --runs 10
    done
done
