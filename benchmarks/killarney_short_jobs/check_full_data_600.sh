#!/bin/bash

# Login-node-safe metadata audit. This never invokes Python or Slurm.

set -u
set -o pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
PROJECT_ROOT=$(readlink -f "$SCRIPT_DIR/../..")
RESULTS_ROOT=${REDO600_RESULTS_ROOT:-$PROJECT_ROOT/benchmarks/full_data_600/results}
EXPECTED_PROTOCOL=full_data_all_ssl_600epoch_v1

if (( $# > 1 )); then
    echo "Usage: bash $0 [REDO600_BATCH]" >&2
    exit 2
fi

shopt -s nullglob
if (( $# == 1 )); then
    batch=$1
else
    batches=("$RESULTS_ROOT"/batch_*)
    (( ${#batches[@]} > 0 )) || {
        echo "No 600-epoch batches found under $RESULTS_ROOT" >&2
        exit 1
    }
    batch=${batches[0]}
    for candidate in "${batches[@]}"; do
        batch=$candidate
    done
fi

[[ -d $batch ]] || { echo "Missing batch: $batch" >&2; exit 1; }
[[ -s $batch/protocol.txt ]] || { echo "Missing $batch/protocol.txt" >&2; exit 1; }
[[ $(<"$batch/protocol.txt") == "$EXPECTED_PROTOCOL" ]] || {
    echo "Unexpected protocol in $batch/protocol.txt" >&2
    exit 1
}

printf '%s\n' 'Fresh full-data 600-epoch audit'
printf 'Batch: %s\n' "$batch"
printf 'Protocol: %s\n\n' "$EXPECTED_PROTOCOL"

extract_number() {
    local key=$1
    local file=$2
    sed -nE 's/^[[:space:]]*"'"$key"'"[[:space:]]*:[[:space:]]*([-+0-9.eE]+).*/\1/p' "$file" | sed -n '1p'
}

checkpoint_count=0
evaluation_count=0
failed=0

check_encoder() {
    local identifier=$1
    local seed filename matches config configs
    local complete=0

    for (( seed=0; seed<10; seed++ )); do
        printf -v filename 'seed_%03d.pt' "$seed"
        matches=("$batch/encoders/$identifier"/shard_*/"$filename")
        if (( ${#matches[@]} == 1 )) && [[ -s ${matches[0]} ]]; then
            complete=$((complete + 1))
            checkpoint_count=$((checkpoint_count + 1))
        else
            printf 'MISSING checkpoint %-55s seed=%d found=%d\n' \
                "$identifier" "$seed" "${#matches[@]}" >&2
            failed=1
        fi
    done

    configs=("$batch/encoders/$identifier"/shard_*/config.json)
    if (( ${#configs[@]} != 2 )); then
        printf 'MISSING configs %-58s found=%d expected=2\n' \
            "$identifier" "${#configs[@]}" >&2
        failed=1
    fi
    for config in "${configs[@]}"; do
        grep -Eq '"protocol"[[:space:]]*:[[:space:]]*"'"$EXPECTED_PROTOCOL"'"' "$config" || {
            echo "BAD protocol: $config" >&2
            failed=1
        }
        grep -Eq '"ssl_epochs"[[:space:]]*:[[:space:]]*600([,[:space:]]|$)' "$config" || {
            echo "BAD epoch count: $config" >&2
            failed=1
        }
    done
    printf 'ENCODER %-55s checkpoints=%d/10\n' "$identifier" "$complete"
}

check_evaluation() {
    local identifier=$1
    local target=$2
    local label=$3
    local eval_dir="$batch/evaluations/${identifier}__${target}_lbl${label}"
    local job_dir summary mean std valid=0

    for job_dir in "$eval_dir"/job_*; do
        summary="$job_dir/summary.json"
        [[ -s $summary ]] || continue
        grep -Eq '"protocol"[[:space:]]*:[[:space:]]*"'"$EXPECTED_PROTOCOL"'"' "$summary" || continue
        grep -Eq '"ssl_epochs"[[:space:]]*:[[:space:]]*600([,[:space:]]|$)' "$summary" || continue
        grep -Eq '"complete"[[:space:]]*:[[:space:]]*true' "$summary" || continue
        grep -Eq '"completed_runs"[[:space:]]*:[[:space:]]*10([,[:space:]]|$)' "$summary" || continue
        grep -Eq '"expected_runs"[[:space:]]*:[[:space:]]*10([,[:space:]]|$)' "$summary" || continue
        valid=$((valid + 1))
        mean=$(extract_number mean_percent "$summary")
        std=$(extract_number std_percent "$summary")
    done

    if (( valid == 1 )); then
        evaluation_count=$((evaluation_count + 1))
        printf 'EVAL %-55s target=%-12s labels=%s%% mean=%s std=%s\n' \
            "$identifier" "$target" "$label" "$mean" "$std"
    else
        printf 'MISSING evaluation %-52s target=%-12s labels=%s%% valid=%d\n' \
            "$identifier" "$target" "$label" "$valid" >&2
        failed=1
    fi
}

for spec in \
    'cifar10_ConvNet_simclr_full_size100_600ep' \
    'cifar10_VGG11_simclr_full_size100_600ep' \
    'cifar10_ResNet18_simclr_full_size100_600ep' \
    'cifar100_ResNet18_simclr_full_size100_600ep' \
    'cifar100_ResNet18_barlowtwins_full_size100_600ep' \
    'tinyimagenet_ResNet18_simclr_full_size100_600ep' \
    'tinyimagenet_ResNet18_barlowtwins_full_size100_600ep'; do
    check_encoder "$spec"
    for target in CIFAR10 CIFAR100 Aircraft CUB2011 Dogs Flowers TinyImageNet; do
        for label in 1 5; do
            check_evaluation "$spec" "$target" "$label"
        done
    done
done

printf '\nCheckpoints: %d/70\n' "$checkpoint_count"
printf 'Evaluations: %d/98\n' "$evaluation_count"
if (( failed == 0 && checkpoint_count == 70 && evaluation_count == 98 )); then
    printf '%s\n' 'AUDIT PASSED: all fresh 600-epoch encoders and evaluations are complete.'
    exit 0
fi

printf '%s\n' 'AUDIT FAILED: one or more artifacts are missing, duplicated, or invalid.' >&2
exit 1
