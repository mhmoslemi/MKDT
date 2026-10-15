#!/bin/bash

# Login-node-safe metadata audit. This invokes neither Python nor Slurm.

set -u
set -o pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
PROJECT_ROOT=$(readlink -f "$SCRIPT_DIR/../..")
RESULTS_ROOT=${REMAINING600_RESULTS_ROOT:-$PROJECT_ROOT/benchmarks/full_data_600_remaining/results}
EXPECTED_PROTOCOL=full_data_all_ssl_600epoch_v1
IDENTIFIER=cifar10_ResNet18_barlowtwins_full_size100_600ep

if (( $# > 1 )); then
    echo "Usage: bash $0 [REMAINING600_BATCH]" >&2
    exit 2
fi

shopt -s nullglob
if (( $# == 1 )); then
    batch=$1
else
    batches=("$RESULTS_ROOT"/batch_*)
    (( ${#batches[@]} > 0 )) || {
        echo "No remaining-cell batches found under $RESULTS_ROOT" >&2
        exit 1
    }
    batch=${batches[0]}
    for candidate in "${batches[@]}"; do batch=$candidate; done
fi

[[ -d $batch ]] || { echo "Missing batch: $batch" >&2; exit 1; }
[[ -s $batch/protocol.txt ]] || { echo "Missing $batch/protocol.txt" >&2; exit 1; }
[[ $(<"$batch/protocol.txt") == "$EXPECTED_PROTOCOL" ]] || {
    echo "Unexpected protocol in $batch/protocol.txt" >&2
    exit 1
}

extract_number() {
    local key=$1
    local file=$2
    sed -nE 's/^[[:space:]]*"'"$key"'"[[:space:]]*:[[:space:]]*([-+0-9.eE]+).*/\1/p' "$file" | sed -n '1p'
}

checkpoint_count=0
for seed in {0..9}; do
    printf -v filename 'seed_%03d.pt' "$seed"
    matches=("$batch/encoders/$IDENTIFIER"/shard_*/"$filename")
    if (( ${#matches[@]} == 1 )) && [[ -s ${matches[0]} ]]; then
        checkpoint_count=$((checkpoint_count + 1))
    fi
done

evaluation_count=0
failed=0
printf '%s\n' 'Remaining full-data-cell audit'
printf 'Batch: %s\n' "$batch"
printf 'Encoder checkpoints present: %d/10\n\n' "$checkpoint_count"

for target in CIFAR10 CIFAR100 Aircraft CUB2011 Dogs Flowers TinyImageNet; do
    for label in 1 5; do
        eval_dir="$batch/evaluations/${IDENTIFIER}__${target}_lbl${label}"
        valid=0
        for job_dir in "$eval_dir"/job_*; do
            summary="$job_dir/summary.json"
            [[ -s $summary ]] || continue
            grep -Eq '"protocol"[[:space:]]*:[[:space:]]*"'"$EXPECTED_PROTOCOL"'"' "$summary" || continue
            grep -Eq '"source"[[:space:]]*:[[:space:]]*"CIFAR10"' "$summary" || continue
            grep -Eq '"model"[[:space:]]*:[[:space:]]*"ResNet18"' "$summary" || continue
            grep -Eq '"ssl_method"[[:space:]]*:[[:space:]]*"barlowtwins"' "$summary" || continue
            grep -Eq '"ssl_epochs"[[:space:]]*:[[:space:]]*600([,[:space:]]|$)' "$summary" || continue
            grep -Eq '"complete"[[:space:]]*:[[:space:]]*true' "$summary" || continue
            grep -Eq '"completed_runs"[[:space:]]*:[[:space:]]*10([,[:space:]]|$)' "$summary" || continue
            grep -Eq '"expected_runs"[[:space:]]*:[[:space:]]*10([,[:space:]]|$)' "$summary" || continue
            seed_files=("$job_dir"/seed_*.json)
            (( ${#seed_files[@]} == 10 )) || continue
            valid=$((valid + 1))
            mean=$(extract_number mean_percent "$summary")
            std=$(extract_number std_percent "$summary")
        done

        if (( valid == 1 )); then
            evaluation_count=$((evaluation_count + 1))
            printf 'OK target=%-12s labels=%s%% mean=%s std=%s\n' \
                "$target" "$label" "$mean" "$std"
        else
            printf 'MISSING target=%-12s labels=%s%% valid=%d\n' \
                "$target" "$label" "$valid" >&2
            failed=1
        fi
    done
done

printf '\nComplete evaluations: %d/14\n' "$evaluation_count"
if (( failed == 0 && evaluation_count == 14 )); then
    printf '%s\n' 'AUDIT PASSED: all remaining non-Ours full-data cells are ready.'
    exit 0
fi

printf '%s\n' 'AUDIT FAILED: one or more table cells are not ready.' >&2
exit 1
