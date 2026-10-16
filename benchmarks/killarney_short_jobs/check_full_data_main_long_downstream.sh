#!/bin/bash

# Login-node-safe metadata audit for the 1000/1500-epoch transfer results.
# This invokes neither Python nor Slurm.

set -u
set -o pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
PROJECT_ROOT=$(readlink -f "$SCRIPT_DIR/../..")
RESULTS_ROOT=${MAINLONG_RESULTS_ROOT:-$PROJECT_ROOT/benchmarks/full_data_main_long/results}
EXPECTED_PROTOCOL=full_data_main_simclr1000_barlow1500_v1

if (( $# > 1 )); then
    echo "Usage: bash $0 [MAINLONG_BATCH]" >&2
    exit 2
fi

shopt -s nullglob
if (( $# == 1 )); then
    batch=$1
else
    batches=("$RESULTS_ROOT"/batch_*)
    (( ${#batches[@]} > 0 )) || {
        echo "No long-run main-table batches found under $RESULTS_ROOT" >&2
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

evaluation_count=0
failed=0

printf '%s\n' 'Long-run CIFAR downstream audit'
printf 'Batch: %s\n' "$batch"
printf 'Protocol: %s\n\n' "$EXPECTED_PROTOCOL"

for source in CIFAR10 CIFAR100; do
    if [[ $source == CIFAR10 ]]; then
        prefix=cifar10
        targets=(CIFAR100 Aircraft CUB2011 Dogs Flowers TinyImageNet)
        models=(ConvNet VGG11 ResNet18)
    else
        prefix=cifar100
        targets=(TinyImageNet CIFAR10 Aircraft CUB2011 Dogs Flowers)
        models=(ResNet18)
    fi

    for model in "${models[@]}"; do
        for ssl in simclr barlowtwins; do
            if [[ $ssl == simclr ]]; then epochs=1000; else epochs=1500; fi
            identifier="${prefix}_${model}_${ssl}_full_size100_${epochs}ep"
            config_count=0

            for target in "${targets[@]}"; do
                for label in 1 5; do
                    if [[ $label == 1 ]]; then probe=500; else probe=300; fi
                    eval_dir="$batch/evaluations/${identifier}__${target}_lbl${label}"
                    valid=0
                    for job_dir in "$eval_dir"/job_*; do
                        summary="$job_dir/summary.json"
                        [[ -s $summary ]] || continue
                        grep -Eq '"protocol"[[:space:]]*:[[:space:]]*"'"$EXPECTED_PROTOCOL"'"' "$summary" || continue
                        grep -Eq '"source"[[:space:]]*:[[:space:]]*"'"$source"'"' "$summary" || continue
                        grep -Eq '"target"[[:space:]]*:[[:space:]]*"'"$target"'"' "$summary" || continue
                        grep -Eq '"model"[[:space:]]*:[[:space:]]*"'"$model"'"' "$summary" || continue
                        grep -Eq '"ssl_method"[[:space:]]*:[[:space:]]*"'"$ssl"'"' "$summary" || continue
                        grep -Eq '"ssl_epochs"[[:space:]]*:[[:space:]]*'"$epochs"'([,[:space:]]|$)' "$summary" || continue
                        grep -Eq '"label_percentage"[[:space:]]*:[[:space:]]*'"$label"'([,[:space:]]|$)' "$summary" || continue
                        grep -Eq '"probe_epochs"[[:space:]]*:[[:space:]]*'"$probe"'([,[:space:]]|$)' "$summary" || continue
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
                        config_count=$((config_count + 1))
                        printf 'EVAL %-68s target=%-12s labels=%s%% mean=%s std=%s\n' \
                            "$identifier" "$target" "$label" "$mean" "$std"
                    else
                        printf 'MISSING %-65s target=%-12s labels=%s%% valid=%d\n' \
                            "$identifier" "$target" "$label" "$valid" >&2
                        failed=1
                    fi
                done
            done
            printf 'CONFIG %-66s downstream=%d/12\n' "$identifier" "$config_count"
        done
    done
done

printf '\nDownstream evaluations: %d/96\n' "$evaluation_count"
if (( failed == 0 && evaluation_count == 96 )); then
    printf '%s\n' 'AUDIT PASSED: every long-run downstream result is complete.'
    exit 0
fi

printf '%s\n' 'AUDIT FAILED: one or more downstream results are missing, duplicated, or invalid.' >&2
exit 1
