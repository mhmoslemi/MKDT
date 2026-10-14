#!/bin/bash

# Metadata-only result audit. This does not run Python or submit Slurm jobs.

set -u
set -o pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
PROJECT_ROOT=$(readlink -f "$SCRIPT_DIR/../..")
RESULTS_ROOT=${COMPLETION_RESULTS_ROOT:-$PROJECT_ROOT/benchmarks/full_data_completion/results}

if (( $# > 1 )); then
    echo "Usage: bash $0 [COMPLETION_BATCH]" >&2
    exit 2
fi

shopt -s nullglob
if (( $# == 1 )); then
    batch=$1
else
    batches=("$RESULTS_ROOT"/batch_*)
    (( ${#batches[@]} > 0 )) || {
        echo "No completion batches found under $RESULTS_ROOT" >&2
        exit 1
    }
    batch=${batches[0]}
    for candidate in "${batches[@]}"; do batch=$candidate; done
fi

[[ -d $batch ]] || { echo "Missing batch: $batch" >&2; exit 1; }
[[ -s $batch/cross_batch.txt ]] || { echo "Missing $batch/cross_batch.txt" >&2; exit 1; }
cross_batch=$(<"$batch/cross_batch.txt")

printf '%s\n' 'Full-data table-completion audit'
printf 'Completion batch: %s\n' "$batch"
printf 'Existing cross batch: %s\n\n' "$cross_batch"

# Confirm the 36 blank downstream cells supplied by existing evaluations.
if ! bash "$SCRIPT_DIR/check_full_data_transfer.sh" "$cross_batch"; then
    echo "FAILED: prerequisite cross-batch results are incomplete." >&2
    exit 1
fi

extract_number() {
    local key=$1
    local file=$2
    sed -nE 's/^[[:space:]]*"'"$key"'"[[:space:]]*:[[:space:]]*([-+0-9.eE]+).*/\1/p' "$file" | sed -n '1p'
}

check_evaluation() {
    local identifier=$1
    local target=$2
    local label=$3
    local eval_dir="$batch/evaluations/${identifier}__${target}_lbl${label}"
    local job_dir seed path count mean std

    for job_dir in "$eval_dir"/job_*; do
        [[ -d $job_dir && -s $job_dir/summary.json ]] || continue
        grep -Eq '"complete"[[:space:]]*:[[:space:]]*true' "$job_dir/summary.json" || continue
        grep -Eq '"completed_runs"[[:space:]]*:[[:space:]]*10([,[:space:]]|$)' "$job_dir/summary.json" || continue
        count=0
        for (( seed=0; seed<10; seed++ )); do
            printf -v path '%s/seed_%03d.json' "$job_dir" "$seed"
            [[ -s $path ]] && count=$((count + 1))
        done
        (( count == 10 )) || continue
        mean=$(extract_number mean_percent "$job_dir/summary.json")
        std=$(extract_number std_percent "$job_dir/summary.json")
        printf 'OK  %-49s target=%-12s labels=%s%% mean=%s std=%s\n' \
            "$identifier" "$target" "$label" "$mean" "$std"
        return 0
    done

    printf 'MISSING %-45s target=%-12s labels=%s%%\n' \
        "$identifier" "$target" "$label" >&2
    return 1
}

completed=0
expected=20

for spec in \
    'cifar100_ResNet18_barlowtwins_full_size100 CIFAR100' \
    'tinyimagenet_ResNet18_simclr_full_size100 TinyImageNet' \
    'tinyimagenet_ResNet18_barlowtwins_full_size100 TinyImageNet'; do
    read -r identifier target <<< "$spec"
    for label in 1 5; do
        check_evaluation "$identifier" "$target" "$label" && completed=$((completed + 1))
    done
done

c10_identifier=cifar10_ResNet18_barlowtwins_full_size100
for target in CIFAR10 CIFAR100 Aircraft CUB2011 Dogs Flowers TinyImageNet; do
    for label in 1 5; do
        check_evaluation "$c10_identifier" "$target" "$label" && completed=$((completed + 1))
    done
done

printf '\nNew completion results: %d/%d table cells complete.\n' "$completed" "$expected"
if (( completed == expected )); then
    printf '%s\n' 'AUDIT PASSED: all 56 blank non-Ours cells are covered (36 existing + 20 new).'
    exit 0
fi

printf '%s\n' 'AUDIT FAILED: one or more new evaluations are missing or incomplete.' >&2
exit 1
