#!/bin/bash

# Audit the saved 300-epoch full-data ResNet18 encoders and their transfer
# evaluations.  This is a metadata-only check and is safe to run on a login
# node; it does not load checkpoints or submit jobs.

set -u
set -o pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
PROJECT_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd -P)
RESULTS_ROOT=${CROSS_RESULTS_ROOT:-$PROJECT_ROOT/benchmarks/cross_downstream/results}
EXPECTED_SSL_EPOCHS=${EXPECTED_SSL_EPOCHS:-300}
EXPECTED_SEEDS=10

usage() {
    printf 'Usage: bash %s [CROSS_DOWNSTREAM_BATCH]\n' "$0" >&2
    printf '\nWith no argument, every batch under %s is checked.\n' "$RESULTS_ROOT" >&2
}

if (( $# > 1 )); then
    usage
    exit 2
fi

shopt -s nullglob
if (( $# == 1 )); then
    batches=("$1")
elif [[ -n ${CROSS_BATCH:-} ]]; then
    batches=("$CROSS_BATCH")
else
    batches=("$RESULTS_ROOT"/batch_*)
fi

if (( ${#batches[@]} == 0 )); then
    printf 'AUDIT FAILED: no cross-downstream batch directories found under %s\n' "$RESULTS_ROOT" >&2
    exit 1
fi

extract_ssl_epochs() {
    sed -nE 's/^[[:space:]]*"ssl_epochs"[[:space:]]*:[[:space:]]*([0-9]+).*/\1/p' "$1" | sed -n '1p'
}

count_seed_files() {
    local directory=$1
    local extension=$2
    local count=0
    local seed path

    for (( seed=0; seed<EXPECTED_SEEDS; seed++ )); do
        printf -v path '%s/seed_%03d.%s' "$directory" "$seed" "$extension"
        [[ -s $path ]] && count=$((count + 1))
    done
    printf '%d\n' "$count"
}

audit_evaluation() {
    local eval_dir=$1
    local job_dir seed_count
    local best_seed_count=0

    for job_dir in "$eval_dir"/job_*; do
        [[ -d $job_dir ]] || continue
        seed_count=$(count_seed_files "$job_dir" json)
        (( seed_count > best_seed_count )) && best_seed_count=$seed_count

        if [[ -s $job_dir/summary.json ]] \
            && grep -Eq '"complete"[[:space:]]*:[[:space:]]*true' "$job_dir/summary.json" \
            && grep -Eq '"completed_runs"[[:space:]]*:[[:space:]]*10([,[:space:]]|$)' "$job_dir/summary.json" \
            && (( seed_count == EXPECTED_SEEDS )); then
            EVAL_SEED_COUNT=$seed_count
            return 0
        fi
    done

    EVAL_SEED_COUNT=$best_seed_count
    return 1
}

audit_batch() {
    local batch=$1
    local source method source_label targets identifier encoder_dir config epochs
    local checkpoint_count eval_count eval_seed_count target label eval_dir
    local set_ok complete_sets=0 total_checkpoints=0 total_evals=0 total_eval_seeds=0

    if [[ ! -d $batch ]]; then
        printf '\nBATCH MISSING: %s\n' "$batch" >&2
        return 1
    fi

    printf '\nBATCH %s\n' "$batch"

    for source in cifar100 tinyimagenet; do
        if [[ $source == cifar100 ]]; then
            source_label=CIFAR100
            targets='TinyImageNet CIFAR10 Aircraft CUB2011 Dogs Flowers'
        else
            source_label=TinyImageNet
            targets='CIFAR10 CIFAR100 Aircraft CUB2011 Dogs Flowers'
        fi

        for method in simclr barlowtwins; do
            identifier="${source}_ResNet18_${method}_full_size100"
            encoder_dir="$batch/encoders/$identifier"
            config="$encoder_dir/config.json"
            set_ok=1
            epochs=missing

            if [[ -s $config ]]; then
                epochs=$(extract_ssl_epochs "$config")
                [[ -n $epochs ]] || epochs=unknown
                if [[ $epochs != "$EXPECTED_SSL_EPOCHS" ]]; then
                    printf '  WRONG   %-58s ssl_epochs=%s (expected %s)\n' \
                        "$identifier" "$epochs" "$EXPECTED_SSL_EPOCHS" >&2
                    set_ok=0
                fi
            else
                printf '  MISSING %-58s config.json\n' "$identifier" >&2
                set_ok=0
            fi

            checkpoint_count=$(count_seed_files "$encoder_dir" pt)
            total_checkpoints=$((total_checkpoints + checkpoint_count))
            if (( checkpoint_count != EXPECTED_SEEDS )); then
                printf '  MISSING %-58s checkpoints=%d/%d\n' \
                    "$identifier" "$checkpoint_count" "$EXPECTED_SEEDS" >&2
                set_ok=0
            fi

            eval_count=0
            eval_seed_count=0
            for target in $targets; do
                for label in 1 5; do
                    eval_dir="$batch/evaluations/${identifier}__${target}_lbl${label}"
                    EVAL_SEED_COUNT=0
                    if audit_evaluation "$eval_dir"; then
                        eval_count=$((eval_count + 1))
                    else
                        printf '  MISSING %-58s evaluation=%s_lbl%s (seeds=%d/%d)\n' \
                            "$identifier" "$target" "$label" \
                            "$EVAL_SEED_COUNT" "$EXPECTED_SEEDS" >&2
                        set_ok=0
                    fi
                    eval_seed_count=$((eval_seed_count + EVAL_SEED_COUNT))
                done
            done
            total_evals=$((total_evals + eval_count))
            total_eval_seeds=$((total_eval_seeds + eval_seed_count))

            if (( set_ok )); then
                complete_sets=$((complete_sets + 1))
                printf '  OK      %-58s checkpoints=%d/10 epochs=%s evaluations=%d/12 seeds=%d/120\n' \
                    "$identifier" "$checkpoint_count" "$epochs" "$eval_count" "$eval_seed_count"
            else
                printf '  PARTIAL %-58s checkpoints=%d/10 epochs=%s evaluations=%d/12 seeds=%d/120\n' \
                    "$identifier" "$checkpoint_count" "$epochs" "$eval_count" "$eval_seed_count"
            fi
        done
    done

    if (( complete_sets == 4 )); then
        printf '  COMPLETE: 4/4 encoder sets, %d/40 checkpoints, %d/48 evaluations, %d/480 evaluation seeds.\n' \
            "$total_checkpoints" "$total_evals" "$total_eval_seeds"
        COMPLETE_BATCHES+=("$batch")
        return 0
    fi

    printf '  INCOMPLETE: %d/4 encoder sets, %d/40 checkpoints, %d/48 evaluations, %d/480 evaluation seeds.\n' \
        "$complete_sets" "$total_checkpoints" "$total_evals" "$total_eval_seeds" >&2
    return 1
}

printf '%s\n' 'Full-data transfer artifact audit'
printf 'Expected: ResNet18, full data, 10 seeds, %s SSL epochs\n' "$EXPECTED_SSL_EPOCHS"
printf 'Repository: %s\n' "$PROJECT_ROOT"

COMPLETE_BATCHES=()
for batch in "${batches[@]}"; do
    audit_batch "$batch" || true
done

printf '\n'
if (( ${#COMPLETE_BATCHES[@]} > 0 )); then
    printf 'AUDIT PASSED: %d complete usable batch(es):\n' "${#COMPLETE_BATCHES[@]}"
    printf '  %s\n' "${COMPLETE_BATCHES[@]}"
    exit 0
fi

printf '%s\n' 'AUDIT FAILED: no batch contains all required 300-epoch checkpoints and evaluations.' >&2
exit 1
