#!/bin/bash
set -euo pipefail

# Immediate second-round recovery for the ECC failures from retry jobs
# 6313701-6313717. Only seeds whose logs showed terminal ECC failures are
# candidates here; the seven still-running seeds are deliberately excluded.
# New evaluator jobs wait for both this retry and the corresponding old array,
# so training can be resubmitted now without duplicating active work.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
PROJECT_ROOT=$(readlink -f "$SCRIPT_DIR/../..")
RESULTS_ROOT=${MAINLONG_RESULTS_ROOT:-$PROJECT_ROOT/benchmarks/full_data_main_long/results}
ACCOUNT=${ACCOUNT:-aip-yiweilu}
VENV_ACTIVATE=${VENV_ACTIVATE:-$HOME/ENV/bin/activate}
EXCLUDED_NODES=kn072,kn117,kn140,kn161
EXPECTED_PROTOCOL=full_data_main_simclr1000_barlow1500_v1

dry_run=0
batch_arg=
for arg in "$@"; do
    case "$arg" in
        --dry-run) dry_run=1 ;;
        -*) echo "Usage: bash $0 [--dry-run] [MAINLONG_BATCH]" >&2; exit 2 ;;
        *)
            [[ -z $batch_arg ]] || {
                echo "Usage: bash $0 [--dry-run] [MAINLONG_BATCH]" >&2
                exit 2
            }
            batch_arg=$arg
            ;;
    esac
done

fail() {
    echo "FATAL: $*" >&2
    exit 1
}

case "$PROJECT_ROOT" in
    /scratch/*|/project/*) ;;
    *) fail "project root is not on scratch/project: $PROJECT_ROOT" ;;
esac

shopt -s nullglob
if [[ -n $batch_arg ]]; then
    MAINLONG_BATCH=$(readlink -f "$batch_arg")
else
    batches=("$RESULTS_ROOT"/batch_*)
    (( ${#batches[@]} > 0 )) || fail "no batches found under $RESULTS_ROOT"
    MAINLONG_BATCH=${batches[0]}
    for candidate in "${batches[@]}"; do MAINLONG_BATCH=$candidate; done
fi

[[ -d $MAINLONG_BATCH ]] || fail "missing batch: $MAINLONG_BATCH"
[[ -s $MAINLONG_BATCH/protocol.txt ]] || fail "missing batch protocol"
[[ $(<"$MAINLONG_BATCH/protocol.txt") == "$EXPECTED_PROTOCOL" ]] || \
    fail "unexpected protocol in $MAINLONG_BATCH/protocol.txt"

MAINLONG_CODE_ROOT="$MAINLONG_BATCH/code"
WORKER="$MAINLONG_CODE_ROOT/benchmarks/killarney_short_jobs/full_data_main_long_job.sh"
AUDIT="$PROJECT_ROOT/benchmarks/killarney_short_jobs/check_full_data_main_long.sh"
MANIFEST="$MAINLONG_BATCH/retry_round2_submitted.tsv"
[[ -s $WORKER ]] || fail "missing immutable batch worker: $WORKER"
[[ -s $VENV_ACTIVATE ]] || fail "missing Python environment: $VENV_ACTIVATE"

if [[ -s $MAINLONG_BATCH/targets/CIFAR10.pt && -s $MAINLONG_BATCH/targets/CIFAR100.pt ]]; then
    TARGET_CACHE_ROOT="$MAINLONG_BATCH/targets"
elif [[ -s $MAINLONG_BATCH/target_source_batch.txt ]]; then
    target_source_batch=$(<"$MAINLONG_BATCH/target_source_batch.txt")
    TARGET_CACHE_ROOT="$target_source_batch/targets"
else
    fail "cannot locate target caches for $MAINLONG_BATCH"
fi
for target in CIFAR10 CIFAR100; do
    [[ -s $TARGET_CACHE_ROOT/$target.pt ]] || \
        fail "missing prepared target cache: $TARGET_CACHE_ROOT/$target.pt"
done

if (( ! dry_run )); then
    command -v sbatch >/dev/null 2>&1 || fail "sbatch is not available"
    command -v sinfo >/dev/null 2>&1 || fail "sinfo is not available"
    command -v squeue >/dev/null 2>&1 || fail "squeue is not available"

    gpu_inventory=$(sinfo -h -o '%G' --Node | sort -u)
    grep -q 'gpu:l40s:' <<< "$gpu_inventory" || {
        printf 'Discovered GPU inventory:\n%s\n' "$gpu_inventory" >&2
        fail 'gpu:l40s is not present in the discovered Killarney inventory'
    }

    queued_names=$(squeue -h -u "$USER" -o '%j')
    if grep -Eq '^kml2_' <<< "$queued_names"; then
        printf 'Matching queued second-round jobs:\n' >&2
        grep -E '^kml2_' <<< "$queued_names" >&2
        fail 'second-round recovery is already queued; refusing duplicate submission'
    fi

    if [[ ! -e $MANIFEST ]]; then
        printf 'submitted_utc\tjob_id\taccount\tkind\tsource\tmodel\tssl_method\tseeds\tdependency\texcluded_nodes\n' \
            > "$MANIFEST"
    fi
fi

export PROJECT_ROOT MAINLONG_BATCH MAINLONG_CODE_ROOT TARGET_CACHE_ROOT VENV_ACTIVATE

valid_evaluation_count() {
    local source=$1
    local model=$2
    local ssl=$3
    local epochs=$4
    local label=$5
    local probe=$6
    local prefix identifier eval_dir summary
    local valid=0
    local seed_files=()

    if [[ $source == CIFAR10 ]]; then prefix=cifar10; else prefix=cifar100; fi
    identifier="${prefix}_${model}_${ssl}_full_size100_${epochs}ep"
    eval_dir="$MAINLONG_BATCH/evaluations/${identifier}__${source}_lbl${label}"
    for summary in "$eval_dir"/job_*/summary.json; do
        [[ -s $summary ]] || continue
        grep -Eq '"protocol"[[:space:]]*:[[:space:]]*"'"$EXPECTED_PROTOCOL"'"' "$summary" || continue
        grep -Eq '"source"[[:space:]]*:[[:space:]]*"'"$source"'"' "$summary" || continue
        grep -Eq '"target"[[:space:]]*:[[:space:]]*"'"$source"'"' "$summary" || continue
        grep -Eq '"model"[[:space:]]*:[[:space:]]*"'"$model"'"' "$summary" || continue
        grep -Eq '"ssl_method"[[:space:]]*:[[:space:]]*"'"$ssl"'"' "$summary" || continue
        grep -Eq '"ssl_epochs"[[:space:]]*:[[:space:]]*'"$epochs"'([,[:space:]]|$)' "$summary" || continue
        grep -Eq '"label_percentage"[[:space:]]*:[[:space:]]*'"$label"'([,[:space:]]|$)' "$summary" || continue
        grep -Eq '"probe_epochs"[[:space:]]*:[[:space:]]*'"$probe"'([,[:space:]]|$)' "$summary" || continue
        grep -Eq '"complete"[[:space:]]*:[[:space:]]*true' "$summary" || continue
        grep -Eq '"completed_runs"[[:space:]]*:[[:space:]]*10([,[:space:]]|$)' "$summary" || continue
        grep -Eq '"expected_runs"[[:space:]]*:[[:space:]]*10([,[:space:]]|$)' "$summary" || continue
        seed_files=("${summary%/summary.json}"/seed_*.json)
        (( ${#seed_files[@]} == 10 )) || continue
        valid=$((valid + 1))
    done
    printf '%d\n' "$valid"
}

still_missing_failed_seeds() {
    local source=$1
    local model=$2
    local ssl=$3
    local epochs=$4
    local candidates=$5
    local prefix identifier encoder_root seed checkpoint
    local selected=()
    local matches=()
    local candidate_seeds=()

    [[ $candidates != - ]] || return 0
    if [[ $source == CIFAR10 ]]; then prefix=cifar10; else prefix=cifar100; fi
    identifier="${prefix}_${model}_${ssl}_full_size100_${epochs}ep"
    encoder_root="$MAINLONG_BATCH/encoders/$identifier"
    IFS=, read -r -a candidate_seeds <<< "$candidates"
    for seed in "${candidate_seeds[@]}"; do
        printf -v checkpoint 'seed_%03d.pt' "$seed"
        matches=("$encoder_root"/shard_*/"$checkpoint")
        if (( ${#matches[@]} == 0 )); then
            selected+=("$seed")
        elif (( ${#matches[@]} == 1 )) && [[ -s ${matches[0]} ]]; then
            printf 'SKIP recovered checkpoint: %s seed=%s\n' "$identifier" "$seed" >&2
        else
            fail "$identifier seed $seed has ${#matches[@]} ambiguous/invalid checkpoints"
        fi
    done
    if (( ${#selected[@]} > 0 )); then
        (IFS=,; printf '%s' "${selected[*]}")
    fi
}

LAST_JOB_ID=
dry_id=994000
submitted_records=0
submitted_tasks=0

submit_pretrain_array() {
    local name=$1
    local limit=$2
    local seed_spec=$3
    shift 3
    local source=$1 model=$2 ssl=$3
    local args=(sbatch --parsable
        --account="$ACCOUNT"
        --job-name="$name"
        --array="$seed_spec"
        --nodes=1
        --ntasks=1
        --cpus-per-task=1
        --gres=gpu:l40s:1
        --exclude="$EXCLUDED_NODES"
        --time="$limit"
        --mem=8G
        --chdir="$PROJECT_ROOT"
        --output="$MAINLONG_BATCH/logs/%x-%A_%a.out")
    local submitted seed_count

    if (( dry_run )); then
        printf 'DRY RUN: '
        printf '%q ' "${args[@]}" "$WORKER" pretrain "$source" "$model" "$ssl"
        printf '\n'
        dry_id=$((dry_id + 1))
        LAST_JOB_ID=$dry_id
    else
        submitted=$("${args[@]}" "$WORKER" pretrain "$source" "$model" "$ssl")
        LAST_JOB_ID=${submitted%%;*}
        printf '%s\t%s\t%s\tpretrain\t%s\t%s\t%s\t%s\t\t%s\n' \
            "$(date -u +%FT%TZ)" "$LAST_JOB_ID" "$ACCOUNT" "$source" \
            "$model" "$ssl" "$seed_spec" "$EXCLUDED_NODES" >> "$MANIFEST"
        printf 'Submitted %-22s as array %s (seeds %s)\n' \
            "$name" "$LAST_JOB_ID" "$seed_spec"
    fi
    seed_count=$(awk -F, '{print NF}' <<< "$seed_spec")
    submitted_records=$((submitted_records + 1))
    submitted_tasks=$((submitted_tasks + seed_count))
}

submit_evaluation() {
    local name=$1
    local dependency=$2
    shift 2
    local source=$1 model=$2 ssl=$3
    local args=(sbatch --parsable
        --account="$ACCOUNT"
        --job-name="$name"
        --nodes=1
        --ntasks=1
        --cpus-per-task=1
        --gres=gpu:l40s:1
        --exclude="$EXCLUDED_NODES"
        --time=01:30:00
        --mem=8G
        --chdir="$PROJECT_ROOT"
        --output="$MAINLONG_BATCH/logs/%x-%j.out")
    local submitted
    if [[ -n $dependency ]]; then
        args+=(--dependency="$dependency" --kill-on-invalid-dep=yes)
    fi

    if (( dry_run )); then
        printf 'DRY RUN: '
        printf '%q ' "${args[@]}" "$WORKER" evaluate "$source" "$model" "$ssl"
        printf '\n'
        dry_id=$((dry_id + 1))
        LAST_JOB_ID=$dry_id
    else
        submitted=$("${args[@]}" "$WORKER" evaluate "$source" "$model" "$ssl")
        LAST_JOB_ID=${submitted%%;*}
        printf '%s\t%s\t%s\tevaluate\t%s\t%s\t%s\t-\t%s\t%s\n' \
            "$(date -u +%FT%TZ)" "$LAST_JOB_ID" "$ACCOUNT" "$source" \
            "$model" "$ssl" "${dependency:--}" "$EXCLUDED_NODES" >> "$MANIFEST"
        printf 'Submitted %-22s as %s' "$name" "$LAST_JOB_ID"
        [[ -z $dependency ]] || printf ' with dependency %s' "$dependency"
        printf '\n'
    fi
    submitted_records=$((submitted_records + 1))
    submitted_tasks=$((submitted_tasks + 1))
}

# tag|source|model|ssl|epochs|limit|terminal-ECC seeds|old array still running
configurations=(
    'c10_cn_s|CIFAR10|ConvNet|simclr|1000|02:00:00|6,8,9|6313701'
    'c10_cn_b|CIFAR10|ConvNet|barlowtwins|1500|03:00:00|0,1,2,4,5,9|6313703'
    'c10_vg_s|CIFAR10|VGG11|simclr|1000|02:30:00|2,3,4,6,7,9|6313705'
    'c10_vg_b|CIFAR10|VGG11|barlowtwins|1500|03:30:00|-|-'
    'c100_cn_s|CIFAR100|ConvNet|simclr|1000|02:00:00|0,2,3,4,5,6,7,8|6313708'
    'c100_cn_b|CIFAR100|ConvNet|barlowtwins|1500|03:00:00|0,1,2,4,6,7,9|6313710'
    'c100_vg_s|CIFAR100|VGG11|simclr|1000|02:30:00|0,1,3,4,5,7|-'
    'c100_vg_b|CIFAR100|VGG11|barlowtwins|1500|03:30:00|9|-'
    'c100_r18_s|CIFAR100|ResNet18|simclr|1000|05:30:00|0,1,2|-'
)

for spec in "${configurations[@]}"; do
    IFS='|' read -r tag source model ssl epochs limit failed_seeds wait_parent <<< "$spec"
    label1_count=$(valid_evaluation_count "$source" "$model" "$ssl" "$epochs" 1 500)
    label5_count=$(valid_evaluation_count "$source" "$model" "$ssl" "$epochs" 5 300)
    (( label1_count <= 1 && label5_count <= 1 )) || \
        fail "$tag has duplicate complete evaluations (1%=$label1_count, 5%=$label5_count)"
    if (( label1_count == 1 && label5_count == 1 )); then
        printf 'SKIP complete evaluation: %s\n' "$tag"
        continue
    fi
    (( label1_count == 0 && label5_count == 0 )) || \
        fail "$tag has only one complete label setting; refusing to create a duplicate"

    seed_spec=$(still_missing_failed_seeds \
        "$source" "$model" "$ssl" "$epochs" "$failed_seeds")
    new_array_id=
    if [[ -n $seed_spec ]]; then
        submit_pretrain_array "kml2_${tag}_pre" "$limit" "$seed_spec" \
            "$source" "$model" "$ssl"
        new_array_id=$LAST_JOB_ID
    fi

    dependency_parts=()
    [[ -z $new_array_id ]] || dependency_parts+=("afterok:$new_array_id")
    [[ $wait_parent == - ]] || dependency_parts+=("afterany:$wait_parent")
    dependency=
    if (( ${#dependency_parts[@]} > 0 )); then
        dependency=$(IFS=,; printf '%s' "${dependency_parts[*]}")
    fi
    submit_evaluation "kml2_${tag}_eval" "$dependency" \
        "$source" "$model" "$ssl"
done

printf '\nImmediate second-round recovery plan complete.\n'
printf 'Batch: %s\n' "$MAINLONG_BATCH"
printf 'Account: %s\n' "$ACCOUNT"
printf 'Excluded nodes: %s\n' "$EXCLUDED_NODES"
printf 'Submitted/planned Slurm records: %d\n' "$submitted_records"
printf 'Submitted/planned GPU tasks: %d\n' "$submitted_tasks"
printf '%s\n' 'The seven currently running first-round seeds were not resubmitted.'
if (( dry_run )); then
    printf '%s\n' 'Dry run only: nothing was submitted or changed.'
else
    printf 'Submission record: %s\n' "$MANIFEST"
    printf 'After completion, audit with:\n  bash %q %q\n' "$AUDIT" "$MAINLONG_BATCH"
fi
