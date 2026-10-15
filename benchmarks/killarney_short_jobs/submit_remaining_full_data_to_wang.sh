#!/bin/bash
set -euo pipefail

# Submit only the remaining non-Ours full-data block:
# CIFAR-10 / ResNet18 / Barlow Twins, 600 SSL epochs, 10 seeds.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
PROJECT_ROOT=$(readlink -f "$SCRIPT_DIR/../..")
BUNDLE="$PROJECT_ROOT/benchmarks/killarney_short_jobs"
WORKER="$BUNDLE/full_data_600_job.sh"
ENVIRONMENT="$BUNDLE/full_data_600_environment.sh"
AUDIT="$BUNDLE/check_remaining_full_data_600.sh"
ACCOUNT=aip-boyuwang

TARGET_SOURCE_BATCH=${TARGET_SOURCE_BATCH:-$PROJECT_ROOT/benchmarks/cross_downstream/results/batch_20261009T171449Z_o91BKP}
RESULTS_ROOT=${REMAINING600_RESULTS_ROOT:-$PROJECT_ROOT/benchmarks/full_data_600_remaining/results}
VENV_ACTIVATE=${VENV_ACTIVATE:-$HOME/ENV/bin/activate}

dry_run=0
if (( $# == 1 )) && [[ $1 == --dry-run ]]; then
    dry_run=1
elif (( $# != 0 )); then
    echo "Usage: bash $0 [--dry-run]" >&2
    exit 2
fi

fail() {
    echo "FATAL: $*" >&2
    exit 1
}

case "$PROJECT_ROOT" in
    /scratch/*|/project/*) ;;
    *) fail "project root is not on scratch/project: $PROJECT_ROOT" ;;
esac

cd "$PROJECT_ROOT"
for path in \
    "$WORKER" \
    "$ENVIRONMENT" \
    "$AUDIT" \
    benchmarks/benchmark_full_data_600.py \
    benchmarks/benchmark_full_refresh.py \
    benchmarks/benchmark_cifar10.py \
    benchmarks/benchmark_dataset.py \
    benchmarks/downstream_data.py \
    OT-SSL-DD/utils.py \
    OT-SSL-DD/networks.py; do
    [[ -s $path ]] || fail "missing required file: $path"
done
grep -Fq 'CIFAR10:ResNet18:barlowtwins' "$WORKER" || \
    fail "worker does not allow CIFAR10 ResNet18 Barlow Twins"
[[ -s $VENV_ACTIVATE ]] || fail "missing Python environment: $VENV_ACTIVATE"

for target in CIFAR10 CIFAR100 TinyImageNet Aircraft CUB2011 Dogs Flowers; do
    [[ -s $TARGET_SOURCE_BATCH/targets/$target.pt ]] || \
        fail "missing prepared target cache: $TARGET_SOURCE_BATCH/targets/$target.pt"
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

    # Refuse accidental duplication of either the previous attempt or this one.
    queued_names=$(squeue -h -u "$USER" -o '%j')
    if grep -Eq '^(kil_full_c10_bt_|k600_c10_r18_b_)' <<< "$queued_names"; then
        printf 'Matching queued jobs:\n' >&2
        grep -E '^(kil_full_c10_bt_|k600_c10_r18_b_)' <<< "$queued_names" >&2
        fail 'CIFAR-10 ResNet18 Barlow-Twins work is already queued; refusing duplicates'
    fi
fi

if (( dry_run )); then
    REDO600_BATCH="$RESULTS_ROOT/<new-batch>"
    REDO600_CODE_ROOT="$REDO600_BATCH/code"
    TARGET_CACHE_ROOT="$REDO600_BATCH/targets"
    RUN_WORKER="$WORKER"
else
    mkdir -p "$RESULTS_ROOT"
    REDO600_BATCH=$(mktemp -d "$RESULTS_ROOT/batch_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")
    REDO600_CODE_ROOT="$REDO600_BATCH/code"
    TARGET_CACHE_ROOT="$REDO600_BATCH/targets"
    RUN_WORKER="$REDO600_CODE_ROOT/benchmarks/killarney_short_jobs/full_data_600_job.sh"

    mkdir -p \
        "$REDO600_BATCH/logs" \
        "$REDO600_BATCH/encoders" \
        "$REDO600_BATCH/evaluations" \
        "$REDO600_CODE_ROOT/benchmarks/killarney_short_jobs" \
        "$REDO600_CODE_ROOT/OT-SSL-DD" \
        "$TARGET_CACHE_ROOT"

    cp \
        benchmarks/benchmark_full_data_600.py \
        benchmarks/benchmark_full_refresh.py \
        benchmarks/benchmark_cifar10.py \
        benchmarks/benchmark_dataset.py \
        benchmarks/downstream_data.py \
        "$REDO600_CODE_ROOT/benchmarks/"
    cp OT-SSL-DD/utils.py OT-SSL-DD/networks.py "$REDO600_CODE_ROOT/OT-SSL-DD/"
    cp "$WORKER" "$ENVIRONMENT" "$AUDIT" \
        "$REDO600_CODE_ROOT/benchmarks/killarney_short_jobs/"

    for target in CIFAR10 CIFAR100 TinyImageNet Aircraft CUB2011 Dogs Flowers; do
        ln -s "$TARGET_SOURCE_BATCH/targets/$target.pt" "$TARGET_CACHE_ROOT/$target.pt"
    done

    printf '%s\n' "$TARGET_SOURCE_BATCH" > "$REDO600_BATCH/target_source_batch.txt"
    printf '%s\n' 'full_data_all_ssl_600epoch_v1' > "$REDO600_BATCH/protocol.txt"
    printf '%s\n' 'CIFAR10 ResNet18 barlowtwins 600 epochs 10 seeds' \
        > "$REDO600_BATCH/configuration.txt"
    printf 'submitted_utc\tjob_id\taccount\ttask\tdependency\n' \
        > "$REDO600_BATCH/submitted.tsv"

    tar -czf "$REDO600_BATCH/source.tar.gz" \
        benchmarks/benchmark_full_data_600.py \
        benchmarks/benchmark_full_refresh.py \
        benchmarks/benchmark_cifar10.py \
        benchmarks/benchmark_dataset.py \
        benchmarks/downstream_data.py \
        benchmarks/killarney_short_jobs/full_data_600_job.sh \
        benchmarks/killarney_short_jobs/full_data_600_environment.sh \
        benchmarks/killarney_short_jobs/submit_remaining_full_data_to_wang.sh \
        benchmarks/killarney_short_jobs/check_remaining_full_data_600.sh \
        OT-SSL-DD/utils.py \
        OT-SSL-DD/networks.py
fi

export PROJECT_ROOT REDO600_BATCH REDO600_CODE_ROOT TARGET_CACHE_ROOT VENV_ACTIVATE

LAST_JOB_ID=
dry_id=997000
submitted_count=0
submit_gpu() {
    local name=$1
    local limit=$2
    local dependency=$3
    shift 3
    local args=(sbatch --parsable
        --account="$ACCOUNT"
        --job-name="$name"
        --nodes=1
        --ntasks=1
        --cpus-per-task=1
        --gres=gpu:l40s:1
        --time="$limit"
        --mem=8G
        --chdir="$PROJECT_ROOT"
        --output="$REDO600_BATCH/logs/%x-%j.out")
    if [[ -n $dependency ]]; then
        args+=(--dependency="$dependency" --kill-on-invalid-dep=yes)
    fi

    local submitted task
    task="$*"
    if (( dry_run )); then
        printf 'DRY RUN: '
        printf '%q ' "${args[@]}" "$RUN_WORKER" "$@"
        printf '\n'
        dry_id=$((dry_id + 1))
        LAST_JOB_ID=$dry_id
    else
        submitted=$("${args[@]}" "$RUN_WORKER" "$@")
        LAST_JOB_ID=${submitted%%;*}
        printf '%s\t%s\t%s\t%s\t%s\n' \
            "$(date -u +%FT%TZ)" "$LAST_JOB_ID" "$ACCOUNT" "$task" "$dependency" \
            >> "$REDO600_BATCH/submitted.tsv"
        printf 'Submitted %-24s as %s on %s\n' "$name" "$LAST_JOB_ID" "$ACCOUNT"
    fi
    submitted_count=$((submitted_count + 1))
}

# One seed per job keeps every training allocation safely below its walltime.
pretrain_ids=()
for seed in {0..9}; do
    submit_gpu "k600_c10_r18_b_s${seed}" 03:30:00 '' \
        pretrain CIFAR10 ResNet18 barlowtwins "$seed" 1
    pretrain_ids+=("$LAST_JOB_ID")
done

dependency_ids=$(IFS=:; echo "${pretrain_ids[*]}")
submit_gpu k600_c10_r18_b_eval 03:00:00 "afterok:$dependency_ids" \
    evaluate CIFAR10 ResNet18 barlowtwins

printf '\nSubmission plan complete.\n'
printf 'Account: %s for all %d GPU jobs\n' "$ACCOUNT" "$submitted_count"
printf '%s\n' 'Coverage: all 14 remaining non-Ours full-data cells'
printf '%s\n' 'Training: CIFAR10 / ResNet18 / Barlow Twins / 600 epochs / seeds 0-9'
printf 'Results: %s\n' "$REDO600_BATCH"
printf '%s\n' 'CPU Slurm jobs: 0'
if (( dry_run )); then
    printf '%s\n' 'Dry run only: nothing was created or submitted.'
else
    printf 'After completion, audit with:\n  bash %q %q\n' "$AUDIT" "$REDO600_BATCH"
fi
