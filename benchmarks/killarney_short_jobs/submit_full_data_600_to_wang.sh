#!/bin/bash
set -euo pipefail

# Create a fresh 600-epoch full-data batch and submit its GPU-only dependency
# graph. Nothing is reused except the immutable prepared dataset caches.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
PROJECT_ROOT=$(readlink -f "$SCRIPT_DIR/../..")
BUNDLE="$PROJECT_ROOT/benchmarks/killarney_short_jobs"
WORKER="$BUNDLE/full_data_600_job.sh"
ENVIRONMENT="$BUNDLE/full_data_600_environment.sh"
AUDIT="$BUNDLE/check_full_data_600.sh"
ACCOUNT=aip-boyuwang

TARGET_SOURCE_BATCH=${TARGET_SOURCE_BATCH:-$PROJECT_ROOT/benchmarks/cross_downstream/results/batch_20261009T171449Z_o91BKP}
RESULTS_ROOT=${REDO600_RESULTS_ROOT:-$PROJECT_ROOT/benchmarks/full_data_600/results}
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
[[ -s $VENV_ACTIVATE ]] || fail "missing Python environment: $VENV_ACTIVATE"

for target in CIFAR10 CIFAR100 TinyImageNet Aircraft CUB2011 Dogs Flowers; do
    [[ -s $TARGET_SOURCE_BATCH/targets/$target.pt ]] || \
        fail "missing prepared target cache: $TARGET_SOURCE_BATCH/targets/$target.pt"
done

if (( ! dry_run )); then
    command -v sbatch >/dev/null 2>&1 || fail "sbatch is not available"
    command -v sinfo >/dev/null 2>&1 || fail "sinfo is not available"
    gpu_inventory=$(sinfo -h -o '%G' --Node | sort -u)
    grep -q 'gpu:l40s:' <<< "$gpu_inventory" || {
        printf 'Discovered GPU inventory:\n%s\n' "$gpu_inventory" >&2
        fail 'gpu:l40s is not present in the discovered Killarney inventory'
    }
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
    cp "$WORKER" "$ENVIRONMENT" "$REDO600_CODE_ROOT/benchmarks/killarney_short_jobs/"

    for target in CIFAR10 CIFAR100 TinyImageNet Aircraft CUB2011 Dogs Flowers; do
        ln -s "$TARGET_SOURCE_BATCH/targets/$target.pt" "$TARGET_CACHE_ROOT/$target.pt"
    done

    printf '%s\n' "$TARGET_SOURCE_BATCH" > "$REDO600_BATCH/target_source_batch.txt"
    printf '%s\n' 'full_data_all_ssl_600epoch_v1' > "$REDO600_BATCH/protocol.txt"
    printf '%s\n' 'source model ssl_method pretrain_time' > "$REDO600_BATCH/configurations.txt"
    printf '%s\n' \
        'CIFAR10 ConvNet simclr 04:00:00' \
        'CIFAR10 VGG11 simclr 04:30:00' \
        'CIFAR10 ResNet18 simclr 13:00:00' \
        'CIFAR100 ResNet18 simclr 13:00:00' \
        'CIFAR100 ResNet18 barlowtwins 13:00:00' \
        'TinyImageNet ResNet18 simclr 23:30:00' \
        'TinyImageNet ResNet18 barlowtwins 23:30:00' \
        >> "$REDO600_BATCH/configurations.txt"
    printf 'submitted_utc\tjob_id\taccount\ttask\tdependency\n' > "$REDO600_BATCH/submitted.tsv"

    tar -czf "$REDO600_BATCH/source.tar.gz" \
        benchmarks/benchmark_full_data_600.py \
        benchmarks/benchmark_full_refresh.py \
        benchmarks/benchmark_cifar10.py \
        benchmarks/benchmark_dataset.py \
        benchmarks/downstream_data.py \
        benchmarks/killarney_short_jobs/full_data_600_job.sh \
        benchmarks/killarney_short_jobs/full_data_600_environment.sh \
        benchmarks/killarney_short_jobs/submit_full_data_600_to_wang.sh \
        benchmarks/killarney_short_jobs/check_full_data_600.sh \
        OT-SSL-DD/utils.py \
        OT-SSL-DD/networks.py
fi

export PROJECT_ROOT REDO600_BATCH REDO600_CODE_ROOT TARGET_CACHE_ROOT VENV_ACTIVATE

LAST_JOB_ID=
dry_id=996000
submitted_count=0
submit_gpu() {
    local name=$1
    local limit=$2
    local memory=$3
    local dependency=$4
    shift 4
    local args=(sbatch --parsable
        --account="$ACCOUNT"
        --job-name="$name"
        --nodes=1
        --ntasks=1
        --cpus-per-task=1
        --gres=gpu:l40s:1
        --time="$limit"
        --mem="$memory"
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
        printf 'Submitted %-22s as %s on %s\n' "$name" "$LAST_JOB_ID" "$ACCOUNT"
    fi
    submitted_count=$((submitted_count + 1))
}

submit_configuration() {
    local tag=$1
    local source_dataset=$2
    local model=$3
    local ssl_method=$4
    local pretrain_limit=$5
    local ids=()
    local seed_start dependency_ids

    for seed_start in 0 5; do
        submit_gpu "k600_${tag}_p${seed_start}" "$pretrain_limit" 8G '' \
            pretrain "$source_dataset" "$model" "$ssl_method" "$seed_start" 5
        ids+=("$LAST_JOB_ID")
    done
    dependency_ids=$(IFS=:; echo "${ids[*]}")
    submit_gpu "k600_${tag}_eval" 03:00:00 8G "afterok:$dependency_ids" \
        evaluate "$source_dataset" "$model" "$ssl_method"
}

submit_configuration c10_cn_s CIFAR10 ConvNet simclr 04:00:00
submit_configuration c10_vg_s CIFAR10 VGG11 simclr 04:30:00
submit_configuration c10_r18_s CIFAR10 ResNet18 simclr 13:00:00
submit_configuration c100_r18_s CIFAR100 ResNet18 simclr 13:00:00
submit_configuration c100_r18_b CIFAR100 ResNet18 barlowtwins 13:00:00
submit_configuration tiny_r18_s TinyImageNet ResNet18 simclr 23:30:00
submit_configuration tiny_r18_b TinyImageNet ResNet18 barlowtwins 23:30:00

printf '\nSubmission plan complete.\n'
printf 'Account: %s for all %d GPU jobs\n' "$ACCOUNT" "$submitted_count"
printf 'Fresh training: 7 configurations x 10 seeds x 600 epochs\n'
printf 'Evaluation: 7 configurations x 7 targets x 2 label settings\n'
printf 'Results: %s\n' "$REDO600_BATCH"
printf '%s\n' 'CPU Slurm jobs: 0'
if (( dry_run )); then
    printf '%s\n' 'Dry run only: nothing was created or submitted.'
else
    printf 'After all jobs finish, audit with:\n  bash %q %q\n' "$AUDIT" "$REDO600_BATCH"
fi
