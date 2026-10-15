#!/bin/bash
set -euo pipefail

# Targeted recovery for the five CIFAR-10 / ResNet18 / Barlow-Twins seeds
# that failed with uncorrectable ECC errors in batch_20261015T140715Z_DIz35R.
#
# Run this file normally to submit. Slurm invokes the same file with the
# private --worker-pretrain mode for each replacement training job.

worker_pretrain() {
    (( $# == 1 )) || {
        echo "Usage: $0 --worker-pretrain SEED" >&2
        exit 2
    }

    : "${SLURM_JOB_ID:?This worker must run through sbatch.}"
    : "${PROJECT_ROOT:?}"
    : "${REDO600_BATCH:?}"
    : "${REDO600_CODE_ROOT:?}"
    : "${TARGET_CACHE_ROOT:?}"

    local seed=$1
    [[ $seed =~ ^(0|1|4|6|9)$ ]] || {
        echo "Refusing unexpected retry seed: $seed" >&2
        exit 2
    }

    source "$REDO600_CODE_ROOT/benchmarks/killarney_short_jobs/full_data_600_environment.sh"

    local identifier=cifar10_ResNet18_barlowtwins_full_size100_600ep
    local encoder_root="$REDO600_BATCH/encoders/$identifier"
    local checkpoint
    printf -v checkpoint 'seed_%03d.pt' "$seed"

    shopt -s nullglob
    local existing=("$encoder_root"/shard_*/"$checkpoint")
    if (( ${#existing[@]} == 1 )) && [[ -s ${existing[0]} ]]; then
        printf 'Checkpoint already exists for seed %s: %s\n' "$seed" "${existing[0]}"
        exit 0
    fi
    (( ${#existing[@]} == 0 )) || {
        echo "Expected zero valid checkpoints before retrying seed $seed; found ${#existing[@]}" >&2
        printf '  %s\n' "${existing[@]}" >&2
        exit 1
    }

    local output="$encoder_root/shard_${seed}_${seed}_retry_${SLURM_JOB_ID}"
    [[ ! -e $output ]] || {
        echo "Refusing to overwrite retry output: $output" >&2
        exit 1
    }
    [[ -s $TARGET_CACHE_ROOT/CIFAR10.pt ]] || {
        echo "Missing source cache: $TARGET_CACHE_ROOT/CIFAR10.pt" >&2
        exit 1
    }

    mkdir -p "$encoder_root"
    cd "$REDO600_CODE_ROOT"
    srun --unbuffered python3 benchmarks/benchmark_full_data_600.py pretrain \
        --source CIFAR10 \
        --source-cache "$TARGET_CACHE_ROOT/CIFAR10.pt" \
        --output "$output" \
        --model ResNet18 \
        --ssl-method barlowtwins \
        --ssl-epochs 600 \
        --seed-start "$seed" \
        --runs 1 \
        --ssl-log-every 100
}

if [[ ${1:-} == --worker-pretrain ]]; then
    shift
    worker_pretrain "$@"
    exit 0
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
PROJECT_ROOT=$(readlink -f "$SCRIPT_DIR/../..")
ACCOUNT=aip-boyuwang
FAILED_SEEDS=(0 1 4 6 9)
ORIGINAL_GOOD_JOB_IDS=(6268965 6268966 6268968 6268970 6268971)
EXCLUDED_NODES=kn051,kn117

REDO600_BATCH=${REMAINING600_BATCH:-$PROJECT_ROOT/benchmarks/full_data_600_remaining/results/batch_20261015T140715Z_DIz35R}
REDO600_CODE_ROOT="$REDO600_BATCH/code"
VENV_ACTIVATE=${VENV_ACTIVATE:-$HOME/ENV/bin/activate}
EVALUATION_WORKER="$REDO600_CODE_ROOT/benchmarks/killarney_short_jobs/full_data_600_job.sh"
RUN_SCRIPT="$REDO600_CODE_ROOT/benchmarks/killarney_short_jobs/$(basename "$0")"
MANIFEST="$REDO600_BATCH/retry_failed_submission.tsv"

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

[[ -d $REDO600_BATCH ]] || fail "missing original batch: $REDO600_BATCH"
[[ -s $REDO600_BATCH/protocol.txt ]] || fail "missing batch protocol"
[[ $(<"$REDO600_BATCH/protocol.txt") == full_data_all_ssl_600epoch_v1 ]] || \
    fail "unexpected protocol in $REDO600_BATCH/protocol.txt"
[[ -s $EVALUATION_WORKER ]] || fail "missing evaluation worker: $EVALUATION_WORKER"
[[ -s $VENV_ACTIVATE ]] || fail "missing Python environment: $VENV_ACTIVATE"

if [[ -d $REDO600_BATCH/targets ]]; then
    TARGET_CACHE_ROOT="$REDO600_BATCH/targets"
elif [[ -s $REDO600_BATCH/target_source_batch.txt ]]; then
    target_source_batch=$(<"$REDO600_BATCH/target_source_batch.txt")
    TARGET_CACHE_ROOT="$target_source_batch/targets"
else
    fail "cannot locate target caches for $REDO600_BATCH"
fi

for target in CIFAR10 CIFAR100 TinyImageNet Aircraft CUB2011 Dogs Flowers; do
    [[ -s $TARGET_CACHE_ROOT/$target.pt ]] || \
        fail "missing prepared target cache: $TARGET_CACHE_ROOT/$target.pt"
done

if (( ! dry_run )); then
    command -v sbatch >/dev/null 2>&1 || fail "sbatch is not available"
    command -v sinfo >/dev/null 2>&1 || fail "sinfo is not available"

    gpu_inventory=$(sinfo -h -o '%G' --Node | sort -u)
    grep -q 'gpu:l40s:' <<< "$gpu_inventory" || {
        printf 'Discovered GPU inventory:\n%s\n' "$gpu_inventory" >&2
        fail 'gpu:l40s is not present in the discovered Killarney inventory'
    }

    cp "$0" "$RUN_SCRIPT"
    chmod u+x "$RUN_SCRIPT"

    if [[ ! -e $MANIFEST ]]; then
        printf 'submitted_utc\tjob_id\taccount\tkind\tseed\tdependency\n' > "$MANIFEST"
    fi
fi

export PROJECT_ROOT REDO600_BATCH REDO600_CODE_ROOT TARGET_CACHE_ROOT VENV_ACTIVATE

LAST_JOB_ID=
dry_id=998000
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
        --exclude="$EXCLUDED_NODES"
        --time="$limit"
        --mem="$memory"
        --chdir="$PROJECT_ROOT"
        --output="$REDO600_BATCH/logs/%x-%j.out")
    if [[ -n $dependency ]]; then
        args+=(--dependency="$dependency" --kill-on-invalid-dep=yes)
    fi

    local submitted
    if (( dry_run )); then
        printf 'DRY RUN: '
        printf '%q ' "${args[@]}" "$@"
        printf '\n'
        dry_id=$((dry_id + 1))
        LAST_JOB_ID=$dry_id
    else
        submitted=$("${args[@]}" "$@")
        LAST_JOB_ID=${submitted%%;*}
        printf 'Submitted %-24s as %s on %s (excluding %s)\n' \
            "$name" "$LAST_JOB_ID" "$ACCOUNT" "$EXCLUDED_NODES"
    fi
}

manifest_job_id() {
    local kind=$1
    local seed=$2
    [[ -s $MANIFEST ]] || return 0
    awk -F '\t' -v kind="$kind" -v seed="$seed" \
        '$4 == kind && $5 == seed { id=$2 } END { print id }' "$MANIFEST"
}

retry_ids=()
for seed in "${FAILED_SEEDS[@]}"; do
    existing_id=
    if (( ! dry_run )); then
        existing_id=$(manifest_job_id pretrain "$seed")
    fi

    if [[ -n $existing_id ]]; then
        LAST_JOB_ID=$existing_id
        printf 'Reusing recorded retry for seed %s: job %s\n' "$seed" "$LAST_JOB_ID"
    else
        submit_gpu "k600_c10_bt_retry_s${seed}" 03:30:00 8G '' \
            "$RUN_SCRIPT" --worker-pretrain "$seed"
        if (( ! dry_run )); then
            printf '%s\t%s\t%s\tpretrain\t%s\t\n' \
                "$(date -u +%FT%TZ)" "$LAST_JOB_ID" "$ACCOUNT" "$seed" >> "$MANIFEST"
        fi
    fi
    retry_ids+=("$LAST_JOB_ID")
done

all_pretrain_ids=("${ORIGINAL_GOOD_JOB_IDS[@]}" "${retry_ids[@]}")
dependency_ids=$(IFS=:; echo "${all_pretrain_ids[*]}")
dependency="afterok:$dependency_ids"

existing_eval_id=
if (( ! dry_run )); then
    existing_eval_id=$(manifest_job_id evaluate -)
fi
if [[ -n $existing_eval_id ]]; then
    printf 'Replacement evaluation is already recorded as job %s\n' "$existing_eval_id"
else
    submit_gpu k600_c10_bt_retry_eval 03:00:00 8G "$dependency" \
        "$EVALUATION_WORKER" evaluate CIFAR10 ResNet18 barlowtwins
    if (( ! dry_run )); then
        printf '%s\t%s\t%s\tevaluate\t-\t%s\n' \
            "$(date -u +%FT%TZ)" "$LAST_JOB_ID" "$ACCOUNT" "$dependency" >> "$MANIFEST"
    fi
fi

printf '\nTargeted retry submission complete.\n'
printf 'Retried seeds only: %s\n' "${FAILED_SEEDS[*]}"
printf 'Original healthy seed jobs reused: %s\n' "${ORIGINAL_GOOD_JOB_IDS[*]}"
printf 'Excluded nodes: %s\n' "$EXCLUDED_NODES"
printf 'Batch: %s\n' "$REDO600_BATCH"
if (( dry_run )); then
    printf '%s\n' 'Dry run only: nothing was created or submitted.'
else
    printf 'Submission record: %s\n' "$MANIFEST"
    printf 'After completion, audit with:\n  bash %q %q\n' \
        "$PROJECT_ROOT/benchmarks/killarney_short_jobs/check_remaining_full_data_600.sh" \
        "$REDO600_BATCH"
fi
