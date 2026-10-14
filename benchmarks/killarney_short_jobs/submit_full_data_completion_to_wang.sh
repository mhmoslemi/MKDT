#!/bin/bash
set -euo pipefail

# Submit only the GPU work still needed to fill non-Ours table cells. Existing
# complete cross-dataset evaluations are reused rather than duplicated.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
PROJECT_ROOT=$(readlink -f "$SCRIPT_DIR/../..")
BUNDLE="$PROJECT_ROOT/benchmarks/killarney_short_jobs"
WORKER="$BUNDLE/full_data_completion_job.sh"
ENVIRONMENT="$BUNDLE/full_data_completion_environment.sh"
AUDIT="$BUNDLE/check_full_data_transfer.sh"
ACCOUNT=aip-boyuwang

CROSS_BATCH=${CROSS_BATCH:-$PROJECT_ROOT/benchmarks/cross_downstream/results/batch_20261009T171449Z_o91BKP}
FULL_REFRESH_BATCH=${FULL_REFRESH_BATCH:-$PROJECT_ROOT/benchmarks/full_data_refresh/results/batch_20261010T160040Z_73NiHv}
RESULTS_ROOT=${COMPLETION_RESULTS_ROOT:-$PROJECT_ROOT/benchmarks/full_data_completion/results}
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
[[ -s $WORKER ]] || fail "missing worker: $WORKER"
[[ -s $ENVIRONMENT ]] || fail "missing environment: $ENVIRONMENT"
[[ -s $AUDIT ]] || fail "missing audit: $AUDIT"
[[ -s $VENV_ACTIVATE ]] || fail "missing Python environment: $VENV_ACTIVATE"
for path in \
    benchmarks/benchmark_cifar10.py \
    benchmarks/benchmark_dataset.py \
    benchmarks/benchmark_full_refresh.py \
    benchmarks/downstream_data.py \
    OT-SSL-DD/utils.py \
    OT-SSL-DD/networks.py; do
    [[ -s $path ]] || fail "missing code file: $path"
done
for path in \
    "$CROSS_BATCH/code/benchmarks/benchmark_cross_dataset.py" \
    "$CROSS_BATCH/code/benchmarks/benchmark_cifar10.py" \
    "$CROSS_BATCH/code/benchmarks/benchmark_dataset.py" \
    "$CROSS_BATCH/code/benchmarks/downstream_data.py"; do
    [[ -s $path ]] || fail "missing immutable cross-batch code: $path"
done

# Local metadata check only; this submits nothing.
bash "$AUDIT" "$CROSS_BATCH"

for target in CIFAR10 CIFAR100 TinyImageNet Aircraft CUB2011 Dogs Flowers; do
    [[ -s $CROSS_BATCH/targets/$target.pt ]] || fail "missing target cache: $target"
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

extract_ssl_epochs() {
    sed -nE 's/^[[:space:]]*"ssl_epochs"[[:space:]]*:[[:space:]]*([0-9]+).*/\1/p' "$1" | sed -n '1p'
}

checkpoint_for_seed() {
    local root=$1
    local seed=$2
    local direct
    local matches=()
    printf -v direct '%s/seed_%03d.pt' "$root" "$seed"
    [[ -s $direct ]] && matches+=("$direct")
    for direct in "$root"/shard_*/seed_$(printf '%03d' "$seed").pt; do
        [[ -s $direct ]] && matches+=("$direct")
    done
    (( ${#matches[@]} == 1 ))
}

find_compatible_refresh_encoder() {
    local root=$1
    local seed config epochs
    [[ -d $root ]] || return 1
    for (( seed=0; seed<10; seed++ )); do
        checkpoint_for_seed "$root" "$seed" || return 1
    done

    config=
    for config in "$root"/config.json "$root"/shard_*/config.json; do
        [[ -s $config ]] && break
        config=
    done
    [[ -n $config ]] || return 1
    grep -Eq '"protocol"[[:space:]]*:[[:space:]]*"full_data_refresh_v1_random_labels_barlow2x"' "$config" || return 1
    grep -Eq '"source"[[:space:]]*:[[:space:]]*"CIFAR10"' "$config" || return 1
    grep -Eq '"model"[[:space:]]*:[[:space:]]*"ResNet18"' "$config" || return 1
    grep -Eq '"ssl_method"[[:space:]]*:[[:space:]]*"barlowtwins"' "$config" || return 1
    epochs=$(extract_ssl_epochs "$config")
    [[ $epochs =~ ^[0-9]+$ ]] && (( epochs > 0 )) || return 1
    C10_EXISTING_EPOCHS=$epochs
    return 0
}

default_c10_encoder="$FULL_REFRESH_BATCH/runs/full_CIFAR10_ResNet18_barlowtwins/encoders"
C10_EXISTING_ROOT=${CIFAR10_BT_ENCODER_ROOT:-$default_c10_encoder}
C10_EXISTING_EPOCHS=
reuse_c10=0
if find_compatible_refresh_encoder "$C10_EXISTING_ROOT"; then
    [[ -s $FULL_REFRESH_BATCH/code/benchmarks/benchmark_full_refresh.py ]] || \
        fail "missing immutable full-refresh code under $FULL_REFRESH_BATCH/code"
    reuse_c10=1
    printf 'Reusing existing CIFAR-10 Barlow-Twins checkpoints: %s (%s epochs)\n' \
        "$C10_EXISTING_ROOT" "$C10_EXISTING_EPOCHS"
else
    printf '%s\n' 'No complete compatible CIFAR-10 Barlow-Twins checkpoint set found; a new 600-epoch set will be trained.'
fi

if (( dry_run )); then
    COMPLETION_BATCH="$RESULTS_ROOT/<new-batch>"
    COMPLETION_CODE_ROOT="$COMPLETION_BATCH/code"
    TARGET_CACHE_ROOT="$COMPLETION_BATCH/targets"
    RUN_WORKER="$WORKER"
else
    mkdir -p "$RESULTS_ROOT"
    COMPLETION_BATCH=$(mktemp -d "$RESULTS_ROOT/batch_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")
    COMPLETION_CODE_ROOT="$COMPLETION_BATCH/code"
    TARGET_CACHE_ROOT="$COMPLETION_BATCH/targets"
    RUN_WORKER="$COMPLETION_CODE_ROOT/benchmarks/killarney_short_jobs/full_data_completion_job.sh"

    mkdir -p "$COMPLETION_BATCH/logs" "$COMPLETION_BATCH/evaluations" \
        "$COMPLETION_CODE_ROOT/benchmarks/killarney_short_jobs" \
        "$COMPLETION_CODE_ROOT/OT-SSL-DD" "$TARGET_CACHE_ROOT"
    cp benchmarks/benchmark_cifar10.py benchmarks/benchmark_dataset.py \
        benchmarks/benchmark_full_refresh.py \
        benchmarks/downstream_data.py "$COMPLETION_CODE_ROOT/benchmarks/"
    cp OT-SSL-DD/utils.py OT-SSL-DD/networks.py "$COMPLETION_CODE_ROOT/OT-SSL-DD/"
    cp "$WORKER" "$ENVIRONMENT" "$COMPLETION_CODE_ROOT/benchmarks/killarney_short_jobs/"
    for target in CIFAR10 CIFAR100 TinyImageNet Aircraft CUB2011 Dogs Flowers; do
        ln -s "$CROSS_BATCH/targets/$target.pt" "$TARGET_CACHE_ROOT/$target.pt"
    done
    ln -s "$CROSS_BATCH/encoders" "$COMPLETION_BATCH/existing_cross_encoders"
    ln -s "$CROSS_BATCH/evaluations" "$COMPLETION_BATCH/existing_cross_evaluations"
    printf '%s\n' "$CROSS_BATCH" > "$COMPLETION_BATCH/cross_batch.txt"
    printf 'submitted_utc\tjob_id\taccount\ttask\tdependency\n' > "$COMPLETION_BATCH/submitted.tsv"
    tar -czf "$COMPLETION_BATCH/source.tar.gz" \
        benchmarks/benchmark_cifar10.py benchmarks/benchmark_dataset.py \
        benchmarks/benchmark_full_refresh.py \
        benchmarks/downstream_data.py \
        benchmarks/killarney_short_jobs/full_data_completion_job.sh \
        benchmarks/killarney_short_jobs/full_data_completion_environment.sh \
        benchmarks/killarney_short_jobs/submit_full_data_completion_to_wang.sh \
        benchmarks/killarney_short_jobs/check_full_data_completion.sh \
        OT-SSL-DD/utils.py OT-SSL-DD/networks.py
fi

export PROJECT_ROOT COMPLETION_BATCH COMPLETION_CODE_ROOT TARGET_CACHE_ROOT CROSS_BATCH
export FULL_REFRESH_BATCH VENV_ACTIVATE

LAST_JOB_ID=
dry_id=990000
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
        --output="$COMPLETION_BATCH/logs/%x-%j.out")
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
            >> "$COMPLETION_BATCH/submitted.tsv"
        printf 'Submitted %s as %s on %s\n' "$name" "$LAST_JOB_ID" "$ACCOUNT"
    fi
}

# Six missing main-table cells from already saved 300-epoch checkpoint sets.
submit_gpu kil_full_c100_bt_main 00:30:00 8G '' \
    cross-main CIFAR100 barlowtwins
submit_gpu kil_full_tiny_sim_main 00:30:00 8G '' \
    cross-main TinyImageNet simclr
submit_gpu kil_full_tiny_bt_main 00:30:00 8G '' \
    cross-main TinyImageNet barlowtwins

# Fourteen missing CIFAR-10 Barlow-Twins cells: main plus six downstream targets.
if (( reuse_c10 )); then
    C10_ENCODER_ROOT=$C10_EXISTING_ROOT
    C10_SSL_EPOCHS=$C10_EXISTING_EPOCHS
    C10_ENGINE=refresh
    c10_dependency=
else
    C10_ENCODER_ROOT="$COMPLETION_BATCH/encoders/cifar10_ResNet18_barlowtwins_full_size100"
    C10_SSL_EPOCHS=600
    C10_ENGINE=completion
    c10_pretrain_ids=()
    for seed_start in 0 5; do
        submit_gpu "kil_full_c10_bt_pre_${seed_start}" 13:00:00 8G '' \
            c10-pretrain "$seed_start" 5
        c10_pretrain_ids+=("$LAST_JOB_ID")
    done
    dependency_ids=$(IFS=:; echo "${c10_pretrain_ids[*]}")
    c10_dependency="afterok:$dependency_ids"
fi

if (( ! dry_run )); then
    printf '%s\n' "$C10_ENCODER_ROOT" > "$COMPLETION_BATCH/c10_encoder_root.txt"
    printf '%s\n' "$C10_SSL_EPOCHS" > "$COMPLETION_BATCH/c10_ssl_epochs.txt"
    printf '%s\n' "$C10_ENGINE" > "$COMPLETION_BATCH/c10_engine.txt"
fi

submit_gpu kil_full_c10_bt_eval 02:30:00 8G "$c10_dependency" \
    c10-evaluate "$C10_ENCODER_ROOT" "$C10_SSL_EPOCHS" "$C10_ENGINE"

printf '\n%s\n' 'Submission plan complete.'
printf 'Account: %s for every job\n' "$ACCOUNT"
printf 'Results: %s\n' "$COMPLETION_BATCH"
printf '%s\n' 'CPU Slurm jobs: 0'
if (( dry_run )); then
    printf '%s\n' 'Dry run only: nothing was submitted.'
else
    printf 'After completion, run:\n  bash %q %q\n' \
        "$BUNDLE/check_full_data_completion.sh" "$COMPLETION_BATCH"
fi
