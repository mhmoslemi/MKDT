#!/bin/bash
set -euo pipefail

# Evaluate the already-trained 1000/1500-epoch encoders needed by the CIFAR
# downstream tables. This script never launches pretraining. Before submitting
# anything, it verifies that every pending configuration has exactly one
# nonempty checkpoint for each encoder seed 0..9.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
PROJECT_ROOT=$(readlink -f "$SCRIPT_DIR/../..")
RESULTS_ROOT=${MAINLONG_RESULTS_ROOT:-$PROJECT_ROOT/benchmarks/full_data_main_long/results}
WORKER_SOURCE="$SCRIPT_DIR/full_data_main_long_downstream_job.sh"
AUDIT="$SCRIPT_DIR/check_full_data_main_long_downstream.sh"
ACCOUNT=${ACCOUNT:-aip-boyuwang}
VENV_ACTIVATE=${VENV_ACTIVATE:-$HOME/ENV/bin/activate}
REQUESTED_EXCLUDED_NODES=${EXCLUDED_NODES:-kn072,kn140,kn161}
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

cd "$PROJECT_ROOT"
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
[[ -s $WORKER_SOURCE ]] || fail "missing downstream worker: $WORKER_SOURCE"
[[ -s $AUDIT ]] || fail "missing downstream audit: $AUDIT"
[[ -s $VENV_ACTIVATE ]] || fail "missing Python environment: $VENV_ACTIVATE"

MAINLONG_CODE_ROOT="$MAINLONG_BATCH/code"
WORKER_DEST="$MAINLONG_CODE_ROOT/benchmarks/killarney_short_jobs/full_data_main_long_downstream_job.sh"
ENVIRONMENT="$MAINLONG_CODE_ROOT/benchmarks/killarney_short_jobs/full_data_main_long_environment.sh"
PROGRAM="$MAINLONG_CODE_ROOT/benchmarks/benchmark_full_data_main_long.py"
MANIFEST="$MAINLONG_BATCH/downstream_submitted.tsv"
[[ -s $ENVIRONMENT ]] || fail "missing immutable batch environment: $ENVIRONMENT"
[[ -s $PROGRAM ]] || fail "missing immutable batch program: $PROGRAM"

target_source_batch=
if [[ -s $MAINLONG_BATCH/target_source_batch.txt ]]; then
    target_source_batch=$(<"$MAINLONG_BATCH/target_source_batch.txt")
fi
if [[ -n $target_source_batch && -d $target_source_batch/targets ]]; then
    TARGET_CACHE_ROOT="$target_source_batch/targets"
elif [[ -d $MAINLONG_BATCH/targets ]]; then
    TARGET_CACHE_ROOT="$MAINLONG_BATCH/targets"
else
    fail "cannot locate target caches for $MAINLONG_BATCH"
fi
for target in CIFAR10 CIFAR100 TinyImageNet Aircraft CUB2011 Dogs Flowers; do
    [[ -s $TARGET_CACHE_ROOT/$target.pt ]] || \
        fail "missing prepared target cache: $TARGET_CACHE_ROOT/$target.pt"
done

valid_evaluation_count() {
    local source=$1
    local target=$2
    local model=$3
    local ssl=$4
    local epochs=$5
    local label=$6
    local probe=$7
    local prefix identifier eval_dir job_dir summary
    local valid=0
    local seed_files=()

    if [[ $source == CIFAR10 ]]; then prefix=cifar10; else prefix=cifar100; fi
    identifier="${prefix}_${model}_${ssl}_full_size100_${epochs}ep"
    eval_dir="$MAINLONG_BATCH/evaluations/${identifier}__${target}_lbl${label}"

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
    done
    printf '%d\n' "$valid"
}

checkpoint_count() {
    local source=$1
    local model=$2
    local ssl=$3
    local epochs=$4
    local prefix identifier encoder_root seed checkpoint
    local present=0
    local matches=()

    if [[ $source == CIFAR10 ]]; then prefix=cifar10; else prefix=cifar100; fi
    identifier="${prefix}_${model}_${ssl}_full_size100_${epochs}ep"
    encoder_root="$MAINLONG_BATCH/encoders/$identifier"
    for seed in {0..9}; do
        printf -v checkpoint 'seed_%03d.pt' "$seed"
        matches=("$encoder_root"/shard_*/"$checkpoint")
        if (( ${#matches[@]} == 1 )) && [[ -s ${matches[0]} ]]; then
            present=$((present + 1))
        elif (( ${#matches[@]} > 1 )); then
            fail "$identifier seed $seed has ${#matches[@]} ambiguous checkpoints"
        fi
    done
    printf '%d\n' "$present"
}

configurations=(
    'c10_cn_s|CIFAR10|ConvNet|simclr|1000'
    'c10_cn_b|CIFAR10|ConvNet|barlowtwins|1500'
    'c10_vg_s|CIFAR10|VGG11|simclr|1000'
    'c10_vg_b|CIFAR10|VGG11|barlowtwins|1500'
    'c10_r18_s|CIFAR10|ResNet18|simclr|1000'
    'c10_r18_b|CIFAR10|ResNet18|barlowtwins|1500'
    'c100_r18_s|CIFAR100|ResNet18|simclr|1000'
    'c100_r18_b|CIFAR100|ResNet18|barlowtwins|1500'
)

declare -A pending_by_tag
checkpoint_preflight_failed=0
pending_configurations=0
printf '%s\n' 'Downstream preflight:'
for spec in "${configurations[@]}"; do
    IFS='|' read -r tag source model ssl epochs <<< "$spec"
    if [[ $source == CIFAR10 ]]; then
        targets=(CIFAR100 Aircraft CUB2011 Dogs Flowers TinyImageNet)
    else
        targets=(TinyImageNet CIFAR10 Aircraft CUB2011 Dogs Flowers)
    fi
    complete=0
    for target in "${targets[@]}"; do
        for label in 1 5; do
            if [[ $label == 1 ]]; then probe=500; else probe=300; fi
            valid=$(valid_evaluation_count \
                "$source" "$target" "$model" "$ssl" "$epochs" "$label" "$probe")
            (( valid <= 1 )) || \
                fail "$tag target=$target label=$label has $valid duplicate complete evaluations"
            complete=$((complete + valid))
        done
    done
    pending=$((12 - complete))
    pending_by_tag[$tag]=$pending
    if (( pending == 0 )); then
        printf '  COMPLETE %-12s results=12/12 (no job needed)\n' "$tag"
        continue
    fi

    checkpoints=$(checkpoint_count "$source" "$model" "$ssl" "$epochs")
    printf '  PENDING  %-12s results=%d/12 checkpoints=%d/10\n' \
        "$tag" "$complete" "$checkpoints"
    pending_configurations=$((pending_configurations + 1))
    if (( checkpoints != 10 )); then checkpoint_preflight_failed=1; fi
done

(( pending_configurations > 0 )) || {
    printf '%s\n' 'Every downstream result is already complete; nothing to submit.'
    exit 0
}
if (( checkpoint_preflight_failed && ! dry_run )); then
    fail 'checkpoint preflight failed; no jobs were submitted (this script never retrains encoders)'
fi

ACTIVE_EXCLUDED_NODES=
if (( ! dry_run )); then
    command -v sbatch >/dev/null 2>&1 || fail "sbatch is not available"
    command -v sinfo >/dev/null 2>&1 || fail "sinfo is not available"
    command -v squeue >/dev/null 2>&1 || fail "squeue is not available"

    gpu_inventory=$(sinfo -h -o '%G' --Node | sort -u)
    grep -q 'gpu:l40s:' <<< "$gpu_inventory" || {
        printf 'Discovered GPU inventory:\n%s\n' "$gpu_inventory" >&2
        fail 'gpu:l40s is not present in the discovered cluster inventory'
    }

    node_inventory=$(sinfo -h -N -o '%N' | sort -u)
    IFS=, read -r -a requested_nodes <<< "$REQUESTED_EXCLUDED_NODES"
    active_nodes=()
    for node in "${requested_nodes[@]}"; do
        if grep -Fxq "$node" <<< "$node_inventory"; then active_nodes+=("$node"); fi
    done
    if (( ${#active_nodes[@]} > 0 )); then
        ACTIVE_EXCLUDED_NODES=$(IFS=,; printf '%s' "${active_nodes[*]}")
    fi

    queued_names=$(squeue -h -u "$USER" -o '%j')
    if grep -Eq '^kmld_' <<< "$queued_names"; then
        printf 'Matching queued downstream jobs:\n' >&2
        grep -E '^kmld_' <<< "$queued_names" >&2
        fail 'long-run downstream work is already queued; refusing duplicate submission'
    fi

    if [[ -e $WORKER_DEST ]]; then
        cmp -s "$WORKER_SOURCE" "$WORKER_DEST" || \
            fail "immutable downstream worker already exists with different contents: $WORKER_DEST"
    else
        cp "$WORKER_SOURCE" "$WORKER_DEST"
        chmod 750 "$WORKER_DEST"
    fi
    RUN_WORKER=$WORKER_DEST
    mkdir -p "$MAINLONG_BATCH/logs"
    if [[ ! -e $MANIFEST ]]; then
        printf 'submitted_utc\tjob_id\taccount\tsource\tmodel\tssl_method\tpending_results\texcluded_nodes\n' \
            > "$MANIFEST"
    fi
else
    RUN_WORKER=$WORKER_SOURCE
fi

export PROJECT_ROOT MAINLONG_BATCH MAINLONG_CODE_ROOT TARGET_CACHE_ROOT VENV_ACTIVATE

submitted_jobs=0
for spec in "${configurations[@]}"; do
    IFS='|' read -r tag source model ssl epochs <<< "$spec"
    pending=${pending_by_tag[$tag]}
    (( pending > 0 )) || continue

    args=(sbatch --parsable
        --account="$ACCOUNT"
        --job-name="kmld_$tag"
        --nodes=1
        --ntasks=1
        --cpus-per-task=1
        --gres=gpu:l40s:1
        --time=03:00:00
        --mem=8G
        --chdir="$PROJECT_ROOT"
        --output="$MAINLONG_BATCH/logs/%x-%j.out")
    [[ -z $ACTIVE_EXCLUDED_NODES ]] || args+=(--exclude="$ACTIVE_EXCLUDED_NODES")

    if (( dry_run )); then
        printf 'DRY RUN: '
        printf '%q ' "${args[@]}" "$RUN_WORKER" "$source" "$model" "$ssl"
        printf '\n'
    else
        submitted=$("${args[@]}" "$RUN_WORKER" "$source" "$model" "$ssl")
        job_id=${submitted%%;*}
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$(date -u +%FT%TZ)" "$job_id" "$ACCOUNT" "$source" "$model" \
            "$ssl" "$pending" "${ACTIVE_EXCLUDED_NODES:--}" >> "$MANIFEST"
        printf 'Submitted %-16s as %s (%d pending results)\n' \
            "kmld_$tag" "$job_id" "$pending"
    fi
    submitted_jobs=$((submitted_jobs + 1))
done

printf '\nDownstream submission plan complete.\n'
printf 'Batch: %s\n' "$MAINLONG_BATCH"
printf 'Account: %s\n' "$ACCOUNT"
printf 'Evaluation-only GPU jobs: %d\n' "$submitted_jobs"
printf '%s\n' 'Scope: CIFAR-10 ConvNet/VGG11/ResNet18 plus CIFAR-100 ResNet18'
printf '%s\n' 'Each configuration: six targets x labels 1%/5%; probes 500/300 epochs; 10 runs'
printf '%s\n' 'Encoder training jobs: 0'
if (( dry_run )); then
    printf '%s\n' 'Dry run only: nothing was copied, created, or submitted.'
    if (( checkpoint_preflight_failed )); then
        printf '%s\n' \
            'DRY RUN BLOCKED: this filesystem copy lacks one or more required checkpoints.' >&2
        exit 1
    fi
else
    printf 'Submission record: %s\n' "$MANIFEST"
    printf 'After completion, audit with:\n  bash %q %q\n' "$AUDIT" "$MAINLONG_BATCH"
fi
