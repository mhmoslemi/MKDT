#!/bin/bash
set -euo pipefail

# Killarney: evaluate only the four long-epoch ResNet18 full-data encoders
# needed by the active CIFAR-10/CIFAR-100 downstream tables. No SSL training.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
PROJECT_ROOT=$(readlink -f "$SCRIPT_DIR/../..")
MAINLONG_BATCH="$PROJECT_ROOT/benchmarks/full_data_main_long/results/batch_20261016T042912Z_MLpgLE"
MAINLONG_CODE_ROOT="$MAINLONG_BATCH/code"
VENV_ACTIVATE="${VENV_ACTIVATE:-$HOME/ENV/bin/activate}"
ACCOUNT=aip-boyuwang
WORKER_SOURCE="$SCRIPT_DIR/full_data_main_long_downstream_job.sh"
WORKER_DEST="$MAINLONG_CODE_ROOT/benchmarks/killarney_short_jobs/full_data_main_long_downstream_job.sh"
DEPENDENCY_JOB=6322342

fail() {
    echo "FATAL: $*" >&2
    exit 1
}

cd "$PROJECT_ROOT"
[[ -s $MAINLONG_BATCH/protocol.txt ]] || fail "missing long-epoch batch: $MAINLONG_BATCH"
[[ $(<"$MAINLONG_BATCH/protocol.txt") == full_data_main_simclr1000_barlow1500_v1 ]] || \
    fail "wrong protocol in $MAINLONG_BATCH/protocol.txt"
[[ -s $VENV_ACTIVATE ]] || fail "missing Python environment: $VENV_ACTIVATE"
[[ -s $MAINLONG_BATCH/target_source_batch.txt ]] || fail "missing target-source pointer"
target_source_batch=$(<"$MAINLONG_BATCH/target_source_batch.txt")
TARGET_CACHE_ROOT="$target_source_batch/targets"
for target in CIFAR10 CIFAR100 Aircraft CUB2011 Dogs Flowers TinyImageNet; do
    [[ -s $TARGET_CACHE_ROOT/$target.pt ]] || fail "missing target cache: $TARGET_CACHE_ROOT/$target.pt"
done

gpu_inventory=$(sinfo -h -o '%G' --Node | sort -u)
grep -q 'gpu:l40s:' <<< "$gpu_inventory" || fail "gpu:l40s is not available"

if [[ -e $WORKER_DEST ]]; then
    cmp -s "$WORKER_SOURCE" "$WORKER_DEST" || fail "batch worker differs from repository worker"
else
    cp "$WORKER_SOURCE" "$WORKER_DEST"
    chmod 750 "$WORKER_DEST"
fi
mkdir -p "$MAINLONG_BATCH/logs"

checkpoint_count() {
    local source=$1 ssl=$2 epochs=$3 prefix identifier root seed checkpoint
    local matches=() count=0
    [[ $source == CIFAR10 ]] && prefix=cifar10 || prefix=cifar100
    identifier="${prefix}_ResNet18_${ssl}_full_size100_${epochs}ep"
    root="$MAINLONG_BATCH/encoders/$identifier"
    shopt -s nullglob
    for seed in {0..9}; do
        printf -v checkpoint 'seed_%03d.pt' "$seed"
        matches=("$root"/shard_*/"$checkpoint")
        (( ${#matches[@]} <= 1 )) || fail "$identifier seed $seed has duplicate checkpoints"
        if (( ${#matches[@]} == 1 )) && [[ -s ${matches[0]} ]]; then count=$((count + 1)); fi
    done
    printf '%d\n' "$count"
}

export PROJECT_ROOT MAINLONG_BATCH MAINLONG_CODE_ROOT TARGET_CACHE_ROOT VENV_ACTIVATE

specs=(
    'kmld_c10_r18_s|CIFAR10|simclr|1000|'
    'kmld_c10_r18_b|CIFAR10|barlowtwins|1500|'
    'kmld_c100_r18_b|CIFAR100|barlowtwins|1500|'
    'kmld_c100_r18_s|CIFAR100|simclr|1000|6322342'
)

manifest="$MAINLONG_BATCH/resnet18_downstream_submitted.tsv"
if [[ ! -e $manifest ]]; then
    printf 'submitted_utc\tjob_id\taccount\tsource\tssl_method\tdependency\n' > "$manifest"
fi

submitted=0
for spec in "${specs[@]}"; do
    IFS='|' read -r name source ssl epochs dependency <<< "$spec"
    queued=$(squeue -h -u "$USER" -n "$name" -o '%i' || true)
    if [[ -n $queued ]]; then
        printf 'SKIP %s: already queued as %s\n' "$name" "$queued"
        continue
    fi

    count=$(checkpoint_count "$source" "$ssl" "$epochs")
    if [[ -z $dependency ]]; then
        (( count == 10 )) || fail "$name has only $count/10 checkpoints"
    else
        printf '%s currently has %d/10 checkpoints; evaluation will wait for %s\n' \
            "$name" "$count" "$dependency"
    fi

    args=(sbatch --parsable
        --account="$ACCOUNT"
        --job-name="$name"
        --nodes=1
        --ntasks=1
        --cpus-per-task=1
        --gres=gpu:l40s:1
        --mem=8G
        --time=03:00:00
        --chdir="$PROJECT_ROOT"
        --output="$MAINLONG_BATCH/logs/%x-%j.out")
    [[ -z $dependency ]] || args+=(--dependency="afterok:$dependency")

    job_id=$("${args[@]}" "$WORKER_DEST" "$source" ResNet18 "$ssl")
    job_id=${job_id%%;*}
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$(date -u +%FT%TZ)" "$job_id" "$ACCOUNT" "$source" "$ssl" \
        "${dependency:--}" >> "$manifest"
    printf 'Submitted %s as %s%s\n' "$name" "$job_id" \
        "${dependency:+ (afterok:$dependency)}"
    submitted=$((submitted + 1))
done

printf 'Submitted jobs: %d\nScope: ResNet18 full-data downstream only; no SSL training\n' "$submitted"
