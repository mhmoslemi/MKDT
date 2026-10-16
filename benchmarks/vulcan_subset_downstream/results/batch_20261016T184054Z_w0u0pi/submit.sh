#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
export PROJECT_ROOT
cd "$PROJECT_ROOT"

dry_run=0
if (( $# == 1 )) && [[ $1 == --dry-run ]]; then
    dry_run=1
elif (( $# != 0 )); then
    echo "Usage: bash $0 [--dry-run]" >&2
    exit 2
fi

ACCOUNT="${ACCOUNT:-aip-yiweilu}"
UPSTREAM_BATCH="${UPSTREAM_BATCH:-$PROJECT_ROOT/benchmarks/vulcan_subset_refresh/results/batch_20261016T163046Z_utW9z3}"
VENV_ACTIVATE="${VENV_ACTIVATE:-$HOME/ENV/bin/activate}"
export ACCOUNT UPSTREAM_BATCH VENV_ACTIVATE

[[ $ACCOUNT == aip-yiweilu ]] || {
    echo "This launcher is intentionally pinned to the Lu account (aip-yiweilu)." >&2
    exit 1
}
[[ -f $UPSTREAM_BATCH/configs.tsv ]] || {
    echo "Missing upstream manifest: $UPSTREAM_BATCH/configs.tsv" >&2
    exit 1
}
upstream_code="$UPSTREAM_BATCH/code"
for source_file in \
    benchmarks/benchmark_cifar10.py \
    benchmarks/benchmark_dataset.py \
    benchmarks/benchmark_subset_refresh.py \
    benchmarks/downstream_data.py \
    OT-SSL-DD/utils.py \
    OT-SSL-DD/networks.py; do
    [[ -f "$upstream_code/$source_file" ]] || {
        echo "Missing upstream snapshot file: $upstream_code/$source_file" >&2
        exit 1
    }
done
[[ -f $VENV_ACTIVATE ]] || { echo "Missing Python environment: $VENV_ACTIVATE" >&2; exit 1; }
command -v jq >/dev/null || { echo "jq is required" >&2; exit 1; }

cache_root="$PROJECT_ROOT/benchmarks/cross_downstream/results/batch_20261009T171449Z_o91BKP/targets"
for target in CIFAR10 CIFAR100 Aircraft CUB2011 Dogs Flowers TinyImageNet; do
    [[ -f "$cache_root/$target.pt" ]] || {
        echo "Missing target cache: $cache_root/$target.pt" >&2
        exit 1
    }
done

results_root="$PROJECT_ROOT/benchmarks/vulcan_subset_downstream/results"
mkdir -p "$results_root"
DOWNSTREAM_BATCH="$(mktemp -d "$results_root/batch_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")"
DOWNSTREAM_CODE_ROOT="$DOWNSTREAM_BATCH/code"
TARGET_CACHE_ROOT="$DOWNSTREAM_BATCH/targets"
export DOWNSTREAM_BATCH DOWNSTREAM_CODE_ROOT TARGET_CACHE_ROOT

mkdir -p "$DOWNSTREAM_BATCH/logs" "$DOWNSTREAM_BATCH/runs" \
    "$DOWNSTREAM_CODE_ROOT/benchmarks/vulcan_subset_downstream" \
    "$DOWNSTREAM_CODE_ROOT/OT-SSL-DD" "$TARGET_CACHE_ROOT"
for target in CIFAR10 CIFAR100 Aircraft CUB2011 Dogs Flowers TinyImageNet; do
    ln -s "$cache_root/$target.pt" "$TARGET_CACHE_ROOT/$target.pt"
done

cp "$upstream_code/benchmarks/benchmark_cifar10.py" \
    "$upstream_code/benchmarks/benchmark_dataset.py" \
    "$upstream_code/benchmarks/benchmark_subset_refresh.py" \
    "$upstream_code/benchmarks/downstream_data.py" \
    "$DOWNSTREAM_CODE_ROOT/benchmarks/"
cp benchmarks/vulcan_subset_downstream/environment.sh \
    benchmarks/vulcan_subset_downstream/job.sh \
    "$DOWNSTREAM_CODE_ROOT/benchmarks/vulcan_subset_downstream/"
cp "$upstream_code/OT-SSL-DD/utils.py" "$upstream_code/OT-SSL-DD/networks.py" \
    "$DOWNSTREAM_CODE_ROOT/OT-SSL-DD/"
cp OT-SSL-DD/training_epochs.json "$DOWNSTREAM_CODE_ROOT/OT-SSL-DD/training_epochs.json"
cp OT-SSL-DD/training_epochs.json "$DOWNSTREAM_BATCH/training_epochs.json"
printf '%s\n' "$UPSTREAM_BATCH" > "$DOWNSTREAM_BATCH/upstream_batch.txt"

cp benchmarks/vulcan_subset_downstream/submit.sh "$DOWNSTREAM_BATCH/submit.sh"
tar -czf "$DOWNSTREAM_BATCH/source.tar.gz" -C "$DOWNSTREAM_BATCH" \
    code submit.sh training_epochs.json upstream_batch.txt

printf 'config\tsource\tmethod\tmodel\tssl_method\tsubset_percentage\tssl_epochs\tencoder_dir\n' \
    > "$DOWNSTREAM_BATCH/configs.tsv"
printf 'config\tsource\tmethod\tmodel\tssl_method\tsubset_percentage\tssl_epochs\tcheckpoint_count\tpretrain_job\n' \
    > "$DOWNSTREAM_BATCH/pending.tsv"

ready=0
pending=0
while IFS=$'\t' read -r config source method model ssl subset epochs runs probe1 probe5 pretrain_job evaluate_job; do
    [[ $config == config ]] && continue

    # The CIFAR-10 downstream tables contain all three backbones. The
    # CIFAR-100 downstream table contains ResNet18 only.
    if [[ $source == CIFAR100 && $model != ResNet18 ]]; then
        continue
    fi

    encoder_dir="$UPSTREAM_BATCH/runs/$config/encoders"
    count=0
    complete=1
    for seed in 000 001 002 003 004 005 006 007 008 009; do
        if [[ -s "$encoder_dir/seed_${seed}.pt" ]]; then
            count=$((count + 1))
        else
            complete=0
        fi
    done

    if (( complete )); then
        [[ -f "$encoder_dir/config.json" ]] || {
            echo "Missing encoder config for ready set: $config" >&2
            exit 1
        }
        jq -e \
            --arg source "$source" --arg method "$method" --arg model "$model" \
            --arg ssl "$ssl" --argjson subset "$subset" --argjson epochs "$epochs" \
            '.source_dataset == $source and .method == $method and
             .model == $model and .ssl_method == $ssl and
             .subset_percentage == $subset and .ssl_epochs == $epochs' \
            "$encoder_dir/config.json" >/dev/null || {
                echo "Encoder metadata mismatch: $config" >&2
                exit 1
            }
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$config" "$source" "$method" "$model" "$ssl" "$subset" \
            "$epochs" "$encoder_dir" >> "$DOWNSTREAM_BATCH/configs.tsv"
        ready=$((ready + 1))
    else
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$config" "$source" "$method" "$model" "$ssl" "$subset" \
            "$epochs" "$count" "$pretrain_job" >> "$DOWNSTREAM_BATCH/pending.tsv"
        pending=$((pending + 1))
    fi
done < "$UPSTREAM_BATCH/configs.tsv"

(( ready > 0 )) || { echo "No complete checkpoint sets are ready." >&2; exit 1; }
array_end=$((ready - 1))
args=(sbatch --parsable --account="$ACCOUNT" --job-name=vsd_linear \
    --time=0-00:30:00 --chdir="$PROJECT_ROOT" \
    --array="0-${array_end}%16" \
    --output="$DOWNSTREAM_BATCH/logs/%x-%A_%a.out")

if (( dry_run )); then
    printf 'DRY-RUN: '
    printf '%q ' "${args[@]}" "$DOWNSTREAM_CODE_ROOT/benchmarks/vulcan_subset_downstream/job.sh"
    printf '\n'
    job_id=DRY-RUN
else
    job_id=$("${args[@]}" "$DOWNSTREAM_CODE_ROOT/benchmarks/vulcan_subset_downstream/job.sh")
    job_id=${job_id%%;*}
fi

printf 'submitted_utc\tjob_id\taccount\tarray\tready_configs\tpending_configs\n' \
    > "$DOWNSTREAM_BATCH/submitted.tsv"
printf '%s\t%s\t%s\t0-%s%%16\t%s\t%s\n' \
    "$(date -u +%FT%TZ)" "$job_id" "$ACCOUNT" "$array_end" "$ready" "$pending" \
    >> "$DOWNSTREAM_BATCH/submitted.tsv"

mode=SUBMITTED
(( dry_run )) && mode=DRY-RUN
printf '%s batch: %s\nJob: %s (array 0-%s%%16)\nAccount: %s\nReady configs: %s\nDeferred configs: %s\nEvaluations: %s summaries (%s configs x 6 targets x 2 label budgets)\n' \
    "$mode" "$DOWNSTREAM_BATCH" "$job_id" "$array_end" "$ACCOUNT" \
    "$ready" "$pending" "$((ready * 12))" "$ready"
