#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
export PROJECT_ROOT
cd "$PROJECT_ROOT"

dry_run=0
if [[ ${1:-} == --dry-run ]]; then
    dry_run=1
    shift
fi
[[ $# -eq 0 ]] || { echo "Usage: bash $0 [--dry-run]" >&2; exit 2; }

MKDT_IMAGE_ROOT="${MKDT_IMAGE_ROOT:-$PROJECT_ROOT/logged_files/CIFAR10}"
TARGET_CACHE_ROOT="${TARGET_CACHE_ROOT:-$PROJECT_ROOT/benchmarks/downstream/results/batch_20261009T043709Z_zC8EPD/targets}"
export MKDT_IMAGE_ROOT TARGET_CACHE_ROOT

image_for_size() {
    case "$1" in
        1) printf '%s\n' "${MKDT_IMAGE_1:-$MKDT_IMAGE_ROOT/CIFAR10_1pct/images_5000.pt}" ;;
        2) printf '%s\n' "${MKDT_IMAGE_2:-$MKDT_IMAGE_ROOT/CIFAR10_2pct/images_5000.pt}" ;;
        5) printf '%s\n' "${MKDT_IMAGE_5:-$MKDT_IMAGE_ROOT/CIFAR10_5pct/images_5000.pt}" ;;
    esac
}

missing=0
for size in 1 2 5; do
    image=$(image_for_size "$size")
    if [[ ! -f $image ]]; then
        echo "Missing MKDT ${size}% images: $image" >&2
        missing=1
    fi
done
for target in CIFAR100 Aircraft CUB2011 Dogs Flowers TinyImageNet; do
    if [[ ! -f $TARGET_CACHE_ROOT/$target.pt ]]; then
        echo "Missing target cache: $TARGET_CACHE_ROOT/$target.pt" >&2
        missing=1
    fi
done
if (( missing != 0 && dry_run == 0 )); then
    exit 1
fi

if (( dry_run )); then
    MKDT_BATCH_ROOT="$PROJECT_ROOT/benchmarks/mkdt_images_only/results/<new-batch>"
    MKDT_CODE_ROOT="$PROJECT_ROOT"
else
    command -v sbatch >/dev/null
    [[ -f ${VENV_ACTIVATE:-$HOME/ENV/bin/activate} ]] || { echo 'Python environment is missing.' >&2; exit 1; }
    mkdir -p benchmarks/mkdt_images_only/results
    MKDT_BATCH_ROOT=$(mktemp -d "$PROJECT_ROOT/benchmarks/mkdt_images_only/results/batch_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")
    mkdir -p "$MKDT_BATCH_ROOT/logs" "$MKDT_BATCH_ROOT/runs"
    MKDT_CODE_ROOT="$MKDT_BATCH_ROOT/code"
    mkdir -p "$MKDT_CODE_ROOT/benchmarks/mkdt_images_only" "$MKDT_CODE_ROOT/OT-SSL-DD"
    cp benchmarks/benchmark_cifar10.py benchmarks/benchmark_mkdt_images.py benchmarks/downstream_data.py "$MKDT_CODE_ROOT/benchmarks/"
    cp benchmarks/mkdt_images_only/report.py "$MKDT_CODE_ROOT/benchmarks/mkdt_images_only/"
    cp OT-SSL-DD/utils.py OT-SSL-DD/networks.py "$MKDT_CODE_ROOT/OT-SSL-DD/"
    tar -czf "$MKDT_BATCH_ROOT/source.tar.gz" \
        benchmarks/benchmark_cifar10.py benchmarks/benchmark_mkdt_images.py benchmarks/downstream_data.py \
        benchmarks/mkdt_images_only OT-SSL-DD/utils.py OT-SSL-DD/networks.py
fi
export MKDT_BATCH_ROOT MKDT_CODE_ROOT

submit() {
    local account=$1
    local name=$2
    local time_limit=$3
    local phase=$4
    local size=$5
    local ssl=$6
    local epochs=$7
    local dependency=${8:-}
    local args=(sbatch --parsable --account="$account" --job-name="$name" \
        --time="$time_limit" --chdir="$PROJECT_ROOT" \
        --output="$MKDT_BATCH_ROOT/logs/%x-%j.out")
    [[ -n $dependency ]] && args+=(--dependency="$dependency" --kill-on-invalid-dep=yes)
    if (( dry_run )); then
        printf '%q ' "${args[@]}" >&2
        printf '%q ' "$PROJECT_ROOT/benchmarks/mkdt_images_only/job.sh" >&2
        printf '%q ' "$phase" "$size" "$ssl" "$epochs" >&2
        printf '\n' >&2
        printf 'DRY_JOB_ID\n'
    else
        local id
        id=$("${args[@]}" "$PROJECT_ROOT/benchmarks/mkdt_images_only/job.sh" \
            "$phase" "$size" "$ssl" "$epochs")
        printf '%s\t%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "$id" "$account" "$name" >> "$MKDT_BATCH_ROOT/submitted.tsv"
        printf 'Submitted %s on %s: %s\n' "$name" "$account" "$id" >&2
        printf '%s\n' "${id%%;*}"
    fi
}

declare -A sim_epochs=( [1]=5000 [2]=4000 [5]=3000 )
declare -A bt_epochs=( [1]=10000 [2]=8000 [5]=6000 )
declare -A sim_time=( [1]=0-01:30:00 [2]=0-01:40:00 [5]=0-02:15:00 )
declare -A bt_time=( [1]=0-02:05:00 [2]=0-02:40:00 [5]=0-04:10:00 )

evaluation_ids=()
for size in 1 2 5; do
    for ssl in simclr barlowtwins; do
        if [[ $ssl == simclr ]]; then
            epochs=${sim_epochs[$size]}
            limit=${sim_time[$size]}
        else
            epochs=${bt_epochs[$size]}
            limit=${bt_time[$size]}
        fi
        pretrain_id=$(submit aip-boyuwang "mkdt_pre_s${size}_${ssl}" "$limit" \
            pretrain "$size" "$ssl" "$epochs")
        # One of twelve GPU jobs uses Lu (8.3%); the other eleven use Wang.
        eval_account=aip-boyuwang
        [[ $size == 5 && $ssl == barlowtwins ]] && eval_account=aip-yiweilu
        eval_id=$(submit "$eval_account" "mkdt_eval_s${size}_${ssl}" 0-00:45:00 \
            evaluate "$size" "$ssl" "$epochs" "afterok:$pretrain_id")
        evaluation_ids+=("$eval_id")
    done
done

dependency=$(IFS=:; echo "${evaluation_ids[*]}")
if (( dry_run )); then
    report_id=DRY_JOB_ID
else
    report_id=$(sbatch --parsable --account=aip-boyuwang --job-name=mkdt_tables \
        --chdir="$PROJECT_ROOT" --output="$MKDT_BATCH_ROOT/logs/%x-%j.out" \
        --dependency="afterany:$dependency" \
        "$PROJECT_ROOT/benchmarks/mkdt_images_only/report.sh")
    printf '%s\t%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "$report_id" aip-boyuwang mkdt_tables >> "$MKDT_BATCH_ROOT/submitted.tsv"
fi

printf 'Submitted 12 GPU jobs (11 Wang, 1 Lu) plus one CPU report job.\n'
printf 'Results: %s\nReport job: %s\n' "$MKDT_BATCH_ROOT" "$report_id"
