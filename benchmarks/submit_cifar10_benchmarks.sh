#!/bin/bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
export PROJECT_ROOT
cd "$PROJECT_ROOT"
dry_run=0
if [[ ${1:-} == --dry-run && $# == 1 ]]; then
    dry_run=1
elif (( $# )); then
    echo "Usage: bash $0 [--dry-run]" >&2
    exit 2
fi
JOB_ARCHIVE="$PROJECT_ROOT/benchmarks/cifar10/jobs.tar"
[[ -f $JOB_ARCHIVE && -f benchmarks/cifar10/prepare.sh ]] || { echo 'Missing benchmark jobs.' >&2; exit 1; }
mapfile -t jobs < <(tar -tf "$JOB_ARCHIVE")
if (( dry_run )); then
    BENCH_RUN_ROOT="$PROJECT_ROOT/benchmarks/cifar10/results/<new-batch>"
else
    command -v sbatch >/dev/null
    [[ -f ${VENV_ACTIVATE:-$HOME/ENV/bin/activate} ]] || { echo 'Python environment is missing.' >&2; exit 1; }
    [[ -d ${DATA_ROOT:-$SCRATCH/data}/cifar-10-batches-py ]] || { echo 'Stage CIFAR-10 first; the jobs do not download data.' >&2; exit 1; }
    mkdir -p benchmarks/cifar10/results
    BENCH_RUN_ROOT=$(mktemp -d "$PROJECT_ROOT/benchmarks/cifar10/results/batch_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")
    mkdir "$BENCH_RUN_ROOT/logs"
    cp benchmarks/cifar10/configs.tsv "$BENCH_RUN_ROOT/configs.tsv"
    BENCH_CODE_ROOT="$BENCH_RUN_ROOT/code"
    mkdir -p "$BENCH_CODE_ROOT/OT-SSL-DD" "$BENCH_CODE_ROOT/benchmarks/cifar10"
    cp OT-SSL-DD/utils.py OT-SSL-DD/networks.py "$BENCH_CODE_ROOT/OT-SSL-DD/"
    cp benchmarks/benchmark_cifar10.py "$BENCH_CODE_ROOT/benchmarks/"
    cp benchmarks/cifar10/report.py benchmarks/cifar10/configs.tsv "$BENCH_CODE_ROOT/benchmarks/cifar10/"
    tar -czf "$BENCH_RUN_ROOT/source.tar.gz" benchmarks/benchmark_cifar10.py OT-SSL-DD/utils.py OT-SSL-DD/networks.py benchmarks/cifar10/jobs.tar benchmarks/cifar10/environment.sh benchmarks/cifar10/report.py benchmarks/cifar10/protocol.txt
fi
export BENCH_RUN_ROOT BENCH_CODE_ROOT
export SELECTION_DIR="$BENCH_RUN_ROOT/selections"
submit() {
    local script=$1
    shift
    local args=(sbatch --parsable --chdir="$PROJECT_ROOT" --output="$BENCH_RUN_ROOT/logs/%x-%j.out" "$@")
    if (( dry_run )); then
        printf '%q ' "${args[@]}" >&2
        printf '%s\n' "$script" >&2
        printf 'DRY_JOB_ID\n'
    else
        local id
        if [[ $script == benchmarks/cifar10/jobs/* ]]; then
            id=$(tar -xOf "$JOB_ARCHIVE" "$script" | "${args[@]}")
        else
            id=$("${args[@]}" "$script")
        fi
        printf '%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "$id" "$script" >> "$BENCH_RUN_ROOT/submitted.tsv"
        printf 'Submitted %s: %s\n' "$(basename "$script")" "$id" >&2
        printf '%s\n' "${id%%;*}"
    fi
}
prepare_id=$(submit "$PROJECT_ROOT/benchmarks/cifar10/prepare.sh")
ids=()
for script in "${jobs[@]}"; do
    dependencies=()
    if [[ $script == *_kmeans_* ]]; then
        dependencies=(--dependency="afterok:$prepare_id" --kill-on-invalid-dep=yes)
    fi
    ids+=("$(submit "$script" "${dependencies[@]}")")
done
dependency=$(IFS=:; echo "${ids[*]}")
report_id=$(submit "$PROJECT_ROOT/benchmarks/cifar10/report.sh" --dependency="afterany:$dependency")
printf '%s configurations × 15 runs, plus K-means preparation and a results-table job.\nResults: %s\n' "${#jobs[@]}" "$BENCH_RUN_ROOT"
