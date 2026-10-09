#!/bin/bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$PROJECT_ROOT"
DATA_ROOT="${DATA_ROOT:-$SCRATCH/data}"
TARGET_CACHE_ROOT="${TARGET_CACHE_ROOT:-$PROJECT_ROOT/benchmarks/downstream/results/batch_20261008T175805Z_GKcdxU/targets}"
mkdir -p benchmarks/ours
OURS_BATCH=$(mktemp -d "$PROJECT_ROOT/benchmarks/ours/batch_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")
OURS_CODE_ROOT="$OURS_BATCH/code"
export PROJECT_ROOT DATA_ROOT TARGET_CACHE_ROOT OURS_BATCH OURS_CODE_ROOT
mkdir -p "$OURS_BATCH/logs" "$OURS_CODE_ROOT/OT-SSL-DD" "$OURS_CODE_ROOT/benchmarks"
cp OT-SSL-DD/mina_IF.py OT-SSL-DD/utils.py OT-SSL-DD/networks.py OT-SSL-DD/training_epochs.json "$OURS_CODE_ROOT/OT-SSL-DD/"
cp benchmarks/benchmark_ours.py benchmarks/benchmark_cifar10.py benchmarks/benchmark_downstream.py benchmarks/downstream_data.py "$OURS_CODE_ROOT/benchmarks/"
srun --account=aip-yiweilu --job-name=ours_make_jobs --time=0-00:05:00 --ntasks=1 --cpus-per-task=1 --mem=1G bash -c 'set -euo pipefail; module load StdEnv/2023 python/3.11.5; export PYTHONDONTWRITEBYTECODE=1; python3 "$OURS_CODE_ROOT/benchmarks/benchmark_ours.py" make-jobs --batch "$OURS_BATCH" --project-root "$PROJECT_ROOT" --target-cache "$TARGET_CACHE_ROOT"'
declare -A job_ids=()
evaluation_ids=()
while IFS=$'\t' read -r stage key dependency limit member; do
    [[ $stage != stage ]] || continue
    member=${member%$'\r'}
    dependency_args=()
    if [[ $dependency == all_evaluations ]]; then
        dependency=$(IFS=:; echo "${evaluation_ids[*]}")
        dependency_args=(--dependency="afterany:$dependency")
    elif [[ $dependency != - ]]; then
        dependency_args=(--dependency="afterok:${job_ids[$dependency]}" --kill-on-invalid-dep=yes)
    fi
    id=$(tar -xOf "$OURS_BATCH/jobs.tar" "$member" | sbatch --parsable --account=aip-yiweilu --export=ALL --chdir="$PROJECT_ROOT" --output="$OURS_BATCH/logs/%x-%j.out" "${dependency_args[@]}")
    id=${id%%;*}
    job_ids[$key]=$id
    [[ $stage != evaluate ]] || evaluation_ids+=("$id")
    printf '%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "$id" "$member" >> "$OURS_BATCH/submitted.tsv"
    printf 'Submitted %s %s\n' "$id" "$key"
done < "$OURS_BATCH/jobs.tsv"
printf 'Submitted %s jobs. Results: %s\n' "${#job_ids[@]}" "$OURS_BATCH"
