#!/bin/bash
# Submit one baseline and 21 one-at-a-time variants (22 jobs, up to 66 GPU-hours).
# Preview without submitting: bash submit_mina_sweep.sh --dry-run
# Compare the mean downstream accuracy at iteration 2000 for completed jobs.
# This screens individual settings; it does not test interactions or seed variance.
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
export PROJECT_ROOT
dry_run=0
if [[ ${1:-} == --dry-run && $# == 1 ]]; then
    dry_run=1
elif (( $# )); then
    echo "Usage: bash $0 [--dry-run]" >&2
    exit 2
fi

cd "$PROJECT_ROOT"
jobs=("$PROJECT_ROOT"/benchmarks/mina_sweep/jobs/*.sh)
[[ -f ${jobs[0]} ]] || { echo 'No sweep jobs found.' >&2; exit 1; }
if (( ! dry_run )); then
    command -v sbatch >/dev/null
    [[ -f ${VENV_ACTIVATE:-$HOME/ENV/bin/activate} ]] || { echo 'Python environment is missing.' >&2; exit 1; }
    mkdir -p "$PROJECT_ROOT/benchmarks/mina_sweep/logs"
fi

for job in "${jobs[@]}"; do
    command_args=(sbatch --parsable --chdir="$PROJECT_ROOT"
        --output="$PROJECT_ROOT/benchmarks/mina_sweep/logs/%x-%j.out" "$job")
    if (( dry_run )); then
        printf '%q ' "${command_args[@]}"
        printf '\n'
    else
        job_id=$("${command_args[@]}")
        printf '%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "$job_id" "$(basename "$job" .sh)" >> "$PROJECT_ROOT/benchmarks/mina_sweep/submitted.tsv"
        printf 'Submitted %s: %s\n' "$(basename "$job" .sh)" "$job_id"
    fi
done
printf '%s jobs; each has a 3-hour limit.\n' "${#jobs[@]}"
