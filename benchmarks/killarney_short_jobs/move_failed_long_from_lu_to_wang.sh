#!/bin/bash
set -euo pipefail

# Cancel the second-round full-data recovery graph submitted on aip-yiweilu,
# then recreate the still-needed work on aip-boyuwang. The original running
# Wang jobs (6313701/3/5/8/10 arrays) are deliberately not cancellation targets.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
RETRY_SCRIPT="$SCRIPT_DIR/resubmit_failed_long.sh"
OLD_ACCOUNT=aip-yiweilu
NEW_ACCOUNT=aip-boyuwang
OLD_JOB_IDS=(
    6315176 6315177 6315178 6315179 6315180 6315181
    6315182 6315183 6315184 6315185 6315186 6315187
    6315188 6315189 6315190 6315191 6315192
)

fail() {
    echo "FATAL: $*" >&2
    exit 1
}

[[ -s $RETRY_SCRIPT ]] || fail "missing retry launcher: $RETRY_SCRIPT"

dry_run=0
for arg in "$@"; do
    [[ $arg != --dry-run ]] || dry_run=1
done

if (( dry_run )); then
    printf 'DRY RUN: would cancel these %s jobs:' "$OLD_ACCOUNT"
    printf ' %s' "${OLD_JOB_IDS[@]}"
    printf '\nDRY RUN: replacements would use account %s\n' "$NEW_ACCOUNT"
    ACCOUNT="$NEW_ACCOUNT" exec bash "$RETRY_SCRIPT" "$@"
fi

command -v squeue >/dev/null 2>&1 || fail "squeue is not available"
command -v scancel >/dev/null 2>&1 || fail "scancel is not available"
command -v sbatch >/dev/null 2>&1 || fail "sbatch is not available"

declare -A wanted=()
declare -A active_seen=()
for job_id in "${OLD_JOB_IDS[@]}"; do
    wanted[$job_id]=1
done

active_ids=()
while IFS='|' read -r array_job_id owner account job_name state; do
    [[ -n $array_job_id ]] || continue
    [[ -v "wanted[$array_job_id]" ]] || continue
    [[ $owner == "$USER" ]] || \
        fail "job $array_job_id belongs to $owner, not $USER"
    [[ $account == "$OLD_ACCOUNT" ]] || \
        fail "job $array_job_id uses $account, not $OLD_ACCOUNT"
    [[ $job_name == kml2_* ]] || \
        fail "job $array_job_id has unexpected name $job_name"
    if [[ ! -v "active_seen[$array_job_id]" ]]; then
        active_ids+=("$array_job_id")
        active_seen[$array_job_id]=1
    fi
    printf 'Verified cancellation target: %s  %-16s  %-22s  %s\n' \
        "$array_job_id" "$account" "$job_name" "$state"
done < <(squeue -h -u "$USER" -o '%F|%u|%a|%j|%T')

if (( ${#active_ids[@]} > 0 )); then
    scancel "${active_ids[@]}"
    printf 'Cancelled %d active %s job records:' \
        "${#active_ids[@]}" "$OLD_ACCOUNT"
    printf ' %s' "${active_ids[@]}"
    printf '\n'
else
    printf 'All 17 old %s job records are already absent from the queue.\n' \
        "$OLD_ACCOUNT"
fi

# A single controller check prevents the retry launcher's duplicate-name guard
# from racing cancellation. If Slurm is still clearing jobs, rerunning this
# same script is safe: absent IDs are ignored and only remaining targets cancel.
remaining_ids=()
while IFS='|' read -r array_job_id account job_name state; do
    [[ -n $array_job_id ]] || continue
    [[ -v "wanted[$array_job_id]" ]] || continue
    remaining_ids+=("$array_job_id")
    printf 'Still leaving queue: %s  %-16s  %-22s  %s\n' \
        "$array_job_id" "$account" "$job_name" "$state" >&2
done < <(squeue -h -u "$USER" -o '%F|%a|%j|%T')

if (( ${#remaining_ids[@]} > 0 )); then
    fail "cancellation is still being processed; rerun this same script shortly"
fi

printf 'Submitting the still-missing recovery work on %s.\n' "$NEW_ACCOUNT"
ACCOUNT="$NEW_ACCOUNT" exec bash "$RETRY_SCRIPT" "$@"
