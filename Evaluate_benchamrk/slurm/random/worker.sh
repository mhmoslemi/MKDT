#!/bin/bash
set -euo pipefail
BATCH=$(readlink -f -- "${1:?batch directory}")
account=${2:?account}
source "$BATCH/environment.sh"
jq -e '.complete == true and (.cases | length == 18)' "$BATCH/validation.json" >/dev/null
case "$account" in aip-boyuwang|aip-yiweilu) ;; *) exit 2 ;; esac
[[ $account == "$SLURM_JOB_ACCOUNT" ]] || { echo 'Account mismatch' >&2; exit 2; }
[[ ${SLURM_ARRAY_TASK_ID:-} =~ ^[0-9]+$ ]] || exit 2
row=$(awk -F '\t' -v index="$SLURM_ARRAY_TASK_ID" 'NR>1 && $1==index {print; found++} END {if(found!=1)exit 1}' "$BATCH/tasks_${account}.tsv")
IFS=$'\t' read -r task_id source_dataset model subset ssl seed runner <<< "$row"
printf 'TASK source=%s model=%s selection=random size=%s ssl=%s seed=%s original_runner=%s\n' \
    "$source_dataset" "$model" "$subset" "$ssl" "$seed" "$runner"
exec python -u "$BATCH/run_seed.py" \
    --source "$source_dataset" --model "$model" --method random --subset "$subset" \
    --ssl "$ssl" --seeds "$seed" --workers 4 --threads 4 \
    --data-root "$BENCH_DATA_ROOT" --output-root "$BATCH/results"
