#!/bin/bash
set -euo pipefail
cd /scratch/mmoslem3/MKDT
task_dir=Evaluate_benchamrk/mipa_table_initialization
mkdir -p "$task_dir/logs"
exec 9>"$task_dir/submit_noise10x.lock"
flock -n 9 || exit 1
active=$(squeue -h -u "$USER" -o '%j')
for percentage in 0.2 1; do
    tag=${percentage/./p}
    name="mipanoise_C10_p${tag}_x10"
    if [[ -s "$task_dir/runs/noise10x/pt$percentage/evaluation/complete.json" ]]; then
        echo "SKIP completed: $name"
        continue
    fi
    if [[ $'\n'"$active"$'\n' == *$'\n'"$name"$'\n'* ]]; then
        echo "SKIP active: $name"
        continue
    fi
    dependencies=()
    parent=$(squeue -h -u "$USER" -n "mipanoise_C10_p${tag}_x3" -o '%i')
    if [[ -n "$parent" ]]; then
        [[ "$parent" =~ ^[0-9]+$ ]] || { echo 'More than one active parent; resolve duplicate 3x jobs first.' >&2; exit 1; }
        dependencies+=(--dependency="afterany:$parent")
    fi
    case "$percentage" in
        0.2) limit=01:05:00 ;;
        1) limit=02:00:00 ;;
    esac
    job=$(sbatch --parsable --job-name="$name" --account=aip-yiweilu --gres=gpu:l40s:1 --cpus-per-task=1 --mem=6G --time="$limit" "${dependencies[@]}" --output="$task_dir/logs/%x-%j.out" --error="$task_dir/logs/%x-%j.err" "$task_dir/job_noise10x.sh" "$percentage")
    printf '%s\t%s\t%s\t%s\n' "$name" "$job" "$limit" "${dependencies[*]}" | tee -a "$task_dir/submitted_noise10x.tsv"
done
