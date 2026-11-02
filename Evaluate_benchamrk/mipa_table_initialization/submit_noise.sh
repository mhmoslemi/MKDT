#!/bin/bash
set -euo pipefail
cd /scratch/mmoslem3/MKDT
task_dir=Evaluate_benchamrk/mipa_table_initialization
mkdir -p "$task_dir/logs"
exec 9>"$task_dir/submit_noise.lock"
flock -n 9 || exit 1
active=$(squeue -h -u "$USER" -o '%j')
for multiplier in 1 3; do
    for percentage in 0.2 1; do
        variant="noise${multiplier}x"
        name="mipanoise_C10_p${percentage/./p}_x${multiplier}"
        if [[ -s "$task_dir/runs/$variant/pt$percentage/evaluation/complete.json" ]]; then
            echo "SKIP completed: $name"
            continue
        fi
        if [[ $'\n'"$active"$'\n' == *$'\n'"$name"$'\n'* ]]; then
            echo "SKIP active: $name"
            continue
        fi
        case "$percentage:$multiplier" in
            0.2:1) limit=00:20:00 ;;
            1:1) limit=00:35:00 ;;
            0.2:3) limit=00:35:00 ;;
            1:3) limit=01:05:00 ;;
        esac
        job=$(sbatch --parsable --job-name="$name" --account=aip-yiweilu --gres=gpu:l40s:1 --cpus-per-task=1 --mem=6G --time="$limit" --output="$task_dir/logs/%x-%j.out" --error="$task_dir/logs/%x-%j.err" "$task_dir/job_noise.sh" "$variant" "$percentage")
        printf '%s\t%s\t%s\n' "$name" "$job" "$limit" | tee -a "$task_dir/submitted_noise.tsv"
    done
done
