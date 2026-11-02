#!/bin/bash
set -euo pipefail
cd /scratch/mmoslem3/MKDT
task_dir=Evaluate_benchamrk/mipa_table_initialization
mkdir -p "$task_dir/logs"
exec 9>"$task_dir/submit.lock"
flock -n 9 || exit 1
active=$(squeue -h -u "$USER" -o '%j')
for variant in kmeans100 iic; do
    for percentage in 0.2 1; do
        tag=${percentage/./p}
        name="mipainit_${variant}_C10_p${tag}"
        if [[ -s "$task_dir/runs/$variant/pt$percentage/evaluation/complete.json" ]]; then
            echo "SKIP completed: $name"
            continue
        fi
        if [[ $'\n'"$active"$'\n' == *$'\n'"$name"$'\n'* ]]; then
            echo "SKIP active: $name"
            continue
        fi
        case "$percentage" in
            0.2) minutes=20 ;;
            1) minutes=35 ;;
        esac
        if [[ "$variant" == kmeans100 ]]; then minutes=$((minutes + 3)); fi
        printf -v limit '%02d:%02d:00' "$((minutes / 60))" "$((minutes % 60))"
        job=$(sbatch --parsable --job-name="$name" --account=aip-yiweilu --gres=gpu:l40s:1 --cpus-per-task=1 --mem=6G --time="$limit" --output="$task_dir/logs/%x-%j.out" --error="$task_dir/logs/%x-%j.err" "$task_dir/job.sh" "$variant" "$percentage")
        printf '%s\t%s\t%s\n' "$name" "$job" "$limit" | tee -a "$task_dir/submitted.tsv"
    done
done
