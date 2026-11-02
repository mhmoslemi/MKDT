#!/bin/bash
set -euo pipefail
cd /scratch/mmoslem3/MKDT
task_dir=Evaluate_benchamrk/mipa_column_ablation
mkdir -p "$task_dir/logs"
exec 9>"$task_dir/submit.lock"
flock -n 9 || exit 1
name=mipa_column_C10_l0p1
if [[ -s "$task_dir/runs/lambda0.1_tc0.2/evaluation/complete.json" ]]; then
    echo "SKIP completed: $name"
    exit 0
fi
active=$(squeue -h -u "$USER" -o '%j')
if [[ $'\n'"$active"$'\n' == *$'\n'"$name"$'\n'* ]]; then
    echo "SKIP active: $name"
    exit 0
fi
job=$(sbatch --parsable --job-name="$name" --account=aip-yiweilu --gres=gpu:l40s:1 --cpus-per-task=1 --mem=6G --time=00:20:00 --output="$task_dir/logs/%x-%j.out" --error="$task_dir/logs/%x-%j.err" "$task_dir/job.sh")
printf '%s\t%s\n' "$name" "$job" | tee -a "$task_dir/submitted.tsv"
