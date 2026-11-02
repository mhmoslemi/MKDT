#!/bin/bash
set -euo pipefail
cd /scratch/mmoslem3/MKDT
task_dir=Evaluate_benchamrk/mipa_duplicate_ablation
mkdir -p "$task_dir/logs"
exec 9>"$task_dir/submit.lock"
flock -n 9 || exit 1
active=$(squeue -h -u "$USER" -o '%j')
for weight in 1 10 100 1000; do
    name="mipa_duplicate_C10_l${weight}"
    if [[ -s "$task_dir/runs/lambda${weight}_rho0.95/evaluation/complete.json" ]]; then
        echo "SKIP completed: $name"
        continue
    fi
    if [[ $'\n'"$active"$'\n' == *$'\n'"$name"$'\n'* ]]; then
        echo "SKIP active: $name"
        continue
    fi
    job=$(sbatch --parsable --job-name="$name" --account=aip-yiweilu --gres=gpu:l40s:1 --cpus-per-task=1 --mem=6G --time=00:18:00 --output="$task_dir/logs/%x-%j.out" --error="$task_dir/logs/%x-%j.err" "$task_dir/job.sh" "$weight")
    printf '%s\t%s\n' "$name" "$job" | tee -a "$task_dir/submitted.tsv"
done
