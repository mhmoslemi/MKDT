#!/bin/bash
set -euo pipefail
cd /scratch/mmoslem3/MKDT
task_dir=Evaluate_benchamrk/mipa_table_initialization
mkdir -p "$task_dir/logs"
exec 9>"$task_dir/submit_cifar100_noise5x.lock"
flock -n 9 || exit 1
name=mipanoise_C100_p0p2_x5
if [[ -s "$task_dir/runs/noise5x_cifar100/pt0.2/figure_export_complete.json" && -s OT-SSL-DD/cifar100_1percent_pairs/figure_preview.pdf && -s OT-SSL-DD/cifar100_1percent_pairs/figure_preview.png ]]; then
    echo "SKIP completed: $name"
    exit 0
fi
active=$(squeue -h -u "$USER" -o '%j')
if [[ $'\n'"$active"$'\n' == *$'\n'"$name"$'\n'* ]]; then
    echo "SKIP active: $name"
    exit 0
fi
job=$(sbatch --parsable --job-name="$name" --account=aip-boyuwang --gres=gpu:l40s:1 --cpus-per-task=1 --mem=6G --time=00:35:00 --output="$task_dir/logs/%x-%j.out" --error="$task_dir/logs/%x-%j.err" "$task_dir/job_cifar100_noise5x.sh")
printf '%s\t%s\n' "$name" "$job" | tee -a "$task_dir/submitted_cifar100_noise5x.tsv"
