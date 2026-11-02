#!/bin/bash
set -euo pipefail
cd /scratch/mmoslem3/MKDT
task_dir=Evaluate_benchamrk/mipa_tsne
mkdir -p "$task_dir/logs"
exec 9>"$task_dir/submit.lock"
flock -n 9 || exit 1
active=$(squeue -h -u "$USER" -o '%j' | awk '$0 == "mipa_tsne_features" || $0 == "mipa_tsne_plot"')
if [[ -n $active ]]; then
    echo 'These t-SNE jobs are already queued/running; no duplicates submitted.'
    exit 0
fi
extract=$(sbatch --parsable --job-name=mipa_tsne_features --account=aip-yiweilu --gres=gpu:l40s:1 --cpus-per-task=1 --mem=6G --time=00:03:00 --output="$task_dir/logs/features-%j.out" --error="$task_dir/logs/features-%j.err" "$task_dir/job.sh" extract)
extract=${extract%%;*}
embed=$(sbatch --parsable --job-name=mipa_tsne_plot --account=aip-yiweilu --cpus-per-task=4 --mem=4G --time=00:15:00 --array=0-5 --dependency="afterok:$extract" --kill-on-invalid-dep=yes --output="$task_dir/logs/plot-%A_%a.out" --error="$task_dir/logs/plot-%A_%a.err" "$task_dir/job.sh" embed)
echo "GPU feature extraction: $extract; six CPU comparisons: $embed"
echo 'Output: /scratch/mmoslem3/MKDT/OT-SSL-DD/tsne_mipa_random/{CIFAR10,CIFAR100}/pt{0.2,1,2}/'
