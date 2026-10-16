#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --gres=gpu:l40s:1
#SBATCH --no-requeue

set -euo pipefail

: "${SLURM_ARRAY_TASK_ID:?This worker must be submitted as an array job.}"
: "${NO_PRETRAIN_BATCH:?}"
: "${NO_PRETRAIN_CODE_ROOT:?}"
: "${TARGET_CACHE_ROOT:?}"

source "$NO_PRETRAIN_CODE_ROOT/benchmarks/vulcan_no_pretrain/environment.sh"

manifest="$NO_PRETRAIN_BATCH/configs.tsv"
row_number=$((SLURM_ARRAY_TASK_ID + 2))
row=$(awk -F '\t' -v row_number="$row_number" 'NR == row_number { print $0 }' "$manifest")
[[ -n $row ]] || {
    echo "No manifest row for array task $SLURM_ARRAY_TASK_ID" >&2
    exit 1
}
IFS=$'\t' read -r task_id config target model <<< "$row"
[[ $task_id == "$SLURM_ARRAY_TASK_ID" ]] || {
    echo "Manifest/task mismatch: task=$SLURM_ARRAY_TASK_ID row=$task_id" >&2
    exit 1
}

case "$target" in
    CIFAR10|CIFAR100|Aircraft|CUB2011|Dogs|Flowers|TinyImageNet) ;;
    *) echo "Unexpected target: $target" >&2; exit 1 ;;
esac
case "$model" in
    ConvNet|VGG11|ResNet18) ;;
    *) echo "Unexpected model: $model" >&2; exit 1 ;;
esac

target_cache="$TARGET_CACHE_ROOT/$target.pt"
[[ -f $target_cache ]] || { echo "Missing target cache: $target_cache" >&2; exit 1; }

for labels in 1 5; do
    probe_epochs=$(jq -er --arg target "$target" --arg labels "$labels" \
        '.evaluation_by_dataset[$target].no_pretraining_linear_probe_epochs_by_label_percentage[$labels]' \
        "$NO_PRETRAIN_BATCH/training_epochs.json")
    if [[ $labels == 1 ]]; then
        [[ $probe_epochs == 1000 ]] || { echo "$target 1% must use 1000 probe epochs" >&2; exit 1; }
    else
        [[ $probe_epochs == 500 ]] || { echo "$target 5% must use 500 probe epochs" >&2; exit 1; }
    fi

    result_id="${target}_${model}_no_pretrain_lbl${labels}"
    output="$NO_PRETRAIN_BATCH/evaluations/$result_id/job_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
    mkdir -p "$(dirname "$output")"

    echo "START task=$task_id config=$config target=$target model=$model labels=$labels probe_epochs=$probe_epochs"
    srun --unbuffered python3 benchmarks/benchmark_downstream.py evaluate \
        --device cuda \
        --target "$target" \
        --target-cache "$target_cache" \
        --output "$output" \
        --method no_pretrain \
        --model "$model" \
        --ssl-method none \
        --subset-percentage 0 \
        --ssl-epochs 0 \
        --label-percentage "$labels" \
        --label-policy random \
        --probe-epochs "$probe_epochs" \
        --runs 10
done

echo "COMPLETE task=$task_id config=$config (both label budgets)"
