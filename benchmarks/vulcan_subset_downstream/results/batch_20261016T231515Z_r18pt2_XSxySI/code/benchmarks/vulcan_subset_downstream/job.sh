#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --gres=gpu:l40s:1
#SBATCH --no-requeue

set -euo pipefail

: "${SLURM_ARRAY_TASK_ID:?This worker must run as a Slurm array task.}"
: "${DOWNSTREAM_BATCH:?}"
: "${DOWNSTREAM_CODE_ROOT:?}"
: "${TARGET_CACHE_ROOT:?}"

source "$DOWNSTREAM_CODE_ROOT/benchmarks/vulcan_subset_downstream/environment.sh"

manifest="$DOWNSTREAM_BATCH/configs.tsv"
line_number=$((SLURM_ARRAY_TASK_ID + 2))
row=$(sed -n "${line_number}p" "$manifest")
[[ -n $row ]] || {
    echo "No manifest row for array task $SLURM_ARRAY_TASK_ID" >&2
    exit 1
}

IFS=$'\t' read -r config source method model ssl subset epochs encoder_dir <<< "$row"
[[ -d $encoder_dir ]] || { echo "Missing encoder directory: $encoder_dir" >&2; exit 1; }
for seed in 000 001 002 003 004 005 006 007 008 009; do
    [[ -s "$encoder_dir/seed_${seed}.pt" ]] || {
        echo "Missing checkpoint: $encoder_dir/seed_${seed}.pt" >&2
        exit 1
    }
done

case "$source" in
    CIFAR10)
        targets=(CIFAR100 Aircraft CUB2011 Dogs Flowers TinyImageNet)
        ;;
    CIFAR100)
        targets=(TinyImageNet CIFAR10 Aircraft CUB2011 Dogs Flowers)
        ;;
    *)
        echo "Unsupported source dataset: $source" >&2
        exit 1
        ;;
esac

source_cache="$TARGET_CACHE_ROOT/$source.pt"
for target in "${targets[@]}"; do
    target_cache="$TARGET_CACHE_ROOT/$target.pt"
    [[ -f $target_cache ]] || { echo "Missing target cache: $target_cache" >&2; exit 1; }
    for labels in 1 5; do
        probe_epochs=$(jq -er --arg target "$target" --arg labels "$labels" \
            '.evaluation_by_dataset[$target].linear_probe_epochs_by_label_percentage[$labels]' \
            "$DOWNSTREAM_BATCH/training_epochs.json")
        output="$DOWNSTREAM_BATCH/runs/$config/evaluations/${target}_lbl${labels}/job_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
        mkdir -p "$(dirname "$output")"
        srun --unbuffered python3 benchmarks/benchmark_subset_refresh.py evaluate \
            --source "$source" --source-cache "$source_cache" \
            --encoder-dir "$encoder_dir" --target "$target" \
            --target-cache "$target_cache" --output "$output" \
            --method "$method" --model "$model" --ssl-method "$ssl" \
            --subset-percentage "$subset" --ssl-epochs "$epochs" \
            --label-percentage "$labels" --probe-epochs "$probe_epochs" \
            --runs 10
    done
done

echo "DOWNSTREAM COMPLETE config=$config targets=6 label_budgets=2 runs=10" \
    "probe_protocol=frozen_encoder"
