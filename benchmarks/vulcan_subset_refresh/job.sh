#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --gres=gpu:l40s:1

set -euo pipefail

[[ $# -ge 2 ]] || {
    echo "Usage: $0 <prepare|pretrain|evaluate> <source> [...]" >&2
    exit 2
}

source "$SUBSET_CODE_ROOT/benchmarks/vulcan_subset_refresh/environment.sh"

phase=$1
source_dataset=$2
source_cache="$TARGET_CACHE_ROOT/$source_dataset.pt"

case "$phase" in
    prepare)
        [[ $# -eq 2 ]] || { echo "prepare takes only a source dataset" >&2; exit 2; }
        : "${SLURM_ARRAY_TASK_ID:?prepare must be an array job}"
        output="$SUBSET_BATCH/selections/$source_dataset"
        srun --unbuffered python3 benchmarks/benchmark_subset_refresh.py prepare \
            --source "$source_dataset" --source-cache "$source_cache" \
            --output "$output" --seed-start "$SLURM_ARRAY_TASK_ID" --runs 1
        ;;
    pretrain)
        [[ $# -eq 8 ]] || {
            echo "pretrain requires source method model ssl subset epochs config" >&2
            exit 2
        }
        : "${SLURM_ARRAY_TASK_ID:?pretrain must be an array job}"
        method=$3
        model=$4
        ssl=$5
        subset=$6
        epochs=$7
        config=$8
        output="$SUBSET_BATCH/runs/$config/encoders"
        srun --unbuffered python3 benchmarks/benchmark_subset_refresh.py pretrain \
            --source "$source_dataset" --source-cache "$source_cache" \
            --output "$output" \
            --selection-dir "$SUBSET_BATCH/selections/$source_dataset" \
            --method "$method" --model "$model" --ssl-method "$ssl" \
            --subset-percentage "$subset" --ssl-epochs "$epochs" \
            --seed-start "$SLURM_ARRAY_TASK_ID" --runs 1 \
            --ssl-log-every "${SUBSET_SSL_LOG_EVERY:-0}"
        ;;
    evaluate)
        [[ $# -eq 10 ]] || {
            echo "evaluate requires source method model ssl subset epochs config probe1 probe5" >&2
            exit 2
        }
        method=$3
        model=$4
        ssl=$5
        subset=$6
        epochs=$7
        config=$8
        probe1=$9
        probe5=${10}
        encoder_dir="$SUBSET_BATCH/runs/$config/encoders"
        for labels in 1 5; do
            if [[ $labels == 1 ]]; then
                probe_epochs=$probe1
            else
                probe_epochs=$probe5
            fi
            output="$SUBSET_BATCH/runs/$config/evaluations/${source_dataset}_lbl${labels}/job_$SLURM_JOB_ID"
            mkdir -p "$(dirname "$output")"
            srun --unbuffered python3 benchmarks/benchmark_subset_refresh.py evaluate \
                --source "$source_dataset" --source-cache "$source_cache" \
                --encoder-dir "$encoder_dir" --target "$source_dataset" \
                --target-cache "$source_cache" --output "$output" \
                --method "$method" --model "$model" --ssl-method "$ssl" \
                --subset-percentage "$subset" --ssl-epochs "$epochs" \
                --label-percentage "$labels" --probe-epochs "$probe_epochs" \
                --runs 10
        done
        ;;
    *)
        echo "Unknown phase: $phase" >&2
        exit 2
        ;;
esac
