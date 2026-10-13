#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --gres=gpu:l40s:1
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
[[ $# -ge 6 ]] || { echo "Usage: $0 <pretrain|evaluate> <source> <model> <ssl> <ssl_epochs> <config> [...]" >&2; exit 2; }
source "$PROJECT_ROOT/benchmarks/full_data_refresh/environment.sh"
phase=$1
source_dataset=$2
model=$3
ssl=$4
ssl_epochs=$5
config=$6
encoder_root="$FULL_BATCH/runs/$config/encoders"

case "$phase" in
    pretrain)
        [[ $# -eq 8 ]] || { echo "pretrain additionally requires seed_start and runs" >&2; exit 2; }
        seed_start=$7
        runs=$8
        output="$encoder_root/shard_${seed_start}_$((seed_start + runs - 1))"
        srun --unbuffered python3 benchmarks/benchmark_full_refresh.py pretrain \
            --source "$source_dataset" --source-cache "$TARGET_CACHE_ROOT/$source_dataset.pt" \
            --output "$output" --model "$model" --ssl-method "$ssl" \
            --ssl-epochs "$ssl_epochs" --seed-start "$seed_start" --runs "$runs" \
            --ssl-log-every "${FULL_SSL_LOG_EVERY:-0}"
        ;;
    evaluate)
        [[ $# -eq 7 ]] || { echo "evaluate additionally requires comma-separated targets" >&2; exit 2; }
        IFS=',' read -r -a targets <<< "$7"
        eval_progress_total=$((${#targets[@]} * 2 * 10))
        eval_progress_completed=0
        echo "PROGRESS: 0%"
        for target in "${targets[@]}"; do
            for labels in 1 5; do
                if [[ $labels == 1 ]]; then probe_epochs=500; else probe_epochs=300; fi
                output="$FULL_BATCH/runs/$config/evaluations/${target}_lbl${labels}/job_$SLURM_JOB_ID"
                mkdir -p "$(dirname "$output")"
                export FULL_EVAL_PROGRESS_BASE=$eval_progress_completed
                export FULL_EVAL_PROGRESS_TOTAL=$eval_progress_total
                srun --unbuffered python3 benchmarks/benchmark_full_refresh.py evaluate \
                    --source "$source_dataset" --encoder-dir "$encoder_root" \
                    --target "$target" --target-cache "$TARGET_CACHE_ROOT/$target.pt" \
                    --output "$output" --model "$model" --ssl-method "$ssl" \
                    --ssl-epochs "$ssl_epochs" --label-percentage "$labels" \
                    --probe-epochs "$probe_epochs" --runs 10
                eval_progress_completed=$((eval_progress_completed + 10))
            done
        done
        ;;
    *) echo "Invalid phase: $phase" >&2; exit 2 ;;
esac
