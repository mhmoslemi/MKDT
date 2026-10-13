#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --gres=gpu:l40s:1

set -euo pipefail

[[ $# -eq 6 ]] || {
    echo "Usage: $0 <pretrain|evaluate> <source> <ssl> <epochs> <config> <targets>" >&2
    exit 2
}
source "$PROJECT_ROOT/benchmarks/full_data_refresh/environment.sh"

phase=$1
source_dataset=$2
ssl=$3
ssl_epochs=$4
config=$5
targets_csv=$6
model=ResNet18
encoder_root="$FULL_BATCH/runs/$config/encoders"

case "$phase" in
    pretrain)
        output="$encoder_root/shard_shared_0_job_${SLURM_JOB_ID}"
        srun --unbuffered python3 benchmarks/benchmark_full_refresh.py pretrain \
            --source "$source_dataset" \
            --source-cache "$TARGET_CACHE_ROOT/$source_dataset.pt" \
            --output "$output" --model "$model" --ssl-method "$ssl" \
            --ssl-epochs "$ssl_epochs" --seed-start 0 --runs 1 \
            --ssl-log-every 10
        ;;
    evaluate)
        IFS=',' read -r -a targets <<< "$targets_csv"
        for target in "${targets[@]}"; do
            for labels in 1 5; do
                if [[ $labels == 1 ]]; then probe_epochs=500; else probe_epochs=300; fi
                output="$FULL_BATCH/runs/$config/evaluations/${target}_lbl${labels}/job_${SLURM_JOB_ID}"
                mkdir -p "$(dirname "$output")"
                srun --unbuffered python3 benchmarks/benchmark_full_refresh.py evaluate \
                    --source "$source_dataset" --encoder-dir "$encoder_root" \
                    --reuse-encoder-seed 0 \
                    --target "$target" --target-cache "$TARGET_CACHE_ROOT/$target.pt" \
                    --output "$output" --model "$model" --ssl-method "$ssl" \
                    --ssl-epochs "$ssl_epochs" --label-percentage "$labels" \
                    --probe-epochs "$probe_epochs" --runs 15
            done
        done
        ;;
    *)
        echo "Invalid phase: $phase" >&2
        exit 2
        ;;
esac
