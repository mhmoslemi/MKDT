#!/bin/bash
set -euo pipefail

: "${SLURM_JOB_ID:?Run this inside a GPU salloc allocation.}"

export PROJECT_ROOT=/scratch/mmoslem3/MKDT
export MKDT_MULTI_BATCH="$PROJECT_ROOT/benchmarks/mkdt_multidataset/results/batch_20261010T160046Z_Y6dqZS"
export MKDT_CODE_ROOT="$MKDT_MULTI_BATCH/code"
export MKDT_SYNTH_ROOT="$MKDT_MULTI_BATCH/synthetic"
export TARGET_CACHE_ROOT="$MKDT_MULTI_BATCH/targets"
export DATA_ROOT=/scratch/mmoslem3/data

dataset=CIFAR100
implementation=ResNet18
generator=ResNet18BN
size=1
queued_job=1576340
downstream_jobs=(1576433 1576435 1576445 1576447)
task="${dataset}_${generator}_${size}pct"
output="$MKDT_SYNTH_ROOT/generated/$generator/$dataset/${dataset}_${size}pct/images_5000.pt"
log="$MKDT_MULTI_BATCH/logs/mkdt_syn_${task}-salloc-${SLURM_JOB_ID}.out"

progress_filter() {
    tr '\r' '\n' | awk -v task="$task" '
        BEGIN { milestone = 2 }
        index($0, "Distilling:") {
            value = $0
            sub(/^.*Distilling:[[:space:]]*/, "", value)
            sub(/%.*/, "", value)
            gsub(/[[:space:]]/, "", value)
            percent = value + 0
            while (milestone <= percent && milestone <= 100) {
                printf "[%s] %d%%\n", task, milestone
                fflush()
                milestone += 2
            }
            next
        }
        { print; fflush() }
    '
}

echo "Starting $task; log: $log"
scancel "$queued_job" 2>/dev/null || true
bash "$PROJECT_ROOT/benchmarks/mkdt_multidataset/synthesis_job.sh" \
    distill "$dataset" "$implementation" "$size" "$generator" 2>&1 \
    | progress_filter | tee "$log"
[[ -f "$output" ]] || { echo "Missing final output: $output" >&2; exit 1; }

for job in "${downstream_jobs[@]}"; do
    if scontrol show job "$job" >/dev/null 2>&1; then
        scontrol release "$job"
        echo "Released downstream job $job"
    fi
done
echo "$task completed."
