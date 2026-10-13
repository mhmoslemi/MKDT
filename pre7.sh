#!/bin/bash
set -euo pipefail

: "${SLURM_JOB_ID:?Run this inside a GPU salloc allocation.}"
export PROJECT_ROOT=/scratch/mmoslem3/MKDT
export MKDT_MULTI_BATCH="$PROJECT_ROOT/benchmarks/mkdt_multidataset/results/batch_20261010T160046Z_Y6dqZS"
export MKDT_CODE_ROOT="$MKDT_MULTI_BATCH/code"
export MKDT_SYNTH_ROOT="$MKDT_MULTI_BATCH/synthetic"
export TARGET_CACHE_ROOT="$MKDT_MULTI_BATCH/targets"
export DATA_ROOT=/scratch/mmoslem3/data

source_dataset=CIFAR10
generator=VGG11
model=VGG11
size=2
ssl=simclr
epochs=4000
log_every=800
original_job=1599518
evaluation_job=1599519
task="${source_dataset}_${generator}_${model}_size${size}_${ssl}"
images="$MKDT_SYNTH_ROOT/generated/$generator/$source_dataset/${source_dataset}_${size}pct/images_5000.pt"
encoder="$MKDT_MULTI_BATCH/runs/$task/encoders"
log="$MKDT_MULTI_BATCH/logs/mkpre_${task}-salloc-${SLURM_JOB_ID}.out"
started_epoch=$(date +%s)

report_exit() {
    status=$?
    elapsed=$(($(date +%s) - started_epoch))
    printf 'TOTAL %02d:%02d:%02d status=%d\n' "$((elapsed / 3600))" "$(((elapsed % 3600) / 60))" "$((elapsed % 60))" "$status"
}
trap report_exit EXIT

progress_filter() {
    awk -v task="$task" -v started="$started_epoch" '
        function elapsed(now, seconds) { seconds = now - started; return sprintf("%02d:%02d:%02d", int(seconds / 3600), int(seconds % 3600 / 60), seconds % 60) }
        /^SSL epoch=/ { units++; printf "[%s] [%s] %d%% | %s\n", elapsed(systime()), task, units * 2, $0; fflush(); next }
        /^SKIP pretrain seed=/ { units += 5; printf "[%s] [%s] %d%% | %s\n", elapsed(systime()), task, units * 2, $0; fflush(); next }
        { printf "[%s] %s\n", elapsed(systime()), $0; fflush() }
    '
}

echo "START $(date -Is) $task; log: $log"
scancel "$original_job" 2>/dev/null || true
source "$PROJECT_ROOT/benchmarks/mkdt_multidataset/environment.sh"
srun --unbuffered python3 benchmarks/benchmark_mkdt_multidataset.py pretrain \
    --source "$source_dataset" --generator "$generator" --images "$images" \
    --output "$encoder" --model "$model" --ssl-method "$ssl" \
    --subset-percentage "$size" --ssl-epochs "$epochs" \
    --ssl-log-every "$log_every" 2>&1 | progress_filter | tee "$log"
[[ -f "$encoder/summary.json" ]] && rg -q '"complete": true' "$encoder/summary.json" || { echo "Incomplete encoder summary: $encoder/summary.json" >&2; exit 1; }
scontrol release "$evaluation_job"
echo "END $(date -Is) $task; released evaluation job $evaluation_job"
