#!/bin/sh
set -u

PROJECT_ROOT=/home/mmoslem3/scratch/MKDT
DATA_ROOT=/home/mmoslem3/scratch/data
VENV_ACTIVATE=/home/mmoslem3/ENV/bin/activate
BUNDLE="$PROJECT_ROOT/benchmarks/killarney_short_jobs"
MKDT_BATCH="$PROJECT_ROOT/benchmarks/mkdt_multidataset/results/batch_20261010T160046Z_Y6dqZS"
MKDT_CODE_ROOT="$MKDT_BATCH/code"
MKDT_SYNTH_ROOT="$MKDT_BATCH/synthetic"
TARGET_CACHE_ROOT="$MKDT_BATCH/targets"

errors=0

ok() {
    printf 'OK      %s\n' "$1"
}

missing() {
    printf 'MISSING %s\n' "$1" >&2
    errors=$((errors + 1))
}

check_dir() {
    if [ -d "$1" ]; then ok "$1"; else missing "$1"; fi
}

check_file() {
    if [ -s "$1" ]; then ok "$1"; else missing "$1"; fi
}

printf '%s\n' 'Killarney short-job path preflight'
printf 'Project: %s\n\n' "$PROJECT_ROOT"

check_dir "$PROJECT_ROOT"
check_dir "$DATA_ROOT"
check_file "$VENV_ACTIVATE"
check_dir "$BUNDLE"
check_file "$BUNDLE/environment.sh"
check_file "$BUNDLE/synthesis_job.sh"
check_file "$BUNDLE/mkdt_job.sh"
check_file "$BUNDLE/submit_all.sh"

printf '\n%s\n' 'Code snapshot'
check_dir "$MKDT_CODE_ROOT"
check_file "$MKDT_CODE_ROOT/distill.py"
check_file "$MKDT_CODE_ROOT/utils.py"
check_file "$MKDT_CODE_ROOT/networks.py"
check_file "$MKDT_CODE_ROOT/reparam_module.py"
check_file "$MKDT_CODE_ROOT/benchmarks/benchmark_mkdt_multidataset.py"
check_file "$MKDT_CODE_ROOT/benchmarks/downstream_data.py"
check_file "$MKDT_CODE_ROOT/OT-SSL-DD/utils.py"
check_file "$MKDT_CODE_ROOT/OT-SSL-DD/networks.py"

printf '\n%s\n' 'Evaluation target caches'
for target in CIFAR10 CIFAR100 TinyImageNet Aircraft CUB2011 Dogs Flowers; do
    check_file "$TARGET_CACHE_ROOT/$target.pt"
done

printf '\n%s\n' 'CIFAR VGG11 synthesis inputs'
for dataset in CIFAR10 CIFAR100; do
    check_file "$MKDT_SYNTH_ROOT/work/target_rep/barlow_twins/${dataset}_target_rep_train.pt"

    buffer_dir="$MKDT_SYNTH_ROOT/work/buffers/$dataset/VGG11"
    if [ ! -d "$buffer_dir" ]; then
        missing "$buffer_dir"
    else
        missing_buffers=0
        index=0
        while [ "$index" -lt 100 ]; do
            buffer="$buffer_dir/replay_buffer_${index}.pt"
            if [ ! -s "$buffer" ]; then
                missing "$buffer"
                missing_buffers=$((missing_buffers + 1))
            fi
            index=$((index + 1))
        done
        if [ "$missing_buffers" -eq 0 ]; then
            ok "$buffer_dir (100/100 replay buffers)"
        fi
    fi

    for size in 1 2 5; do
        check_file "$MKDT_SYNTH_ROOT/work/init/$dataset/VGG11/${dataset}_barlow_twins_${size}_high_loss_indices.pkl"
    done
done

printf '\n%s\n' 'Existing CIFAR10 ConvNet encoders used by downstream-only jobs'
for size in 1 2 5; do
    for ssl in simclr barlowtwins; do
        check_file "$MKDT_BATCH/runs/CIFAR10_ConvNet_ConvNet_size${size}_${ssl}/encoders/seed_000.pt"
    done
done

printf '\n%s\n' 'Output locations'
check_dir "$MKDT_BATCH/runs"
check_dir "$MKDT_SYNTH_ROOT"
if [ -w "$MKDT_BATCH/runs" ]; then
    ok "$MKDT_BATCH/runs is writable"
else
    missing "$MKDT_BATCH/runs is not writable"
fi
if [ -w "$MKDT_SYNTH_ROOT" ]; then
    ok "$MKDT_SYNTH_ROOT is writable"
else
    missing "$MKDT_SYNTH_ROOT is not writable"
fi

printf '\n'
if [ "$errors" -eq 0 ]; then
    printf '%s\n' 'PREFLIGHT PASSED: every required path is present and outputs are writable.'
    exit 0
fi

printf 'PREFLIGHT FAILED: %d required path(s) failed validation.\n' "$errors" >&2
exit 1
