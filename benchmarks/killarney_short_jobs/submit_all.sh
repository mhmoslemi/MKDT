#!/bin/sh
set -eu

PROJECT_ROOT=/home/mmoslem3/scratch/MKDT
DATA_ROOT=/home/mmoslem3/scratch/data
BUNDLE="$PROJECT_ROOT/benchmarks/killarney_short_jobs"
MKDT_BATCH="$PROJECT_ROOT/benchmarks/mkdt_multidataset/results/batch_20261010T160046Z_Y6dqZS"
MKDT_CODE_ROOT="$MKDT_BATCH/code"
MKDT_SYNTH_ROOT="$MKDT_BATCH/synthetic"
TARGET_CACHE_ROOT="$MKDT_BATCH/targets"
PART=${SUBMIT_PART:-}
case "$PART" in
    1|2) ;;
    *)
        echo 'Run submit_part1.sh or submit_part2.sh; submit_all.sh does not run directly.' >&2
        exit 2
        ;;
esac
LOG_ROOT="$BUNDLE/logs/part$PART"
MANIFEST="$BUNDLE/submitted_part${PART}.tsv"
WANG_ACCOUNT=aip-boyuwang
LU_ACCOUNT=aip-yiweilu
DRY_RUN=${DRY_RUN:-0}

fail() {
    echo "FATAL: $*" >&2
    exit 1
}

if [ "$DRY_RUN" != 1 ]; then
    command -v sbatch >/dev/null 2>&1 || fail "sbatch is not available"
fi
[ -d "$PROJECT_ROOT" ] || fail "missing project root: $PROJECT_ROOT"
[ -d "$DATA_ROOT" ] || fail "missing data root: $DATA_ROOT"
[ -f /home/mmoslem3/ENV/bin/activate ] || fail "missing /home/mmoslem3/ENV/bin/activate"
[ -d "$MKDT_CODE_ROOT" ] || fail "missing code snapshot: $MKDT_CODE_ROOT"
[ -f "$BUNDLE/mkdt_job.sh" ] || fail "missing benchmark worker"
if [ "$DRY_RUN" != 1 ]; then
    [ ! -s "$MANIFEST" ] || fail "$MANIFEST already contains submissions"
fi

for target in CIFAR10 CIFAR100 TinyImageNet Aircraft CUB2011 Dogs Flowers; do
    [ -f "$TARGET_CACHE_ROOT/$target.pt" ] || fail "missing target cache: $TARGET_CACHE_ROOT/$target.pt"
done

if [ "$PART" = 1 ]; then
    for size in 1 2 5; do
        for ssl in simclr barlowtwins; do
            encoder="$MKDT_BATCH/runs/CIFAR10_ConvNet_ConvNet_size${size}_${ssl}/encoders/seed_000.pt"
            [ -s "$encoder" ] || fail "missing existing CIFAR10 ConvNet encoder: $encoder"
        done
    done
else
    [ -f "$BUNDLE/synthesis_job.sh" ] || fail "missing synthesis worker"
    for dataset in CIFAR10 CIFAR100; do
        target_rep="$MKDT_SYNTH_ROOT/work/target_rep/barlow_twins/${dataset}_target_rep_train.pt"
        [ -s "$target_rep" ] || fail "missing target representations: $target_rep"
        index=0
        while [ "$index" -lt 100 ]; do
            buffer="$MKDT_SYNTH_ROOT/work/buffers/$dataset/VGG11/replay_buffer_${index}.pt"
            [ -s "$buffer" ] || fail "missing VGG11 expert buffer: $buffer"
            index=$((index + 1))
        done
        for size in 1 2 5; do
            init="$MKDT_SYNTH_ROOT/work/init/$dataset/VGG11/${dataset}_barlow_twins_${size}_high_loss_indices.pkl"
            [ -s "$init" ] || fail "missing VGG11 initialization: $init"
        done
    done
fi

if [ "$DRY_RUN" != 1 ]; then
    mkdir -p "$LOG_ROOT"
    cp "$PROJECT_ROOT/benchmarks/benchmark_mkdt_multidataset.py" \
        "$MKDT_CODE_ROOT/benchmarks/benchmark_mkdt_multidataset.py"
    if [ "$PART" = 2 ]; then
        cp "$PROJECT_ROOT/distill.py" "$MKDT_CODE_ROOT/distill.py"
    fi
    printf 'submitted_utc\tjob_id\taccount\tclass\tname\ttime\tmem\tdependency\n' > "$MANIFEST"
fi

gpu_index=0
LAST_JOB_ID=
submit_gpu() {
    class=$1
    name=$2
    limit=$3
    memory=$4
    dependency=$5
    script=$6
    shift 6

    gpu_index=$((gpu_index + 1))
    account=$WANG_ACCOUNT
    if [ $((gpu_index % 10)) -eq 0 ]; then
        account=$LU_ACCOUNT
    fi

    if [ "$DRY_RUN" = 1 ]; then
        LAST_JOB_ID="DRY${gpu_index}"
        echo "Would submit $name ($limit, $memory, $account, dependency=${dependency:-none})"
        return
    fi

    if [ -n "$dependency" ]; then
        submitted=$(sbatch --parsable \
            --account="$account" \
            --job-name="$name" \
            --nodes=1 \
            --ntasks=1 \
            --cpus-per-task=1 \
            --gpus-per-node=1 \
            --time="$limit" \
            --mem="$memory" \
            --chdir="$PROJECT_ROOT" \
            --output="$LOG_ROOT/%x-%j.out" \
            --dependency="$dependency" \
            --kill-on-invalid-dep=yes \
            "$script" "$@")
    else
        submitted=$(sbatch --parsable \
            --account="$account" \
            --job-name="$name" \
            --nodes=1 \
            --ntasks=1 \
            --cpus-per-task=1 \
            --gpus-per-node=1 \
            --time="$limit" \
            --mem="$memory" \
            --chdir="$PROJECT_ROOT" \
            --output="$LOG_ROOT/%x-%j.out" \
            "$script" "$@")
    fi

    LAST_JOB_ID=${submitted%%;*}
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$(date -u +%FT%TZ)" "$LAST_JOB_ID" "$account" "$class" \
        "$name" "$limit" "$memory" "$dependency" >> "$MANIFEST"
    echo "Submitted $name: $LAST_JOB_ID"
}

MKDT_JOB="$BUNDLE/mkdt_job.sh"

if [ "$PART" = 1 ]; then
    for size in 1 2 5; do
        case "$size" in
            1) sim_epochs=5000; bt_epochs=10000 ;;
            2) sim_epochs=4000; bt_epochs=8000 ;;
            5) sim_epochs=3000; bt_epochs=6000 ;;
        esac
        submit_gpu mkdt_evaluate "kil_p1_C10_Conv_down_${size}_simclr" 01:00:00 4G "" \
            "$MKDT_JOB" evaluate CIFAR10 ConvNet ConvNet "$size" simclr "$sim_epochs" downstream
        submit_gpu mkdt_evaluate "kil_p1_C10_Conv_down_${size}_barlow" 01:00:00 4G "" \
            "$MKDT_JOB" evaluate CIFAR10 ConvNet ConvNet "$size" barlowtwins "$bt_epochs" downstream
    done

    if [ "$DRY_RUN" = 1 ]; then
        echo "Validated $gpu_index Part 1 jobs; nothing was submitted."
    else
        echo "Submitted $gpu_index Part 1 jobs."
        echo "Manifest: $MANIFEST"
    fi
    exit 0
fi

SYNTHESIS_JOB="$BUNDLE/synthesis_job.sh"

submit_gpu mkdt_distill kil_p2_syn_C10_VGG_1 01:30:00 8G "" "$SYNTHESIS_JOB" CIFAR10 1
SYN_C10_1=$LAST_JOB_ID
submit_gpu mkdt_distill kil_p2_syn_C10_VGG_2 01:30:00 8G "" "$SYNTHESIS_JOB" CIFAR10 2
SYN_C10_2=$LAST_JOB_ID
submit_gpu mkdt_distill kil_p2_syn_C10_VGG_5 01:30:00 8G "" "$SYNTHESIS_JOB" CIFAR10 5
SYN_C10_5=$LAST_JOB_ID
submit_gpu mkdt_distill kil_p2_syn_C100_VGG_1 01:30:00 8G "" "$SYNTHESIS_JOB" CIFAR100 1
SYN_C100_1=$LAST_JOB_ID
submit_gpu mkdt_distill kil_p2_syn_C100_VGG_2 01:30:00 8G "" "$SYNTHESIS_JOB" CIFAR100 2
SYN_C100_2=$LAST_JOB_ID
submit_gpu mkdt_distill kil_p2_syn_C100_VGG_5 01:30:00 8G "" "$SYNTHESIS_JOB" CIFAR100 5
SYN_C100_5=$LAST_JOB_ID

for source in CIFAR10 CIFAR100; do
    if [ "$source" = CIFAR10 ]; then
        scope=all
    else
        scope=main
    fi
    for size in 1 2 5; do
        case "$source-$size" in
            CIFAR10-1) synth_id=$SYN_C10_1 ;;
            CIFAR10-2) synth_id=$SYN_C10_2 ;;
            CIFAR10-5) synth_id=$SYN_C10_5 ;;
            CIFAR100-1) synth_id=$SYN_C100_1 ;;
            CIFAR100-2) synth_id=$SYN_C100_2 ;;
            CIFAR100-5) synth_id=$SYN_C100_5 ;;
        esac
        case "$size" in
            1) sim_epochs=5000; bt_epochs=10000 ;;
            2) sim_epochs=4000; bt_epochs=8000 ;;
            5) sim_epochs=3000; bt_epochs=6000 ;;
        esac
        for ssl in simclr barlowtwins; do
            if [ "$ssl" = simclr ]; then
                epochs=$sim_epochs
                short_ssl=simclr
            else
                epochs=$bt_epochs
                short_ssl=barlow
            fi
            submit_gpu mkdt_pretrain "kil_p2_pre_${source}_VGG_${size}_${short_ssl}" 01:00:00 4G \
                "afterok:$synth_id" "$MKDT_JOB" pretrain "$source" VGG11 VGG11 \
                "$size" "$ssl" "$epochs" "$scope"
            pretrain_id=$LAST_JOB_ID
            submit_gpu mkdt_evaluate "kil_p2_ev_${source}_VGG_${size}_${short_ssl}" 01:00:00 4G \
                "afterok:$pretrain_id" "$MKDT_JOB" evaluate "$source" VGG11 VGG11 \
                "$size" "$ssl" "$epochs" "$scope"
        done
    done
done

if [ "$DRY_RUN" = 1 ]; then
    echo "Validated $gpu_index Part 2 jobs; nothing was submitted."
else
    echo "Submitted $gpu_index Part 2 jobs."
    echo "Manifest: $MANIFEST"
fi
