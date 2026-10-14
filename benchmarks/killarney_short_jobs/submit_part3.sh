#!/bin/sh
[ -n "${BASH_VERSION:-}" ] || exec /bin/bash "$0" "$@"
set -euo pipefail

PROJECT_ROOT=$(readlink -f /home/mmoslem3/scratch/MKDT) || {
    echo "FATAL: cannot resolve /home/mmoslem3/scratch/MKDT" >&2
    exit 1
}
DATA_ROOT=/home/mmoslem3/scratch/data
BUNDLE="$PROJECT_ROOT/benchmarks/killarney_short_jobs"
MKDT_BATCH="$PROJECT_ROOT/benchmarks/mkdt_multidataset/results/batch_20261010T160046Z_Y6dqZS"
MKDT_CODE_ROOT="$MKDT_BATCH/code"
MKDT_SYNTH_ROOT="$MKDT_BATCH/synthetic"
TARGET_CACHE_ROOT="$MKDT_BATCH/targets"
LOG_ROOT="$BUNDLE/logs/part3"
MANIFEST="$BUNDLE/submitted_part3.tsv"
WANG_ACCOUNT=aip-boyuwang
LU_ACCOUNT=aip-yiweilu
DRY_RUN=${DRY_RUN:-0}

fail() {
    echo "FATAL: $*" >&2
    exit 1
}

case "$PROJECT_ROOT" in
    /scratch/*|/project/*) ;;
    *) fail "resolved project path is not on scratch/project: $PROJECT_ROOT" ;;
esac
cd "$PROJECT_ROOT"

[[ -d $DATA_ROOT ]] || fail "missing data root: $DATA_ROOT"
[[ -s /home/mmoslem3/ENV/bin/activate ]] || fail "missing /home/mmoslem3/ENV/bin/activate"
[[ -d $MKDT_CODE_ROOT ]] || fail "missing code snapshot: $MKDT_CODE_ROOT"
[[ -s $BUNDLE/environment.sh ]] || fail "missing Killarney environment worker"
[[ -s $BUNDLE/mkdt_job.sh ]] || fail "missing MKDT pretrain/evaluation worker"
[[ -s $BUNDLE/synthesis_part3_job.sh ]] || fail "missing Part 3 synthesis worker"
[[ -s $MKDT_CODE_ROOT/benchmarks/benchmark_mkdt_multidataset.py ]] || fail "missing MKDT benchmark driver"
[[ -s $MKDT_CODE_ROOT/buffer.py ]] || fail "missing buffer.py"
[[ -s $MKDT_CODE_ROOT/sort_dset.py ]] || fail "missing sort_dset.py"
[[ -s $MKDT_CODE_ROOT/distill.py ]] || fail "missing distill.py"

for target in CIFAR10 CIFAR100 TinyImageNet Aircraft CUB2011 Dogs Flowers; do
    [[ -s "$TARGET_CACHE_ROOT/$target.pt" ]] || fail "missing target cache: $TARGET_CACHE_ROOT/$target.pt"
done

target_rep="$MKDT_SYNTH_ROOT/work/target_rep/barlow_twins/Tiny_target_rep_train.pt"
[[ -s $target_rep ]] || fail "missing Tiny target representations: $target_rep"
for index in $(seq 0 99); do
    buffer="$MKDT_SYNTH_ROOT/work/buffers/Tiny/ConvNetD4/replay_buffer_${index}.pt"
    [[ -s $buffer ]] || fail "missing Tiny ConvNetD4 expert buffer: $buffer"
done
for size in 1 2 5; do
    init="$MKDT_SYNTH_ROOT/work/init/Tiny/ConvNetD4/Tiny_barlow_twins_${size}_high_loss_indices.pkl"
    [[ -s $init ]] || fail "missing Tiny ConvNetD4 initialization: $init"
done

if [[ $DRY_RUN != 1 ]]; then
    command -v sbatch >/dev/null 2>&1 || fail "sbatch is not available"
    submitted_rows=0
    if [[ -s $MANIFEST ]]; then
        submitted_rows=$(awk 'NR > 1 && NF {count++} END {print count + 0}' "$MANIFEST")
    fi
    [[ $submitted_rows -eq 0 ]] || fail "$MANIFEST already contains submitted job rows"
    mkdir -p "$LOG_ROOT"
    printf 'submitted_utc\tjob_id\taccount\tclass\tname\ttime\tmem\tdependency\n' > "$MANIFEST"
fi

gpu_index=0
LAST_JOB_ID=
submit_gpu() {
    local class=$1
    local name=$2
    local limit=$3
    local memory=$4
    local dependency=$5
    local script=$6
    shift 6

    gpu_index=$((gpu_index + 1))
    local account=$WANG_ACCOUNT
    case "$gpu_index" in
        10|20|30|40|48) account=$LU_ACCOUNT ;;
    esac

    if [[ $DRY_RUN == 1 ]]; then
        LAST_JOB_ID="DRY${gpu_index}"
        printf 'Would submit %-58s time=%s mem=%s account=%s dependency=%s\n' \
            "$name" "$limit" "$memory" "$account" "${dependency:-none}"
        return
    fi

    args=(sbatch --parsable
        --account="$account"
        --job-name="$name"
        --nodes=1
        --ntasks=1
        --cpus-per-task=1
        --gpus-per-node=1
        --time="$limit"
        --mem="$memory"
        --chdir="$PROJECT_ROOT"
        --output="$LOG_ROOT/%x-%j.out")
    if [[ -n $dependency ]]; then
        args+=(--dependency="$dependency" --kill-on-invalid-dep=yes)
    fi

    submitted=$("${args[@]}" "$script" "$@")
    LAST_JOB_ID=${submitted%%;*}
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$(date -u +%FT%TZ)" "$LAST_JOB_ID" "$account" "$class" \
        "$name" "$limit" "$memory" "$dependency" >> "$MANIFEST"
    echo "Submitted $name: $LAST_JOB_ID"
}

SYNTHESIS_JOB="$BUNDLE/synthesis_part3_job.sh"
MKDT_JOB="$BUNDLE/mkdt_job.sh"

declare -A synth_ids

# CIFAR ConvNet distillation took about 1 h 54 m. Tiny uses twice as many
# iterations, one quarter as many unrolled steps, and the deeper 64x64 model;
# the request is the resulting ~3 h 50 m estimate plus a 45-minute buffer.
for size in 1 2 5; do
    submit_gpu mkdt_distill "kil_p3_syn_Tiny_Conv_${size}" 04:40:00 12G "" \
        "$SYNTHESIS_JOB" distill Tiny ConvNetD4 "$size" ConvNet
    synth_ids[ConvNet_${size}]=$LAST_JOB_ID
done

# Five disjoint 20-expert shards. Scaling the completed Tiny ConvNet expert
# shards and CIFAR VGG11 logs gives ~1 h 45 m per shard; add 45 minutes.
tiny_vgg_experts=()
for start in 0 20 40 60 80; do
    submit_gpu mkdt_experts "kil_p3_exp_Tiny_VGG_${start}" 02:30:00 20G "" \
        "$SYNTHESIS_JOB" experts Tiny VGG11 "$start"
    tiny_vgg_experts+=("$LAST_JOB_ID")
done
tiny_vgg_dependency=$(IFS=:; echo "${tiny_vgg_experts[*]}")

# Tiny ConvNet sorting took 10 m 27 s. VGG11 gets a larger allowance plus the
# requested 45-minute non-evaluation buffer.
submit_gpu mkdt_sort kil_p3_sort_Tiny_VGG 01:10:00 26G \
    "afterok:$tiny_vgg_dependency" "$SYNTHESIS_JOB" sort Tiny VGG11
tiny_vgg_sort=$LAST_JOB_ID

# Current Part 2 VGG11 logs project ~80-86 minutes for CIFAR. Tiny uses twice
# the iterations but fewer unrolled steps and batch 32; 2 h 10 m includes the
# requested 45-minute buffer.
for size in 1 2 5; do
    submit_gpu mkdt_distill "kil_p3_syn_Tiny_VGG_${size}" 02:10:00 12G \
        "afterok:$tiny_vgg_sort" "$SYNTHESIS_JOB" \
        distill Tiny VGG11 "$size" VGG11
    synth_ids[VGG11_${size}]=$LAST_JOB_ID
done

declare -A sim_epochs=( [1]=5000 [2]=4000 [5]=3000 )
declare -A bt_epochs=( [1]=10000 [2]=8000 [5]=6000 )

pretrain_limit() {
    case "$1-$2-$3" in
        ConvNet-1-simclr) echo 00:50:00 ;;
        ConvNet-1-barlowtwins) echo 00:55:00 ;;
        ConvNet-2-simclr) echo 00:55:00 ;;
        ConvNet-2-barlowtwins) echo 01:05:00 ;;
        ConvNet-5-simclr) echo 01:00:00 ;;
        ConvNet-5-barlowtwins) echo 01:15:00 ;;
        ResNet18-1-simclr) echo 01:05:00 ;;
        ResNet18-1-barlowtwins) echo 01:25:00 ;;
        ResNet18-2-simclr) echo 01:20:00 ;;
        ResNet18-2-barlowtwins) echo 01:50:00 ;;
        ResNet18-5-simclr) echo 01:45:00 ;;
        ResNet18-5-barlowtwins) echo 02:45:00 ;;
        VGG11-1-simclr) echo 00:55:00 ;;
        VGG11-1-barlowtwins) echo 01:00:00 ;;
        VGG11-2-simclr) echo 01:00:00 ;;
        VGG11-2-barlowtwins) echo 01:05:00 ;;
        VGG11-5-simclr) echo 01:05:00 ;;
        VGG11-5-barlowtwins) echo 01:20:00 ;;
        *) fail "no pretraining walltime for $1 $2 $3" ;;
    esac
}

submit_chain() {
    local generator=$1
    local model=$2
    local size=$3
    local ssl=$4
    local scope=$5
    local synth_dependency=$6
    local epochs
    if [[ $ssl == simclr ]]; then
        epochs=${sim_epochs[$size]}
    else
        epochs=${bt_epochs[$size]}
    fi
    local limit
    limit=$(pretrain_limit "$model" "$size" "$ssl")
    local short_ssl=$ssl
    [[ $ssl == barlowtwins ]] && short_ssl=barlow

    submit_gpu mkdt_pretrain \
        "kil_p3_pre_Tiny_${generator}_${model}_${size}_${short_ssl}" \
        "$limit" 4G "afterok:$synth_dependency" "$MKDT_JOB" \
        pretrain TinyImageNet "$generator" "$model" "$size" "$ssl" "$epochs" "$scope"
    pretrain_id=$LAST_JOB_ID

    eval_limit=00:15:00
    [[ $scope == all ]] && eval_limit=00:20:00
    submit_gpu mkdt_evaluate \
        "kil_p3_ev_Tiny_${generator}_${model}_${size}_${short_ssl}" \
        "$eval_limit" 4G "afterok:$pretrain_id" "$MKDT_JOB" \
        evaluate TinyImageNet "$generator" "$model" "$size" "$ssl" "$epochs" "$scope"
}

# One shared SSL encoder is trained per configuration. The limits below use
# the measured per-encoder time from the completed 10-encoder CIFAR jobs,
# scaled by Tiny's 2x synthetic-set size, then add 45 minutes. Evaluations use
# the measured Part 1 / ResNet probe times plus 10 minutes.
for size in 1 2 5; do
    for ssl in simclr barlowtwins; do
        submit_chain ConvNet ConvNet "$size" "$ssl" main \
            "${synth_ids[ConvNet_${size}]}"
        submit_chain ConvNet ResNet18 "$size" "$ssl" all \
            "${synth_ids[ConvNet_${size}]}"
        submit_chain VGG11 VGG11 "$size" "$ssl" main \
            "${synth_ids[VGG11_${size}]}"
    done
done

if [[ $DRY_RUN == 1 ]]; then
    echo "Validated $gpu_index Part 3 jobs; nothing was submitted."
else
    echo "Submitted $gpu_index Part 3 jobs."
    echo "Manifest: $MANIFEST"
fi
