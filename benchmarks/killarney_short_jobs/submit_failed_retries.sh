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
LOG_ROOT="$BUNDLE/logs/failed_retries"
MANIFEST="$BUNDLE/submitted_failed_retries.tsv"
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
[[ -s /home/mmoslem3/ENV/bin/activate ]] || fail "missing virtual environment"
[[ -s $BUNDLE/environment.sh ]] || fail "missing environment worker"
[[ -s $BUNDLE/mkdt_job.sh ]] || fail "missing MKDT worker"
[[ -s $BUNDLE/synthesis_job.sh ]] || fail "missing CIFAR synthesis worker"
[[ -s $BUNDLE/retry_c10_vgg2_job.sh ]] || fail "missing CIFAR retry worker"
[[ -s $BUNDLE/synthesis_part3_job.sh ]] || fail "missing Tiny synthesis worker"
[[ -s $MKDT_CODE_ROOT/networks.py ]] || fail "missing MKDT networks.py"
grep -Fq 'x = F.adaptive_avg_pool2d(x, (1, 1))' "$MKDT_CODE_ROOT/networks.py" || \
    fail "Tiny VGG11 adaptive pooling fix is missing"
[[ -s $TARGET_CACHE_ROOT/CIFAR10.pt ]] || fail "missing CIFAR10 target cache"
[[ -s $TARGET_CACHE_ROOT/TinyImageNet.pt ]] || fail "missing Tiny ImageNet target cache"
[[ -s $MKDT_SYNTH_ROOT/work/target_rep/barlow_twins/CIFAR10_target_rep_train.pt ]] || \
    fail "missing CIFAR10 target representations"
[[ -s $MKDT_SYNTH_ROOT/work/target_rep/barlow_twins/Tiny_target_rep_train.pt ]] || \
    fail "missing Tiny target representations"
[[ -s $MKDT_SYNTH_ROOT/work/init/CIFAR10/VGG11/CIFAR10_barlow_twins_2_high_loss_indices.pkl ]] || \
    fail "missing CIFAR10 VGG11 2% initialization"
for index in $(seq 0 99); do
    [[ -s $MKDT_SYNTH_ROOT/work/buffers/CIFAR10/VGG11/replay_buffer_${index}.pt ]] || \
        fail "missing CIFAR10 VGG11 expert buffer $index"
done

if [[ $DRY_RUN != 1 ]]; then
    command -v sbatch >/dev/null 2>&1 || fail "sbatch is not available"
    submitted_rows=0
    if [[ -s $MANIFEST ]]; then
        submitted_rows=$(awk 'NR > 1 && NF {count++} END {print count + 0}' "$MANIFEST")
    fi
    [[ $submitted_rows -eq 0 ]] || fail "$MANIFEST already contains submissions"
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
        10|20|26) account=$LU_ACCOUNT ;;
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

MKDT_JOB="$BUNDLE/mkdt_job.sh"
TINY_JOB="$BUNDLE/synthesis_part3_job.sh"

# Retry the failed CIFAR10 VGG11 2% chain.  The 4:15 allocation permits up to
# three unchanged stochastic attempts; one successful attempt took ~80 min.
submit_gpu mkdt_distill kil_retry_syn_C10_VGG_2 04:15:00 8G "" \
    "$BUNDLE/retry_c10_vgg2_job.sh"
c10_synth=$LAST_JOB_ID

for ssl in simclr barlowtwins; do
    if [[ $ssl == simclr ]]; then
        epochs=4000
        short_ssl=simclr
    else
        epochs=8000
        short_ssl=barlow
    fi
    submit_gpu mkdt_pretrain "kil_retry_pre_C10_VGG_2_${short_ssl}" 01:00:00 4G \
        "afterok:$c10_synth" "$MKDT_JOB" \
        pretrain CIFAR10 VGG11 VGG11 2 "$ssl" "$epochs" all
    pretrain_id=$LAST_JOB_ID
    submit_gpu mkdt_evaluate "kil_retry_ev_C10_VGG_2_${short_ssl}" 01:00:00 4G \
        "afterok:$pretrain_id" "$MKDT_JOB" \
        evaluate CIFAR10 VGG11 VGG11 2 "$ssl" "$epochs" all
done

# Rebuild the failed Tiny ImageNet VGG11 chain with the resolution-safe VGG.
tiny_experts=()
for start in 0 20 40 60 80; do
    submit_gpu mkdt_experts "kil_retry_exp_Tiny_VGG_${start}" 02:30:00 20G "" \
        "$TINY_JOB" experts Tiny VGG11 "$start"
    tiny_experts+=("$LAST_JOB_ID")
done
tiny_expert_dependency=$(IFS=:; echo "${tiny_experts[*]}")

submit_gpu mkdt_sort kil_retry_sort_Tiny_VGG 01:10:00 26G \
    "afterok:$tiny_expert_dependency" "$TINY_JOB" sort Tiny VGG11
tiny_sort=$LAST_JOB_ID

declare -A tiny_synth
for size in 1 2 5; do
    submit_gpu mkdt_distill "kil_retry_syn_Tiny_VGG_${size}" 02:10:00 12G \
        "afterok:$tiny_sort" "$TINY_JOB" distill Tiny VGG11 "$size" VGG11
    tiny_synth[$size]=$LAST_JOB_ID
done

declare -A sim_epochs=( [1]=5000 [2]=4000 [5]=3000 )
declare -A bt_epochs=( [1]=10000 [2]=8000 [5]=6000 )
for size in 1 2 5; do
    for ssl in simclr barlowtwins; do
        if [[ $ssl == simclr ]]; then
            epochs=${sim_epochs[$size]}
            short_ssl=simclr
            case "$size" in
                1) pre_limit=00:55:00 ;;
                2) pre_limit=01:00:00 ;;
                5) pre_limit=01:05:00 ;;
            esac
        else
            epochs=${bt_epochs[$size]}
            short_ssl=barlow
            case "$size" in
                1) pre_limit=01:00:00 ;;
                2) pre_limit=01:05:00 ;;
                5) pre_limit=01:20:00 ;;
            esac
        fi
        submit_gpu mkdt_pretrain "kil_retry_pre_Tiny_VGG_${size}_${short_ssl}" \
            "$pre_limit" 4G "afterok:${tiny_synth[$size]}" "$MKDT_JOB" \
            pretrain TinyImageNet VGG11 VGG11 "$size" "$ssl" "$epochs" main
        pretrain_id=$LAST_JOB_ID
        submit_gpu mkdt_evaluate "kil_retry_ev_Tiny_VGG_${size}_${short_ssl}" \
            00:15:00 4G "afterok:$pretrain_id" "$MKDT_JOB" \
            evaluate TinyImageNet VGG11 VGG11 "$size" "$ssl" "$epochs" main
    done
done

if [[ $DRY_RUN == 1 ]]; then
    echo "Validated $gpu_index failed-chain retry jobs; nothing was submitted."
else
    echo "Submitted $gpu_index failed-chain retry jobs."
    echo "Manifest: $MANIFEST"
fi
