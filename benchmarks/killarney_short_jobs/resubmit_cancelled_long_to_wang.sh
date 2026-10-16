#!/bin/bash
set -euo pipefail

ROOT=/scratch/mmoslem3/MKDT
BATCH="$ROOT/benchmarks/full_data_main_long/results/batch_20261016T042912Z_MLpgLE"
CODE="$BATCH/code"
WORKER="$CODE/benchmarks/killarney_short_jobs/full_data_main_long_job.sh"
ACCOUNT=aip-boyuwang
EXCLUDE=kn072,kn117,kn140,kn161
DRY_RUN=${DRY_RUN:-0}

export PROJECT_ROOT="$ROOT"
export MAINLONG_BATCH="$BATCH"
export MAINLONG_CODE_ROOT="$CODE"
export VENV_ACTIVATE="${VENV_ACTIVATE:-$HOME/ENV/bin/activate}"
if [[ -s $BATCH/targets/CIFAR10.pt && -s $BATCH/targets/CIFAR100.pt ]]; then
    export TARGET_CACHE_ROOT="$BATCH/targets"
else
    source_batch=$(<"$BATCH/target_source_batch.txt")
    export TARGET_CACHE_ROOT="$source_batch/targets"
fi

LAST_JOB_ID=
fake_id=995000

submit_cmd() {
    if (( DRY_RUN )); then
        printf 'DRY RUN: '
        printf '%q ' "$@"
        printf '\n'
        fake_id=$((fake_id + 1))
        LAST_JOB_ID=$fake_id
    else
        submitted=$("$@")
        LAST_JOB_ID=${submitted%%;*}
    fi
}

submit_pretrain() {
    local tag=$1 seeds=$2 limit=$3 source=$4 model=$5 ssl=$6
    submit_cmd sbatch --parsable \
        --account="$ACCOUNT" --job-name="kmlw_${tag}_pre" \
        --array="$seeds" --nodes=1 --ntasks=1 --cpus-per-task=1 \
        --gres=gpu:l40s:1 --exclude="$EXCLUDE" --time="$limit" --mem=8G \
        --chdir="$ROOT" --output="$BATCH/logs/%x-%A_%a.out" \
        "$WORKER" pretrain "$source" "$model" "$ssl"
}

submit_eval() {
    local tag=$1 dependency=$2 source=$3 model=$4 ssl=$5
    local args=(sbatch --parsable
        --account="$ACCOUNT" --job-name="kmlw_${tag}_eval"
        --nodes=1 --ntasks=1 --cpus-per-task=1
        --gres=gpu:l40s:1 --exclude="$EXCLUDE" --time=01:30:00 --mem=8G
        --chdir="$ROOT" --output="$BATCH/logs/%x-%j.out")
    [[ -z $dependency ]] || \
        args+=(--dependency="$dependency" --kill-on-invalid-dep=yes)
    submit_cmd "${args[@]}" "$WORKER" evaluate "$source" "$model" "$ssl"
}

submit_pretrain c10_cn_s 9 02:00:00 CIFAR10 ConvNet simclr
c10_cn_s=$LAST_JOB_ID
submit_eval c10_cn_s \
    "afterok:$c10_cn_s,afterany:6313701,afterany:6315176" \
    CIFAR10 ConvNet simclr

submit_pretrain c10_cn_b 0,1,2,4,5,9 03:00:00 CIFAR10 ConvNet barlowtwins
c10_cn_b=$LAST_JOB_ID
submit_eval c10_cn_b "afterok:$c10_cn_b,afterany:6313703" \
    CIFAR10 ConvNet barlowtwins

submit_pretrain c10_vg_s 2,3,4,6,7,9 02:30:00 CIFAR10 VGG11 simclr
c10_vg_s=$LAST_JOB_ID
submit_eval c10_vg_s "afterok:$c10_vg_s,afterany:6313705" \
    CIFAR10 VGG11 simclr

submit_eval c10_vg_b "" CIFAR10 VGG11 barlowtwins

submit_pretrain c100_cn_s 0,2,3,4,5,6,7,8 02:00:00 CIFAR100 ConvNet simclr
c100_cn_s=$LAST_JOB_ID
submit_eval c100_cn_s "afterok:$c100_cn_s,afterany:6313708" \
    CIFAR100 ConvNet simclr

submit_pretrain c100_cn_b 0,1,2,4,6,7,9 03:00:00 CIFAR100 ConvNet barlowtwins
c100_cn_b=$LAST_JOB_ID
submit_eval c100_cn_b "afterok:$c100_cn_b,afterany:6313710" \
    CIFAR100 ConvNet barlowtwins

submit_pretrain c100_vg_s 0,1,3,4,5,7 02:30:00 CIFAR100 VGG11 simclr
c100_vg_s=$LAST_JOB_ID
submit_eval c100_vg_s "afterok:$c100_vg_s" CIFAR100 VGG11 simclr

submit_pretrain c100_vg_b 9 03:30:00 CIFAR100 VGG11 barlowtwins
c100_vg_b=$LAST_JOB_ID
submit_eval c100_vg_b "afterok:$c100_vg_b" CIFAR100 VGG11 barlowtwins

submit_pretrain c100_r18_s 0,1,2 05:30:00 CIFAR100 ResNet18 simclr
c100_r18_s=$LAST_JOB_ID
submit_eval c100_r18_s "afterok:$c100_r18_s" CIFAR100 ResNet18 simclr

echo "Submitted the cancelled jobs on $ACCOUNT."
