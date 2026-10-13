#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
export PROJECT_ROOT
cd "$PROJECT_ROOT"

[[ -f krrst_teacher_ckpt/teacher_cifar10.pt ]] || {
    echo "Missing krrst_teacher_ckpt/teacher_cifar10.pt" >&2
    exit 1
}
for target in CIFAR100 Aircraft CUB2011 Dogs Flowers TinyImageNet; do
    cache="benchmarks/downstream/results/batch_20261009T043709Z_zC8EPD/targets/$target.pt"
    [[ -f $cache ]] || { echo "Missing target cache: $cache" >&2; exit 1; }
done

mkdir -p benchmarks/mkdt_images_only/results
MKDT_BATCH_ROOT=$(mktemp -d "$PROJECT_ROOT/benchmarks/mkdt_images_only/results/batch_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")
MKDT_SYNTH_ROOT="$MKDT_BATCH_ROOT/synthesis"
MKDT_IMAGE_ROOT="$MKDT_SYNTH_ROOT"
MKDT_CODE_ROOT="$MKDT_BATCH_ROOT/code"
TARGET_CACHE_ROOT="$PROJECT_ROOT/benchmarks/downstream/results/batch_20261009T043709Z_zC8EPD/targets"
DATA_ROOT="$SCRATCH/data"
export MKDT_BATCH_ROOT MKDT_SYNTH_ROOT MKDT_IMAGE_ROOT MKDT_CODE_ROOT TARGET_CACHE_ROOT DATA_ROOT

mkdir -p "$MKDT_BATCH_ROOT/logs" "$MKDT_BATCH_ROOT/runs" "$MKDT_CODE_ROOT/benchmarks/mkdt_images_only" "$MKDT_CODE_ROOT/OT-SSL-DD"
cp benchmarks/benchmark_cifar10.py benchmarks/benchmark_mkdt_images.py benchmarks/downstream_data.py "$MKDT_CODE_ROOT/benchmarks/"
cp benchmarks/mkdt_images_only/report.py "$MKDT_CODE_ROOT/benchmarks/mkdt_images_only/"
cp OT-SSL-DD/utils.py OT-SSL-DD/networks.py "$MKDT_CODE_ROOT/OT-SSL-DD/"
tar -czf "$MKDT_BATCH_ROOT/source.tar.gz" \
    get_target_rep.py buffer.py sort_dset.py distill.py utils.py networks.py reparam_module.py \
    benchmarks/benchmark_cifar10.py benchmarks/benchmark_mkdt_images.py benchmarks/downstream_data.py \
    benchmarks/mkdt_images_only/environment.sh benchmarks/mkdt_images_only/job.sh \
    benchmarks/mkdt_images_only/report.py benchmarks/mkdt_images_only/report.sh \
    benchmarks/mkdt_images_only/synthesis_job.sh benchmarks/mkdt_images_only/submit_full.sh \
    OT-SSL-DD/utils.py OT-SSL-DD/networks.py

submit() {
    local account=$1 name=$2 limit=$3 dependency=$4 script=$5
    shift 5
    local args=(sbatch --parsable --account="$account" --job-name="$name" \
        --time="$limit" --chdir="$PROJECT_ROOT" \
        --output="$MKDT_BATCH_ROOT/logs/%x-%j.out")
    [[ -n $dependency ]] && args+=(--dependency="$dependency" --kill-on-invalid-dep=yes)
    local id
    id=$("${args[@]}" "$script" "$@")
    printf '%s\t%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "$id" "$account" "$name" >> "$MKDT_BATCH_ROOT/submitted.tsv"
    printf 'Submitted %s on %s: %s\n' "$name" "$account" "$id" >&2
    printf '%s\n' "${id%%;*}"
}

synth_script="$PROJECT_ROOT/benchmarks/mkdt_images_only/synthesis_job.sh"
bench_script="$PROJECT_ROOT/benchmarks/mkdt_images_only/job.sh"
prepare_id=$(submit aip-boyuwang mkdt_c10_prepare 0-04:00:00 '' "$synth_script" prepare)

declare -A sim_epochs=( [1]=5000 [2]=4000 [5]=3000 )
declare -A bt_epochs=( [1]=10000 [2]=8000 [5]=6000 )
declare -A pretrain_time=(
    [ConvNet_1_simclr]=0-01:30:00 [ConvNet_2_simclr]=0-01:40:00 [ConvNet_5_simclr]=0-02:15:00
    [ConvNet_1_barlowtwins]=0-02:05:00 [ConvNet_2_barlowtwins]=0-02:40:00 [ConvNet_5_barlowtwins]=0-04:10:00
    [VGG11_1_simclr]=0-01:35:00 [VGG11_2_simclr]=0-01:50:00 [VGG11_5_simclr]=0-02:30:00
    [VGG11_1_barlowtwins]=0-02:00:00 [VGG11_2_barlowtwins]=0-02:40:00 [VGG11_5_barlowtwins]=0-04:00:00
    [ResNet18_1_simclr]=0-02:45:00 [ResNet18_2_simclr]=0-03:45:00 [ResNet18_5_simclr]=0-06:05:00
    [ResNet18_1_barlowtwins]=0-04:25:00 [ResNet18_2_barlowtwins]=0-06:25:00 [ResNet18_5_barlowtwins]=0-11:05:00
)
declare -A distill_ids
for size in 1 2 5; do
    distill_ids[$size]=$(submit aip-boyuwang "mkdt_c10_distill${size}" 0-08:00:00 \
        "afterok:$prepare_id" "$synth_script" distill "$size")
done

evaluation_ids=()
for model in ConvNet VGG11 ResNet18; do
    for size in 1 2 5; do
        for ssl in simclr barlowtwins; do
            if [[ $ssl == simclr ]]; then
                epochs=${sim_epochs[$size]}
            else
                epochs=${bt_epochs[$size]}
            fi
            limit=${pretrain_time[${model}_${size}_${ssl}]}
            pretrain_id=$(submit aip-boyuwang "mkdt_pre_${model}_s${size}_${ssl}" "$limit" \
                "afterok:${distill_ids[$size]}" "$bench_script" pretrain "$model" "$size" "$ssl" "$epochs")
            eval_account=aip-boyuwang
            case "${model}_${size}_${ssl}" in
                ConvNet_2_simclr|VGG11_1_barlowtwins|ResNet18_1_simclr|ResNet18_5_barlowtwins)
                    eval_account=aip-yiweilu ;;
            esac
            eval_limit=0-00:45:00
            [[ $model == ResNet18 ]] && eval_limit=0-01:00:00
            eval_id=$(submit "$eval_account" "mkdt_eval_${model}_s${size}_${ssl}" "$eval_limit" \
                "afterok:$pretrain_id" "$bench_script" evaluate "$model" "$size" "$ssl" "$epochs")
            evaluation_ids+=("$eval_id")
        done
    done
done

dependency=$(IFS=:; echo "${evaluation_ids[*]}")
report_id=$(sbatch --parsable --account=aip-boyuwang --job-name=mkdt_tables \
    --chdir="$PROJECT_ROOT" --output="$MKDT_BATCH_ROOT/logs/%x-%j.out" \
    --dependency="afterany:$dependency" \
    "$PROJECT_ROOT/benchmarks/mkdt_images_only/report.sh")
printf '%s\t%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "$report_id" aip-boyuwang mkdt_tables >> "$MKDT_BATCH_ROOT/submitted.tsv"

printf 'Submitted MKDT CIFAR10 synthesis (1/2/5%%), images-only all-backbone benchmarks, and table report.\n'
printf 'Batch: %s\nPrepare: %s\nDistill: 1%%=%s 2%%=%s 5%%=%s\nEvaluation chains: %s\nReport: %s\n' \
    "$MKDT_BATCH_ROOT" "$prepare_id" "${distill_ids[1]}" "${distill_ids[2]}" "${distill_ids[5]}" "${#evaluation_ids[@]}" "$report_id"
