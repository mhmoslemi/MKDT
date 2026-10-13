#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
export PROJECT_ROOT
cd "$PROJECT_ROOT"

for teacher in teacher_cifar10.pt barlow_twins_resnet18_cifar100.pt barlow_twins_resnet18_tinyimagenet.pt; do
    [[ -f "krrst_teacher_ckpt/$teacher" ]] || { echo "Missing teacher: $teacher" >&2; exit 1; }
done

old_targets="$PROJECT_ROOT/benchmarks/downstream/results/batch_20261009T043709Z_zC8EPD/targets"
cross_targets="$PROJECT_ROOT/benchmarks/cross_downstream/results/batch_20261009T171449Z_o91BKP/targets"
for target in CIFAR100 TinyImageNet Aircraft CUB2011 Dogs Flowers; do
    [[ -f "$old_targets/$target.pt" ]] || { echo "Missing immutable target cache: $target" >&2; exit 1; }
done
[[ -f "$cross_targets/CIFAR10.pt" ]] || { echo "Missing immutable target cache: CIFAR10" >&2; exit 1; }

mkdir -p benchmarks/mkdt_multidataset/results
MKDT_MULTI_BATCH=$(mktemp -d "$PROJECT_ROOT/benchmarks/mkdt_multidataset/results/batch_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")
MKDT_CODE_ROOT="$MKDT_MULTI_BATCH/code"
MKDT_SYNTH_ROOT="$MKDT_MULTI_BATCH/synthetic"
TARGET_CACHE_ROOT="$MKDT_MULTI_BATCH/targets"
DATA_ROOT="$SCRATCH/data"
export MKDT_MULTI_BATCH MKDT_CODE_ROOT MKDT_SYNTH_ROOT TARGET_CACHE_ROOT DATA_ROOT

mkdir -p "$MKDT_MULTI_BATCH/logs" "$MKDT_MULTI_BATCH/runs" "$MKDT_SYNTH_ROOT" \
    "$MKDT_CODE_ROOT/benchmarks/mkdt_multidataset" "$MKDT_CODE_ROOT/OT-SSL-DD" "$TARGET_CACHE_ROOT"
for target in CIFAR100 TinyImageNet Aircraft CUB2011 Dogs Flowers; do
    ln -s "$old_targets/$target.pt" "$TARGET_CACHE_ROOT/$target.pt"
done
ln -s "$cross_targets/CIFAR10.pt" "$TARGET_CACHE_ROOT/CIFAR10.pt"

cp get_target_rep.py buffer.py sort_dset.py distill.py utils.py networks.py reparam_module.py "$MKDT_CODE_ROOT/"
cp -a more_dataset data_handler "$MKDT_CODE_ROOT/"
cp benchmarks/benchmark_cifar10.py benchmarks/benchmark_mkdt_multidataset.py benchmarks/downstream_data.py "$MKDT_CODE_ROOT/benchmarks/"
cp OT-SSL-DD/utils.py OT-SSL-DD/networks.py "$MKDT_CODE_ROOT/OT-SSL-DD/"
cp benchmarks/mkdt_multidataset/report.py "$MKDT_CODE_ROOT/benchmarks/mkdt_multidataset/"
tar -czf "$MKDT_MULTI_BATCH/source.tar.gz" \
    get_target_rep.py buffer.py sort_dset.py distill.py utils.py networks.py reparam_module.py \
    more_dataset data_handler \
    benchmarks/benchmark_cifar10.py benchmarks/benchmark_mkdt_multidataset.py benchmarks/downstream_data.py \
    benchmarks/mkdt_multidataset/environment.sh benchmarks/mkdt_multidataset/synthesis_job.sh \
    benchmarks/mkdt_multidataset/benchmark_job.sh benchmarks/mkdt_multidataset/report.py \
    benchmarks/mkdt_multidataset/report.sh benchmarks/mkdt_multidataset/submit.sh \
    OT-SSL-DD/utils.py OT-SSL-DD/networks.py

printf 'submitted_utc\tjob_id\taccount\tclass\tname\n' > "$MKDT_MULTI_BATCH/submitted.tsv"
printf 'source\tgenerator\tevaluator\tsize\tssl_method\tssl_epochs\n' > "$MKDT_MULTI_BATCH/configs.tsv"
gpu_index=0
LAST_JOB_ID=''
submit_gpu() {
    local class=$1 name=$2 limit=$3 memory=$4 dependency=$5 script=$6
    shift 6
    gpu_index=$((gpu_index + 1))
    local account=aip-boyuwang
    (( gpu_index % 10 == 0 )) && account=aip-yiweilu
    local args=(sbatch --parsable --account="$account" --job-name="$name" \
        --time="$limit" --mem="$memory" --chdir="$PROJECT_ROOT" \
        --output="$MKDT_MULTI_BATCH/logs/%x-%j.out")
    [[ -n $dependency ]] && args+=(--dependency="$dependency" --kill-on-invalid-dep=yes)
    local id
    id=$("${args[@]}" "$script" "$@")
    LAST_JOB_ID=${id%%;*}
    printf '%s\t%s\t%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "$LAST_JOB_ID" "$account" "$class" "$name" >> "$MKDT_MULTI_BATCH/submitted.tsv"
    printf 'Submitted %s on %s: %s\n' "$name" "$account" "$LAST_JOB_ID" >&2
}

synthesis="$PROJECT_ROOT/benchmarks/mkdt_multidataset/synthesis_job.sh"
benchmark="$PROJECT_ROOT/benchmarks/mkdt_multidataset/benchmark_job.sh"
declare -A target_ids sort_ids distill_ids

for dataset in CIFAR10 CIFAR100 Tiny; do
    target_time=0-01:00:00
    target_mem=12G
    [[ $dataset == Tiny ]] && { target_time=0-02:00:00; target_mem=24G; }
    submit_gpu target "mkdt_target_${dataset}" "$target_time" "$target_mem" '' "$synthesis" target "$dataset"
    target_ids[$dataset]=$LAST_JOB_ID
done

# Logical ResNet18BN maps to the root repository's ResNet18 implementation,
# which is explicitly constructed with batch normalization in networks.py.
combinations=(
    'CIFAR10 ConvNet ConvNet'
    'CIFAR10 VGG11 VGG11'
    'CIFAR10 ResNet18 ResNet18BN'
    'CIFAR100 ConvNet ConvNet'
    'CIFAR100 VGG11 VGG11'
    'CIFAR100 ResNet18 ResNet18BN'
    'Tiny ConvNetD4 ConvNet'
    'Tiny ResNet18 ResNet18BN'
)
for entry in "${combinations[@]}"; do
    read -r dataset implementation generator <<< "$entry"
    expert_time=0-02:00:00
    expert_mem=16G
    sort_time=0-02:00:00
    sort_mem=16G
    if [[ $dataset == Tiny && $implementation == ConvNetD4 ]]; then
        expert_time=0-04:00:00; expert_mem=24G; sort_time=0-04:00:00; sort_mem=24G
    elif [[ $dataset == Tiny && $implementation == ResNet18 ]]; then
        expert_time=0-10:00:00; expert_mem=24G; sort_time=0-08:00:00; sort_mem=24G
    elif [[ $implementation == ResNet18 || $implementation == VGG11 ]]; then
        expert_time=0-04:00:00; sort_time=0-04:00:00
    fi
    shard_ids=()
    for start in 0 20 40 60 80; do
        submit_gpu experts "mkdt_exp_${dataset}_${generator}_${start}" "$expert_time" "$expert_mem" \
            "afterok:${target_ids[$dataset]}" "$synthesis" experts "$dataset" "$implementation" "$start"
        shard_ids+=("$LAST_JOB_ID")
    done
    shard_dependency=$(IFS=:; echo "${shard_ids[*]}")
    submit_gpu sort "mkdt_sort_${dataset}_${generator}" "$sort_time" "$sort_mem" \
        "afterok:$shard_dependency" "$synthesis" sort "$dataset" "$implementation"
    sort_ids[${dataset}_${generator}]=$LAST_JOB_ID
    for size in 1 2 5; do
        distill_time=0-10:00:00
        distill_mem=16G
        if [[ $dataset == Tiny || $implementation == ResNet18 || $implementation == VGG11 ]]; then
            distill_time=1-00:00:00
            distill_mem=24G
        fi
        submit_gpu distill "mkdt_syn_${dataset}_${generator}_${size}" "$distill_time" "$distill_mem" \
            "afterok:${sort_ids[${dataset}_${generator}]}" \
            "$synthesis" distill "$dataset" "$implementation" "$size" "$generator"
        distill_ids[${dataset}_${generator}_${size}]=$LAST_JOB_ID
    done
done

declare -A sim_epochs=( [1]=5000 [2]=4000 [5]=3000 )
declare -A bt_epochs=( [1]=10000 [2]=8000 [5]=6000 )
declare -A c32_time=(
    [ConvNet_1_simclr]=0-01:30:00 [ConvNet_2_simclr]=0-01:40:00 [ConvNet_5_simclr]=0-02:15:00
    [ConvNet_1_barlowtwins]=0-02:05:00 [ConvNet_2_barlowtwins]=0-02:40:00 [ConvNet_5_barlowtwins]=0-04:10:00
    [VGG11_1_simclr]=0-00:40:00 [VGG11_2_simclr]=0-01:05:00 [VGG11_5_simclr]=0-01:50:00
    [VGG11_1_barlowtwins]=0-01:15:00 [VGG11_2_barlowtwins]=0-01:55:00 [VGG11_5_barlowtwins]=0-03:30:00
    [ResNet18_1_simclr]=0-02:45:00 [ResNet18_2_simclr]=0-03:45:00 [ResNet18_5_simclr]=0-06:05:00
    [ResNet18_1_barlowtwins]=0-04:25:00 [ResNet18_2_barlowtwins]=0-06:25:00 [ResNet18_5_barlowtwins]=0-11:05:00
)
declare -A tiny_time=(
    [ConvNet_1_simclr]=0-01:05:00 [ConvNet_2_simclr]=0-01:35:00 [ConvNet_5_simclr]=0-02:55:00
    [ConvNet_1_barlowtwins]=0-02:30:00 [ConvNet_2_barlowtwins]=0-03:55:00 [ConvNet_5_barlowtwins]=0-07:10:00
    [ResNet18_1_simclr]=0-04:00:00 [ResNet18_2_simclr]=0-06:15:00 [ResNet18_5_simclr]=0-11:35:00
    [ResNet18_1_barlowtwins]=0-07:45:00 [ResNet18_2_barlowtwins]=0-12:25:00 [ResNet18_5_barlowtwins]=0-23:10:00
)

evaluation_ids=()
for source_dataset in CIFAR10 CIFAR100 TinyImageNet; do
    mkdt_dataset=$source_dataset
    [[ $source_dataset == TinyImageNet ]] && mkdt_dataset=Tiny
    generators=(ConvNet ResNet18BN)
    [[ $source_dataset != TinyImageNet ]] && generators+=(VGG11)
    for generator in "${generators[@]}"; do
        implementation=$generator
        [[ $generator == ResNet18BN ]] && implementation=ResNet18
        [[ $source_dataset == TinyImageNet && $generator == ConvNet ]] && implementation=ConvNetD4
        models=(ConvNet ResNet18)
        [[ $generator == VGG11 ]] && models=(VGG11)
        for model in "${models[@]}"; do
            for size in 1 2 5; do
                for ssl in simclr barlowtwins; do
                    if [[ $ssl == simclr ]]; then epochs=${sim_epochs[$size]}; else epochs=${bt_epochs[$size]}; fi
                    key=${model}_${size}_${ssl}
                    if [[ $source_dataset == TinyImageNet ]]; then limit=${tiny_time[$key]}; else limit=${c32_time[$key]}; fi
                    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$source_dataset" "$generator" "$model" "$size" "$ssl" "$epochs" >> "$MKDT_MULTI_BATCH/configs.tsv"
                    submit_gpu pretrain "mkpre_${source_dataset}_${generator}_${model}_${size}_${ssl}" "$limit" 15G \
                        "afterok:${distill_ids[${mkdt_dataset}_${generator}_${size}]}" \
                        "$benchmark" pretrain "$source_dataset" "$generator" "$model" "$size" "$ssl" "$epochs"
                    pretrain_id=$LAST_JOB_ID
                    eval_limit=0-00:30:00
                    [[ $model == ResNet18 ]] && eval_limit=0-01:30:00
                    submit_gpu evaluate "mkev_${source_dataset}_${generator}_${model}_${size}_${ssl}" "$eval_limit" 15G \
                        "afterok:$pretrain_id" \
                        "$benchmark" evaluate "$source_dataset" "$generator" "$model" "$size" "$ssl" "$epochs"
                    evaluation_ids+=("$LAST_JOB_ID")
                done
            done
        done
    done
done

dependency=$(IFS=:; echo "${evaluation_ids[*]}")
report_id=$(sbatch --parsable --account=aip-boyuwang --job-name=mkdt_dual_tables \
    --chdir="$PROJECT_ROOT" --output="$MKDT_MULTI_BATCH/logs/%x-%j.out" \
    --dependency="afterany:$dependency" \
    "$PROJECT_ROOT/benchmarks/mkdt_multidataset/report.sh")
report_id=${report_id%%;*}
printf '%s\t%s\t%s\tCPU\t%s\n' "$(date -u +%FT%TZ)" "$report_id" aip-boyuwang mkdt_dual_tables >> "$MKDT_MULTI_BATCH/submitted.tsv"

printf 'Batch: %s\nSynthetic root: %s\nGPU jobs: %s\nEvaluation chains: %s\nReport: %s\n' \
    "$MKDT_MULTI_BATCH" "$MKDT_SYNTH_ROOT" "$gpu_index" "${#evaluation_ids[@]}" "$report_id"
