#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
FULL_BATCH="$PROJECT_ROOT/benchmarks/full_data_refresh/results/batch_20261010T160040Z_73NiHv"
FULL_CODE_ROOT="$FULL_BATCH/code"
MKDT_MULTI_BATCH="$PROJECT_ROOT/benchmarks/mkdt_multidataset/results/batch_20261010T160046Z_Y6dqZS"
MKDT_CODE_ROOT="$MKDT_MULTI_BATCH/code"
MKDT_SYNTH_ROOT="$MKDT_MULTI_BATCH/synthetic"
TARGET_CACHE_ROOT="$MKDT_MULTI_BATCH/targets"
DATA_ROOT="$SCRATCH/data"
export PROJECT_ROOT FULL_BATCH FULL_CODE_ROOT MKDT_MULTI_BATCH MKDT_CODE_ROOT
export MKDT_SYNTH_ROOT TARGET_CACHE_ROOT DATA_ROOT
cd "$PROJECT_ROOT"

[[ -d $FULL_CODE_ROOT && -d $MKDT_CODE_ROOT ]] || {
    echo 'Missing benchmark batch code snapshot' >&2; exit 1;
}
for target in CIFAR10 CIFAR100 TinyImageNet Aircraft CUB2011 Dogs Flowers; do
    [[ -f "$TARGET_CACHE_ROOT/$target.pt" ]] || {
        echo "Missing target cache: $target" >&2; exit 1;
    }
done
for dataset in CIFAR10 CIFAR100 Tiny; do
    [[ -f "$MKDT_SYNTH_ROOT/work/target_rep/barlow_twins/${dataset}_target_rep_train.pt" ]] || {
        echo "Missing target representations: $dataset" >&2; exit 1;
    }
done
for dataset in CIFAR10 CIFAR100; do
    [[ -f "$MKDT_SYNTH_ROOT/work/buffers/$dataset/VGG11/replay_buffer_99.pt" ]] || {
        echo "Missing VGG11 expert buffers for $dataset" >&2; exit 1;
    }
    for size in 1 2 5; do
        [[ -f "$MKDT_SYNTH_ROOT/work/init/$dataset/VGG11/${dataset}_barlow_twins_${size}_high_loss_indices.pkl" ]] || {
            echo "Missing VGG11 initialization for $dataset ${size}%" >&2; exit 1;
        }
    done
done
[[ -f "$MKDT_SYNTH_ROOT/work/buffers/Tiny/ConvNetD4/replay_buffer_99.pt" ]] || {
    echo 'Missing Tiny ImageNet ConvNet expert buffers' >&2; exit 1;
}
for size in 1 2 5; do
    [[ -f "$MKDT_SYNTH_ROOT/work/init/Tiny/ConvNetD4/Tiny_barlow_twins_${size}_high_loss_indices.pkl" ]] || {
        echo "Missing Tiny ConvNet initialization for ${size}%" >&2; exit 1;
    }
done

manifest="$PROJECT_ROOT/benchmarks/remaining_cells/submitted.tsv"
if [[ -s $manifest ]]; then
    echo "Refusing duplicate submission: $manifest already contains jobs" >&2
    exit 1
fi
printf 'submitted_utc\tjob_id\taccount\tclass\tname\ttime\tmem\tdependency\n' > "$manifest"

# The batch environments execute their immutable code snapshots. Refresh only
# the two benchmark drivers needed for shared-encoder, 15-probe evaluation.
cp benchmarks/benchmark_full_refresh.py "$FULL_CODE_ROOT/benchmarks/benchmark_full_refresh.py"
cp benchmarks/benchmark_mkdt_multidataset.py "$MKDT_CODE_ROOT/benchmarks/benchmark_mkdt_multidataset.py"
cp distill.py "$MKDT_CODE_ROOT/distill.py"

gpu_index=0
LAST_JOB_ID=''
submit_gpu() {
    local class=$1 name=$2 limit=$3 memory=$4 dependency=$5 output_root=$6 script=$7
    shift 7
    gpu_index=$((gpu_index + 1))
    local account=aip-boyuwang
    (( gpu_index % 10 == 0 )) && account=aip-yiweilu
    local args=(sbatch --parsable --account="$account" --job-name="$name"
        --nodes=1 --ntasks=1 --cpus-per-task=1 --gres=gpu:l40s:1
        --time="$limit" --mem="$memory" --chdir="$PROJECT_ROOT"
        --output="$output_root/%x-%j.out")
    [[ -n $dependency ]] && args+=(--dependency="$dependency" --kill-on-invalid-dep=yes)
    local submitted
    submitted=$("${args[@]}" "$script" "$@")
    LAST_JOB_ID=${submitted%%;*}
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$(date -u +%FT%TZ)" "$LAST_JOB_ID" "$account" "$class" "$name" \
        "$limit" "$memory" "$dependency" >> "$manifest"
    printf 'Submitted %s: %s\n' "$name" "$LAST_JOB_ID" >&2
}

full_job="$PROJECT_ROOT/benchmarks/remaining_cells/full_job.sh"
mkdt_job="$PROJECT_ROOT/benchmarks/remaining_cells/mkdt_job.sh"
synthesis_job="$PROJECT_ROOT/benchmarks/mkdt_multidataset/synthesis_job.sh"
full_logs="$FULL_BATCH/logs"
mkdt_logs="$MKDT_MULTI_BATCH/logs"
all_targets='CIFAR10,CIFAR100,TinyImageNet,Aircraft,CUB2011,Dogs,Flowers'

# Four missing Full Data ResNet18 configurations. SSL is trained once; each
# dependent evaluation runs 15 independent label/probe seeds.
full_configs=(
    'CIFAR10 barlowtwins 600 03:00:00'
    'CIFAR100 barlowtwins 600 03:00:00'
    'TinyImageNet simclr 300 03:00:00'
    'TinyImageNet barlowtwins 600 05:30:00'
)
for entry in "${full_configs[@]}"; do
    read -r source ssl epochs limit <<< "$entry"
    config="full_${source}_ResNet18_${ssl}"
    short=${source/TinyImageNet/Tiny}
    submit_gpu full_pretrain "remain_full_${short}_R18_${ssl}" "$limit" 4G '' \
        "$full_logs" "$full_job" pretrain "$source" "$ssl" "$epochs" "$config" "$all_targets"
    pretrain_id=$LAST_JOB_ID
    submit_gpu full_evaluate "remain_fev_${short}_R18_${ssl}" 01:00:00 4G \
        "afterok:$pretrain_id" "$full_logs" "$full_job" \
        evaluate "$source" "$ssl" "$epochs" "$config" "$all_targets"
done

declare -A synth_ids

# Recreate the deleted non-finite CIFAR VGG11 synthetic sets with the current
# finite-checking distillation driver.
for dataset in CIFAR10 CIFAR100; do
    for size in 1 2 5; do
        submit_gpu mkdt_distill "remain_syn_${dataset}_VGG_${size}" 01:30:00 8G '' \
            "$mkdt_logs" "$synthesis_job" distill "$dataset" VGG11 "$size" VGG11
        synth_ids[${dataset}_VGG11_${size}]=$LAST_JOB_ID
    done
done

# Tiny ImageNet ConvNet experts and sorted initializations are already valid.
for size in 1 2 5; do
    submit_gpu mkdt_distill "remain_syn_Tiny_Conv_${size}" 05:00:00 12G '' \
        "$mkdt_logs" "$synthesis_job" distill Tiny ConvNetD4 "$size" ConvNet
    synth_ids[Tiny_ConvNet_${size}]=$LAST_JOB_ID
done

# Tiny ImageNet VGG11 had no completed expert bank, so build one bank once,
# sort it once, and distill the three requested percentages.
tiny_vgg_experts=()
for start in 0 20 40 60 80; do
    submit_gpu mkdt_experts "remain_exp_Tiny_VGG_${start}" 03:00:00 20G '' \
        "$mkdt_logs" "$synthesis_job" experts Tiny VGG11 "$start"
    tiny_vgg_experts+=("$LAST_JOB_ID")
done
tiny_vgg_dependency=$(IFS=:; echo "${tiny_vgg_experts[*]}")
submit_gpu mkdt_sort remain_sort_Tiny_VGG 01:30:00 26G \
    "afterok:$tiny_vgg_dependency" "$mkdt_logs" "$synthesis_job" sort Tiny VGG11
tiny_vgg_sort=$LAST_JOB_ID
for size in 1 2 5; do
    submit_gpu mkdt_distill "remain_syn_Tiny_VGG_${size}" 05:00:00 12G \
        "afterok:$tiny_vgg_sort" "$mkdt_logs" "$synthesis_job" \
        distill Tiny VGG11 "$size" VGG11
    synth_ids[Tiny_VGG11_${size}]=$LAST_JOB_ID
done

declare -A sim_epochs=( [1]=5000 [2]=4000 [5]=3000 )
declare -A bt_epochs=( [1]=10000 [2]=8000 [5]=6000 )

submit_mkdt_chain() {
    local source=$1 generator=$2 model=$3 size=$4 ssl=$5 scope=$6 dependency=$7
    local epochs
    if [[ $ssl == simclr ]]; then epochs=${sim_epochs[$size]}; else epochs=${bt_epochs[$size]}; fi
    local short=${source/TinyImageNet/Tiny}
    local pre_limit=01:00:00
    if [[ $source == TinyImageNet && $model == ResNet18 ]]; then
        pre_limit=03:00:00
    elif [[ $source == TinyImageNet ]]; then
        pre_limit=01:30:00
    fi
    submit_gpu mkdt_pretrain "remain_pre_${short}_${generator}_${model}_${size}_${ssl}" \
        "$pre_limit" 4G "$dependency" "$mkdt_logs" "$mkdt_job" \
        pretrain "$source" "$generator" "$model" "$size" "$ssl" "$epochs" "$scope"
    local pretrain_id=$LAST_JOB_ID
    submit_gpu mkdt_evaluate "remain_ev_${short}_${generator}_${model}_${size}_${ssl}" \
        01:00:00 4G "afterok:$pretrain_id" "$mkdt_logs" "$mkdt_job" \
        evaluate "$source" "$generator" "$model" "$size" "$ssl" "$epochs" "$scope"
}

# Existing CIFAR-10 ConvNet encoders only need their blank downstream cells.
for size in 1 2 5; do
    for ssl in simclr barlowtwins; do
        if [[ $ssl == simclr ]]; then epochs=${sim_epochs[$size]}; else epochs=${bt_epochs[$size]}; fi
        submit_gpu mkdt_evaluate "remain_ev_C10_Conv_down_${size}_${ssl}" 01:00:00 4G '' \
            "$mkdt_logs" "$mkdt_job" evaluate CIFAR10 ConvNet ConvNet \
            "$size" "$ssl" "$epochs" downstream
    done
done

# Missing CIFAR-10/CIFAR-100 VGG11 main cells; CIFAR-10 also has a VGG11
# downstream table, so that evaluation includes all targets.
for source in CIFAR10 CIFAR100; do
    scope=main
    [[ $source == CIFAR10 ]] && scope=all
    for size in 1 2 5; do
        for ssl in simclr barlowtwins; do
            submit_mkdt_chain "$source" VGG11 VGG11 "$size" "$ssl" "$scope" \
                "afterok:${synth_ids[${source}_VGG11_${size}]}"
        done
    done
done

# Tiny ImageNet main cells for ConvNet, VGG11, and ResNet18. The ResNet18
# evaluation also supplies the only downstream backbone requested there.
for size in 1 2 5; do
    for ssl in simclr barlowtwins; do
        submit_mkdt_chain TinyImageNet ConvNet ConvNet "$size" "$ssl" main \
            "afterok:${synth_ids[Tiny_ConvNet_${size}]}"
        submit_mkdt_chain TinyImageNet ConvNet ResNet18 "$size" "$ssl" all \
            "afterok:${synth_ids[Tiny_ConvNet_${size}]}"
        submit_mkdt_chain TinyImageNet VGG11 VGG11 "$size" "$ssl" main \
            "afterok:${synth_ids[Tiny_VGG11_${size}]}"
    done
done

printf 'Submitted %d one-CPU/one-GPU jobs; manifest: %s\n' "$gpu_index" "$manifest"
