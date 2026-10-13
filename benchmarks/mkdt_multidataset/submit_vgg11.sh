#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
MKDT_MULTI_BATCH="${1:-$PROJECT_ROOT/benchmarks/mkdt_multidataset/results/batch_20261010T160046Z_Y6dqZS}"
MKDT_MULTI_BATCH="$(cd -- "$MKDT_MULTI_BATCH" && pwd -P)"
MKDT_CODE_ROOT="$MKDT_MULTI_BATCH/code"
MKDT_SYNTH_ROOT="$MKDT_MULTI_BATCH/synthetic"
TARGET_CACHE_ROOT="$MKDT_MULTI_BATCH/targets"
DATA_ROOT="$SCRATCH/data"
export PROJECT_ROOT MKDT_MULTI_BATCH MKDT_CODE_ROOT MKDT_SYNTH_ROOT TARGET_CACHE_ROOT DATA_ROOT
cd "$PROJECT_ROOT"

for dataset in CIFAR10 CIFAR100; do
    target="$MKDT_SYNTH_ROOT/work/target_rep/barlow_twins/${dataset}_target_rep_train.pt"
    [[ -f $target ]] || { echo "Missing completed target representations: $target" >&2; exit 1; }
done
[[ -f "$MKDT_CODE_ROOT/benchmarks/benchmark_mkdt_multidataset.py" ]] || {
    echo "Missing benchmark code snapshot" >&2
    exit 1
}
grep -q "'VGG11'" "$MKDT_CODE_ROOT/benchmarks/benchmark_mkdt_multidataset.py" || {
    echo "The batch benchmark snapshot does not support VGG11" >&2
    exit 1
}

manifest="$MKDT_MULTI_BATCH/vgg11_submitted.tsv"
if [[ ! -f $manifest ]]; then
    printf 'submitted_utc\tjob_id\taccount\tclass\tname\n' > "$manifest"
fi
gpu_index=0
LAST_JOB_ID=''
submit_gpu() {
    local class=$1 name=$2 limit=$3 memory=$4 dependency=$5 script=$6
    shift 6
    gpu_index=$((gpu_index + 1))
    local account=aip-boyuwang
    (( gpu_index % 10 == 0 )) && account=aip-yiweilu

    local existing
    existing=$(awk -F '\t' -v name="$name" 'NR > 1 && $5 == name {id=$2} END {print id}' "$manifest")
    if [[ -n $existing ]]; then
        LAST_JOB_ID=$existing
        printf 'Reusing recorded %s: %s\n' "$name" "$LAST_JOB_ID" >&2
        return
    fi

    local args=(sbatch --parsable --account="$account" --job-name="$name"
        --time="$limit" --mem="$memory" --chdir="$PROJECT_ROOT"
        --output="$MKDT_MULTI_BATCH/logs/%x-%j.out")
    [[ -n $dependency ]] && args+=(--dependency="$dependency" --kill-on-invalid-dep=yes)
    local id
    id=$("${args[@]}" "$script" "$@")
    LAST_JOB_ID=${id%%;*}
    local row
    row="$(date -u +%FT%TZ)\t$LAST_JOB_ID\t$account\t$class\t$name"
    printf '%b\n' "$row" >> "$manifest"
    printf '%b\n' "$row" >> "$MKDT_MULTI_BATCH/submitted.tsv"
    printf 'Submitted %s on %s: %s\n' "$name" "$account" "$LAST_JOB_ID" >&2
}

synthesis="$PROJECT_ROOT/benchmarks/mkdt_multidataset/synthesis_job.sh"
benchmark="$PROJECT_ROOT/benchmarks/mkdt_multidataset/benchmark_job.sh"
declare -A distill_ids

for dataset in CIFAR10 CIFAR100; do
    expert_ids=()
    for start in 0 20 40 60 80; do
        submit_gpu vgg11_experts "mkdt_exp_${dataset}_VGG11_${start}" 0-04:00:00 16G '' \
            "$synthesis" experts "$dataset" VGG11 "$start"
        expert_ids+=("$LAST_JOB_ID")
    done
    expert_dependency=$(IFS=:; echo "${expert_ids[*]}")
    submit_gpu vgg11_sort "mkdt_sort_${dataset}_VGG11" 0-02:00:00 16G \
        "afterok:$expert_dependency" "$synthesis" sort "$dataset" VGG11
    sort_id=$LAST_JOB_ID

    for size in 1 2 5; do
        submit_gpu vgg11_distill "mkdt_syn_${dataset}_VGG11_${size}" 1-00:00:00 24G \
            "afterok:$sort_id" "$synthesis" distill "$dataset" VGG11 "$size" VGG11
        distill_ids[${dataset}_${size}]=$LAST_JOB_ID
    done
done

declare -A sim_epochs=( [1]=5000 [2]=4000 [5]=3000 )
declare -A bt_epochs=( [1]=10000 [2]=8000 [5]=6000 )
declare -A pretrain_time=(
    [1_simclr]=0-00:40:00 [2_simclr]=0-01:05:00 [5_simclr]=0-01:50:00
    [1_barlowtwins]=0-01:15:00 [2_barlowtwins]=0-01:55:00 [5_barlowtwins]=0-03:30:00
)

evaluation_ids=()
for source in CIFAR10 CIFAR100; do
    for size in 1 2 5; do
        for ssl in simclr barlowtwins; do
            if [[ $ssl == simclr ]]; then epochs=${sim_epochs[$size]}; else epochs=${bt_epochs[$size]}; fi
            config_row="$source\tVGG11\tVGG11\t$size\t$ssl\t$epochs"
            if ! grep -Fqx "$(printf '%b' "$config_row")" "$MKDT_MULTI_BATCH/configs.tsv"; then
                printf '%b\n' "$config_row" >> "$MKDT_MULTI_BATCH/configs.tsv"
            fi
            submit_gpu vgg11_pretrain "mkvpre_${source}_${size}_${ssl}" \
                "${pretrain_time[${size}_${ssl}]}" 15G \
                "afterok:${distill_ids[${source}_${size}]}" \
                "$benchmark" pretrain "$source" VGG11 VGG11 "$size" "$ssl" "$epochs"
            pretrain_id=$LAST_JOB_ID
            submit_gpu vgg11_evaluate "mkvev_${source}_${size}_${ssl}" 0-00:30:00 15G \
                "afterok:$pretrain_id" \
                "$benchmark" evaluate "$source" VGG11 VGG11 "$size" "$ssl" "$epochs"
            evaluation_ids+=("$LAST_JOB_ID")
        done
    done
done

report_name=mkdt_vgg11_tables
existing_report=$(awk -F '\t' -v name="$report_name" 'NR > 1 && $5 == name {id=$2} END {print id}' "$manifest")
if [[ -n $existing_report ]]; then
    report_id=$existing_report
    printf 'Reusing recorded %s: %s\n' "$report_name" "$report_id" >&2
else
    report_dependencies=("${evaluation_ids[@]}")
    if [[ -f "$MKDT_MULTI_BATCH/split_resubmitted.tsv" ]]; then
        base_report=$(awk -F '\t' '$4 == "report" {id=$2} END {print id}' \
            "$MKDT_MULTI_BATCH/split_resubmitted.tsv")
        [[ -n $base_report ]] && report_dependencies+=("$base_report")
    fi
    dependency=$(IFS=:; echo "${report_dependencies[*]}")
    submitted=$(sbatch --parsable --account=aip-boyuwang --job-name="$report_name" \
        --chdir="$PROJECT_ROOT" --output="$MKDT_MULTI_BATCH/logs/%x-%j.out" \
        --dependency="afterany:$dependency" \
        "$PROJECT_ROOT/benchmarks/mkdt_multidataset/report.sh")
    report_id=${submitted%%;*}
    row="$(date -u +%FT%TZ)\t$report_id\taip-boyuwang\tvgg11_report\t$report_name"
    printf '%b\n' "$row" >> "$manifest"
    printf '%b\n' "$row" >> "$MKDT_MULTI_BATCH/submitted.tsv"
fi

printf 'Batch: %s\nVGG11 GPU jobs: %s\nVGG11 evaluation chains: %s\nReport: %s\n' \
    "$MKDT_MULTI_BATCH" "$gpu_index" "${#evaluation_ids[@]}" "$report_id"
