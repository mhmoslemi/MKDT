#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
MKDT_MULTI_BATCH="${1:?Usage: $0 <existing-mkdt-batch>}"
MKDT_MULTI_BATCH="$(cd "$MKDT_MULTI_BATCH" && pwd -P)"
MKDT_CODE_ROOT="$MKDT_MULTI_BATCH/code"
MKDT_SYNTH_ROOT="$MKDT_MULTI_BATCH/synthetic"
TARGET_CACHE_ROOT="$MKDT_MULTI_BATCH/targets"
DATA_ROOT="$SCRATCH/data"
export PROJECT_ROOT MKDT_MULTI_BATCH MKDT_CODE_ROOT MKDT_SYNTH_ROOT TARGET_CACHE_ROOT DATA_ROOT
cd "$PROJECT_ROOT"

[[ -f "$MKDT_SYNTH_ROOT/work/target_rep/barlow_twins/Tiny_target_rep_train.pt" ]] || {
    echo 'Missing completed Tiny ImageNet target representations' >&2
    exit 1
}
[[ -f "$MKDT_CODE_ROOT/networks.py" ]] || { echo 'Missing batch code snapshot' >&2; exit 1; }

printf 'repair_submitted_utc\tjob_id\taccount\tclass\tname\n' > "$MKDT_MULTI_BATCH/tiny_resnet_repair.tsv"
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
    printf '%s\t%s\t%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "$LAST_JOB_ID" "$account" "$class" "$name" \
        | tee -a "$MKDT_MULTI_BATCH/tiny_resnet_repair.tsv" >> "$MKDT_MULTI_BATCH/submitted.tsv"
    printf 'Submitted repair %s on %s: %s\n' "$name" "$account" "$LAST_JOB_ID" >&2
}

synthesis="$PROJECT_ROOT/benchmarks/mkdt_multidataset/synthesis_job.sh"
benchmark="$PROJECT_ROOT/benchmarks/mkdt_multidataset/benchmark_job.sh"

expert_ids=()
for start in 0 20 40 60 80; do
    submit_gpu repair_experts "mkdt_fix_exp_Tiny_ResNet18BN_${start}" 0-10:00:00 24G '' \
        "$synthesis" experts Tiny ResNet18 "$start"
    expert_ids+=("$LAST_JOB_ID")
done
expert_dependency=$(IFS=:; echo "${expert_ids[*]}")
submit_gpu repair_sort mkdt_fix_sort_Tiny_ResNet18BN 0-08:00:00 24G \
    "afterok:$expert_dependency" "$synthesis" sort Tiny ResNet18
sort_id=$LAST_JOB_ID

declare -A distill_ids
for size in 1 2 5; do
    submit_gpu repair_distill "mkdt_fix_syn_Tiny_ResNet18BN_${size}" 1-00:00:00 24G \
        "afterok:$sort_id" "$synthesis" distill Tiny ResNet18 "$size" ResNet18BN
    distill_ids[$size]=$LAST_JOB_ID
done

declare -A sim_epochs=( [1]=5000 [2]=4000 [5]=3000 )
declare -A bt_epochs=( [1]=10000 [2]=8000 [5]=6000 )
declare -A tiny_time=(
    [ConvNet_1_simclr]=0-01:05:00 [ConvNet_2_simclr]=0-01:35:00 [ConvNet_5_simclr]=0-02:55:00
    [ConvNet_1_barlowtwins]=0-02:30:00 [ConvNet_2_barlowtwins]=0-03:55:00 [ConvNet_5_barlowtwins]=0-07:10:00
    [ResNet18_1_simclr]=0-04:00:00 [ResNet18_2_simclr]=0-06:15:00 [ResNet18_5_simclr]=0-11:35:00
    [ResNet18_1_barlowtwins]=0-07:45:00 [ResNet18_2_barlowtwins]=0-12:25:00 [ResNet18_5_barlowtwins]=0-23:10:00
)

new_evaluation_ids=()
for model in ConvNet ResNet18; do
    for size in 1 2 5; do
        for ssl in simclr barlowtwins; do
            if [[ $ssl == simclr ]]; then epochs=${sim_epochs[$size]}; else epochs=${bt_epochs[$size]}; fi
            limit=${tiny_time[${model}_${size}_${ssl}]}
            submit_gpu repair_pretrain "mkfixpre_Tiny_${model}_${size}_${ssl}" "$limit" 15G \
                "afterok:${distill_ids[$size]}" \
                "$benchmark" pretrain TinyImageNet ResNet18BN "$model" "$size" "$ssl" "$epochs"
            pretrain_id=$LAST_JOB_ID
            eval_limit=0-00:30:00
            [[ $model == ResNet18 ]] && eval_limit=0-01:30:00
            submit_gpu repair_evaluate "mkfixev_Tiny_${model}_${size}_${ssl}" "$eval_limit" 15G \
                "afterok:$pretrain_id" \
                "$benchmark" evaluate TinyImageNet ResNet18BN "$model" "$size" "$ssl" "$epochs"
            new_evaluation_ids+=("$LAST_JOB_ID")
        done
    done
done

# Replace the old report, whose dependency graph contains the canceled chain,
# with a report that waits for the unaffected evaluations and this repair.
scancel 1576505 2>/dev/null || true
mapfile -t unaffected_evaluations < <(
    awk -F '\t' 'NR > 1 && $4 == "evaluate" && $5 !~ /^mkev_TinyImageNet_ResNet18BN_/ {print $2}' \
        "$MKDT_MULTI_BATCH/submitted.tsv")
all_evaluations=("${unaffected_evaluations[@]}" "${new_evaluation_ids[@]}")
dependency=$(IFS=:; echo "${all_evaluations[*]}")
report_id=$(sbatch --parsable --account=aip-boyuwang --job-name=mkdt_dual_tables_repaired \
    --chdir="$PROJECT_ROOT" --output="$MKDT_MULTI_BATCH/logs/%x-%j.out" \
    --dependency="afterany:$dependency" \
    "$PROJECT_ROOT/benchmarks/mkdt_multidataset/report.sh")
report_id=${report_id%%;*}
printf '%s\t%s\t%s\trepair_report\t%s\n' "$(date -u +%FT%TZ)" "$report_id" \
    aip-boyuwang mkdt_dual_tables_repaired \
    | tee -a "$MKDT_MULTI_BATCH/tiny_resnet_repair.tsv" >> "$MKDT_MULTI_BATCH/submitted.tsv"

printf 'Batch: %s\nRepair GPU jobs: %s\nReplacement evaluation chains: %s\nReport: %s\n' \
    "$MKDT_MULTI_BATCH" "$gpu_index" "${#new_evaluation_ids[@]}" "$report_id"
