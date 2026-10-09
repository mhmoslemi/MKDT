#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
export PROJECT_ROOT
cd "$PROJECT_ROOT"

if (( $# > 1 )) || (( $# == 1 )) && [[ $1 != --dry-run ]]; then
    echo "Usage: bash $0 [--dry-run]" >&2
    exit 2
fi
dry_run=0
(( $# == 0 )) || dry_run=1

OLD_DOWNSTREAM_BATCH="${OLD_DOWNSTREAM_BATCH:-$PROJECT_ROOT/benchmarks/downstream/results/batch_20261009T043709Z_zC8EPD}"
CIFAR100_MAIN_BATCH="${CIFAR100_MAIN_BATCH:-$PROJECT_ROOT/benchmarks/cifar100/results/batch_20261009T163141Z_eMeTVJ}"
TINYIMAGENET_MAIN_BATCH="${TINYIMAGENET_MAIN_BATCH:-$PROJECT_ROOT/benchmarks/tinyimagenet/results/batch_20261009T163143Z_D8Lnde}"
export CIFAR100_SOURCE_CACHE="${CIFAR100_SOURCE_CACHE:-$OLD_DOWNSTREAM_BATCH/targets/CIFAR100.pt}"
export TINYIMAGENET_SOURCE_CACHE="${TINYIMAGENET_SOURCE_CACHE:-$OLD_DOWNSTREAM_BATCH/targets/TinyImageNet.pt}"
export CIFAR100_SELECTION_DIR="${CIFAR100_SELECTION_DIR:-$CIFAR100_MAIN_BATCH/selections}"
export TINYIMAGENET_SELECTION_DIR="${TINYIMAGENET_SELECTION_DIR:-$TINYIMAGENET_MAIN_BATCH/selections}"

JOB_ARCHIVE="$PROJECT_ROOT/benchmarks/cross_downstream/jobs.tar"
SOURCES="$PROJECT_ROOT/benchmarks/cross_downstream/sources.tsv"
CONFIGS="$PROJECT_ROOT/benchmarks/cross_downstream/configs.tsv"
for path in "$JOB_ARCHIVE" "$SOURCES" "$CONFIGS" \
            benchmarks/cross_downstream/prepare_cifar10.sh \
            benchmarks/cross_downstream/report.sh; do
    [[ -f $path ]] || { echo "Missing generated file: $path" >&2; exit 1; }
done

if (( dry_run )); then
    CROSS_RUN_ROOT="$PROJECT_ROOT/benchmarks/cross_downstream/results/<new-batch>"
    CROSS_CODE_ROOT="$PROJECT_ROOT"
else
    command -v sbatch >/dev/null
    [[ -f ${VENV_ACTIVATE:-$HOME/ENV/bin/activate} ]] || { echo 'Python environment is missing.' >&2; exit 1; }
    data_root="${DATA_ROOT:-$SCRATCH/data}"
    [[ -f $data_root/cifar-10-batches-py/data_batch_1 ]] || { echo "Missing CIFAR-10 under $data_root" >&2; exit 1; }
    for path in "$CIFAR100_SOURCE_CACHE" "$TINYIMAGENET_SOURCE_CACHE"; do
        [[ -f $path ]] || { echo "Missing validated source cache: $path" >&2; exit 1; }
    done
    for directory in "$CIFAR100_SELECTION_DIR" "$TINYIMAGENET_SELECTION_DIR"; do
        [[ $(rg --files "$directory" | wc -l) -eq 10 ]] || { echo "Expected 10 source selections in $directory" >&2; exit 1; }
    done
    for target in CIFAR100 TinyImageNet Aircraft CUB2011 Dogs Flowers; do
        [[ -f $OLD_DOWNSTREAM_BATCH/targets/$target.pt ]] || { echo "Missing validated target cache: $target" >&2; exit 1; }
    done

    mkdir -p benchmarks/cross_downstream/results
    CROSS_RUN_ROOT=$(mktemp -d "$PROJECT_ROOT/benchmarks/cross_downstream/results/batch_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")
    CROSS_CODE_ROOT="$CROSS_RUN_ROOT/code"
    mkdir -p "$CROSS_RUN_ROOT/logs" "$CROSS_RUN_ROOT/targets" \
             "$CROSS_CODE_ROOT/OT-SSL-DD" "$CROSS_CODE_ROOT/benchmarks/cross_downstream"
    for target in CIFAR100 TinyImageNet Aircraft CUB2011 Dogs Flowers; do
        ln -s "$OLD_DOWNSTREAM_BATCH/targets/$target.pt" "$CROSS_RUN_ROOT/targets/$target.pt"
    done
    cp OT-SSL-DD/utils.py OT-SSL-DD/networks.py OT-SSL-DD/training_epochs.json "$CROSS_CODE_ROOT/OT-SSL-DD/"
    cp benchmarks/benchmark_cifar10.py benchmarks/benchmark_dataset.py \
       benchmarks/benchmark_cross_dataset.py benchmarks/downstream_data.py \
       "$CROSS_CODE_ROOT/benchmarks/"
    cp benchmarks/cross_downstream/report.py benchmarks/cross_downstream/configs.tsv \
       benchmarks/cross_downstream/sources.tsv benchmarks/cross_downstream/protocol.txt \
       "$CROSS_CODE_ROOT/benchmarks/cross_downstream/"
    cp "$CONFIGS" "$SOURCES" "$CROSS_RUN_ROOT/"
    cp "$JOB_ARCHIVE" "$CROSS_RUN_ROOT/jobs.tar"
    tar -czf "$CROSS_RUN_ROOT/source.tar.gz" \
        benchmarks/benchmark_cifar10.py benchmarks/benchmark_dataset.py \
        benchmarks/benchmark_cross_dataset.py benchmarks/downstream_data.py \
        benchmarks/cross_downstream/environment.sh benchmarks/cross_downstream/make_jobs.py \
        benchmarks/cross_downstream/report.py benchmarks/cross_downstream/generate.sh \
        benchmarks/cross_downstream/prepare_cifar10.sh benchmarks/cross_downstream/report.sh \
        benchmarks/cross_downstream/configs.tsv benchmarks/cross_downstream/sources.tsv \
        benchmarks/cross_downstream/protocol.txt benchmarks/cross_downstream/jobs.tar \
        benchmarks/submit_cross_downstream.sh \
        OT-SSL-DD/utils.py OT-SSL-DD/networks.py OT-SSL-DD/training_epochs.json
fi
export CROSS_RUN_ROOT CROSS_CODE_ROOT

dry_id=900000
submit() {
    local script=$1
    local account=$2
    shift 2
    local args=(sbatch --parsable --account="$account" --chdir="$PROJECT_ROOT" \
                --output="$CROSS_RUN_ROOT/logs/%x-%j.out" "$@")
    if (( dry_run )); then
        printf '%q ' "${args[@]}" >&2
        printf '%s\n' "$script" >&2
        SUBMITTED_ID=$((++dry_id))
    else
        if [[ $script == benchmarks/cross_downstream/jobs/* ]]; then
            SUBMITTED_ID=$(tar -xOf "$JOB_ARCHIVE" "$script" | "${args[@]}")
        else
            SUBMITTED_ID=$("${args[@]}" "$script")
        fi
        SUBMITTED_ID=${SUBMITTED_ID%%;*}
        printf '%s\t%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "$SUBMITTED_ID" "$account" "$script" >> "$CROSS_RUN_ROOT/submitted.tsv"
        printf 'Submitted %s on %s: %s\n' "$(basename "$script")" "$account" "$SUBMITTED_ID" >&2
    fi
}

submit benchmarks/cross_downstream/prepare_cifar10.sh aip-boyuwang
prepare_id=$SUBMITTED_ID

declare -A encoder_ids=() evaluate_accounts=() evaluate_scripts=()
while IFS=$'\t' read -r source_id source model ssl method size epochs runs estimate limit pre_account eval_account pre_script eval_script; do
    [[ $source_id != source_id ]] || continue
    pre_script=${pre_script%$'\r'}
    eval_script=${eval_script%$'\r'}
    submit "$pre_script" "$pre_account"
    encoder_ids[$source_id]=$SUBMITTED_ID
    evaluate_accounts[$source_id]=$eval_account
    evaluate_scripts[$source_id]=$eval_script
done < "$SOURCES"

evaluation_ids=()
while IFS= read -r source_id; do
    submit "${evaluate_scripts[$source_id]}" "${evaluate_accounts[$source_id]}" \
        --dependency="afterok:$prepare_id:${encoder_ids[$source_id]}" --kill-on-invalid-dep=yes
    evaluation_ids+=("$SUBMITTED_ID")
done < <(awk -F'\t' 'NR>1 {gsub(/\r/, "", $1); print $1}' "$SOURCES")

baseline_account=$(awk -F'\t' 'NR>1 && $7=="no_pretrain" {gsub(/\r/, "", $15); print $15; exit}' "$CONFIGS")
submit benchmarks/cross_downstream/jobs/evaluate/baseline.sh "$baseline_account" \
    --dependency="afterok:$prepare_id" --kill-on-invalid-dep=yes
evaluation_ids+=("$SUBMITTED_ID")

dependency=$(IFS=:; echo "${evaluation_ids[*]}")
submit benchmarks/cross_downstream/report.sh aip-boyuwang --dependency="afterany:$dependency"
report_id=$SUBMITTED_ID

lu=$(awk -F'\t' 'NR>1 {if($11=="aip-yiweilu")n++; if($12=="aip-yiweilu")n++} END{print n+0}' "$SOURCES")
[[ $baseline_account != aip-yiweilu ]] || lu=$((lu + 1))
gpu_jobs=$((${#evaluation_ids[@]} + ${#encoder_ids[@]}))
wang=$((gpu_jobs - lu))
printf '%s GPU jobs: %s Wang, %s Lu (%.2f%% Lu), plus CIFAR-10 preparation and report jobs.\nResults: %s\nReport job: %s\n' \
    "$gpu_jobs" "$wang" "$lu" "$(awk -v n="$lu" -v total="$gpu_jobs" 'BEGIN {print 100*n/total}')" \
    "$CROSS_RUN_ROOT" "$report_id"
