#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
export PROJECT_ROOT
cd "$PROJECT_ROOT"

dry_run=0
if (( $# == 1 )) && [[ $1 == --dry-run ]]; then
    dry_run=1
elif (( $# != 0 )); then
    echo "Usage: bash $0 [--dry-run]" >&2
    exit 2
fi

ACCOUNT="${ACCOUNT:-aip-boyuwang}"
VENV_ACTIVATE="${VENV_ACTIVATE:-$HOME/ENV/bin/activate}"
export ACCOUNT VENV_ACTIVATE

c10_cache="$PROJECT_ROOT/benchmarks/cross_downstream/results/batch_20261009T171449Z_o91BKP/targets/CIFAR10.pt"
c100_cache="$PROJECT_ROOT/benchmarks/downstream/results/batch_20261009T043709Z_zC8EPD/targets/CIFAR100.pt"
[[ -f $c10_cache ]] || { echo "Missing CIFAR-10 cache: $c10_cache" >&2; exit 1; }
[[ -f $c100_cache ]] || { echo "Missing CIFAR-100 cache: $c100_cache" >&2; exit 1; }
[[ -f $VENV_ACTIVATE ]] || { echo "Missing Python environment: $VENV_ACTIVATE" >&2; exit 1; }
command -v jq >/dev/null || { echo "jq is required to read training_epochs.json" >&2; exit 1; }
epoch_file="$PROJECT_ROOT/OT-SSL-DD/training_epochs.json"
jq -e '.protocol == "v4_pt0p2_long_full_dataset_specific"' "$epoch_file" >/dev/null

results_root="$PROJECT_ROOT/benchmarks/vulcan_subset_refresh/results"
mkdir -p "$results_root"
SUBSET_BATCH="$(mktemp -d "$results_root/batch_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")"
SUBSET_CODE_ROOT="$SUBSET_BATCH/code"
TARGET_CACHE_ROOT="$SUBSET_BATCH/targets"
export SUBSET_BATCH SUBSET_CODE_ROOT TARGET_CACHE_ROOT

mkdir -p "$SUBSET_BATCH/logs" "$SUBSET_BATCH/runs" "$SUBSET_BATCH/selections" \
    "$SUBSET_CODE_ROOT/benchmarks/vulcan_subset_refresh" \
    "$SUBSET_CODE_ROOT/OT-SSL-DD" "$TARGET_CACHE_ROOT"
ln -s "$c10_cache" "$TARGET_CACHE_ROOT/CIFAR10.pt"
ln -s "$c100_cache" "$TARGET_CACHE_ROOT/CIFAR100.pt"

cp benchmarks/benchmark_cifar10.py benchmarks/benchmark_dataset.py \
    benchmarks/benchmark_subset_refresh.py benchmarks/downstream_data.py \
    "$SUBSET_CODE_ROOT/benchmarks/"
cp benchmarks/vulcan_subset_refresh/environment.sh \
    benchmarks/vulcan_subset_refresh/job.sh \
    "$SUBSET_CODE_ROOT/benchmarks/vulcan_subset_refresh/"
cp OT-SSL-DD/utils.py OT-SSL-DD/networks.py OT-SSL-DD/training_epochs.json \
    "$SUBSET_CODE_ROOT/OT-SSL-DD/"
cp OT-SSL-DD/training_epochs.json "$SUBSET_BATCH/training_epochs.json"

tar -czf "$SUBSET_BATCH/source.tar.gz" \
    benchmarks/benchmark_cifar10.py benchmarks/benchmark_dataset.py \
    benchmarks/benchmark_subset_refresh.py benchmarks/downstream_data.py \
    benchmarks/vulcan_subset_refresh/environment.sh \
    benchmarks/vulcan_subset_refresh/job.sh \
    benchmarks/vulcan_subset_refresh/submit.sh \
    OT-SSL-DD/utils.py OT-SSL-DD/networks.py OT-SSL-DD/training_epochs.json

printf 'submitted_utc\tjob_id\taccount\tphase\tconfig\n' > "$SUBSET_BATCH/submitted.tsv"
printf 'config\tsource\tmethod\tmodel\tssl_method\tsubset_percentage\tssl_epochs\truns\tlabel_1_probe_epochs\tlabel_5_probe_epochs\tpretrain_job\tevaluate_job\n' \
    > "$SUBSET_BATCH/configs.tsv"

submit_job() {
    local phase=$1
    local name=$2
    local limit=$3
    local array=$4
    local dependency=$5
    shift 5
    local log_pattern="$SUBSET_BATCH/logs/%x-%j.out"
    [[ -n $array ]] && log_pattern="$SUBSET_BATCH/logs/%x-%A_%a.out"
    local args=(sbatch --parsable --account="$ACCOUNT" --job-name="$name" \
        --time="$limit" --chdir="$PROJECT_ROOT" \
        --output="$log_pattern")
    [[ -n $array ]] && args+=(--array="$array")
    [[ -n $dependency ]] && args+=(--dependency="$dependency" --kill-on-invalid-dep=yes)
    local id
    if (( dry_run )); then
        printf 'DRY-RUN: ' >&2
        printf '%q ' "${args[@]}" >&2
        printf '%q ' "$SUBSET_CODE_ROOT/benchmarks/vulcan_subset_refresh/job.sh" "$@" >&2
        printf '\n' >&2
        id=$((9000000 + $(wc -l < "$SUBSET_BATCH/submitted.tsv")))
    else
        id=$("${args[@]}" "$SUBSET_CODE_ROOT/benchmarks/vulcan_subset_refresh/job.sh" "$@")
        id=${id%%;*}
    fi
    printf '%s\t%s\t%s\t%s\t%s\n' \
        "$(date -u +%FT%TZ)" "$id" "$ACCOUNT" "$phase" "$name" \
        >> "$SUBSET_BATCH/submitted.tsv"
    printf '%s\n' "$id"
}

pretrain_limit() {
    local model=$1
    local subset=$2
    if [[ $model == ResNet18 ]]; then
        case "$subset" in
            0.2) printf '0-03:00:00' ;;
            1)   printf '0-04:00:00' ;;
            2)   printf '0-05:00:00' ;;
            5)   printf '0-08:00:00' ;;
        esac
    else
        case "$subset" in
            0.2|1) printf '0-02:00:00' ;;
            2)     printf '0-03:00:00' ;;
            5)     printf '0-04:00:00' ;;
        esac
    fi
}

declare -A prepare_ids
for source in CIFAR10 CIFAR100; do
    short=c10
    [[ $source == CIFAR100 ]] && short=c100
    prepare_ids[$source]=$(submit_job prepare "vsr_${short}_select" 0-02:00:00 '0-9%4' '' \
        prepare "$source")
done

pretrain_jobs=0
evaluation_jobs=0
for source in CIFAR10 CIFAR100; do
    short=c10
    [[ $source == CIFAR100 ]] && short=c100
    probe1=$(jq -er --arg source "$source" \
        '.evaluation_by_dataset[$source].linear_probe_epochs_by_label_percentage["1"]' \
        "$epoch_file")
    probe5=$(jq -er --arg source "$source" \
        '.evaluation_by_dataset[$source].linear_probe_epochs_by_label_percentage["5"]' \
        "$epoch_file")
    for method in random kmeans; do
        method_short=r
        [[ $method == kmeans ]] && method_short=k
        dependency=''
        [[ $method == kmeans ]] && dependency="afterok:${prepare_ids[$source]}"
        for model in ConvNet VGG11 ResNet18; do
            case "$model" in
                ConvNet) model_short=cn ;;
                VGG11) model_short=vg ;;
                ResNet18) model_short=r18 ;;
            esac
            for ssl in simclr barlowtwins; do
                ssl_short=s
                [[ $ssl == barlowtwins ]] && ssl_short=b
                for subset in 0.2 1 2 5; do
                    case "$subset" in
                        0.2) subset_token=0p2 ;;
                        1) subset_token=1 ;;
                        2) subset_token=2 ;;
                        5) subset_token=5 ;;
                    esac
                    epochs=$(jq -er --arg method "$method" --arg subset "$subset" \
                        '.pretraining[($method + "_subset_epochs_by_data_percentage")][$subset]' \
                        "$epoch_file")
                    config="${source}_${method}_${model}_${ssl}_pt${subset_token}"
                    name="vsr_${short}_${method_short}_${model_short}_${ssl_short}_${subset_token}"
                    limit=$(pretrain_limit "$model" "$subset")
                    pretrain_id=$(submit_job pretrain "$name" "$limit" '0-9%4' "$dependency" \
                        pretrain "$source" "$method" "$model" "$ssl" "$subset" "$epochs" "$config")
                    pretrain_jobs=$((pretrain_jobs + 1))
                    eval_limit=0-02:00:00
                    [[ $model == ResNet18 ]] && eval_limit=0-03:00:00
                    eval_id=$(submit_job evaluate "${name}_ev" "$eval_limit" '' "afterok:$pretrain_id" \
                        evaluate "$source" "$method" "$model" "$ssl" "$subset" "$epochs" "$config" \
                        "$probe1" "$probe5")
                    evaluation_jobs=$((evaluation_jobs + 1))
                    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t10\t%s\t%s\t%s\t%s\n' \
                        "$config" "$source" "$method" "$model" "$ssl" "$subset" "$epochs" \
                        "$probe1" "$probe5" "$pretrain_id" "$eval_id" >> "$SUBSET_BATCH/configs.tsv"
                done
            done
        done
    done
done

mode=SUBMITTED
(( dry_run )) && mode=DRY-RUN
printf '%s batch: %s\nSelection arrays: 2 (20 tasks)\nPretraining arrays: %s (960 checkpoint tasks)\nEvaluation jobs: %s (both 1%%/1000 and 5%%/500 probes)\nAccount: %s\n' \
    "$mode" "$SUBSET_BATCH" "$pretrain_jobs" "$evaluation_jobs" "$ACCOUNT"
