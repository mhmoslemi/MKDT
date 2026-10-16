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

fail() {
    echo "FATAL: $*" >&2
    exit 1
}

case "$PROJECT_ROOT" in
    /scratch/*|/project/*) ;;
    *) fail "resolved project path is not on scratch/project: $PROJECT_ROOT" ;;
esac

ACCOUNT=aip-boyuwang
VENV_ACTIVATE="${VENV_ACTIVATE:-/home/mmoslem3/ENV/bin/activate}"
MKDT_INPUT_BATCH="${MKDT_INPUT_BATCH:-$PROJECT_ROOT/benchmarks/mkdt_multidataset/results/batch_20261010T160046Z_Y6dqZS}"
MKDT_SYNTH_ROOT="$MKDT_INPUT_BATCH/synthetic"
TARGET_CACHE_ROOT="$MKDT_INPUT_BATCH/targets"
epoch_file="$PROJECT_ROOT/OT-SSL-DD/training_epochs.json"
export ACCOUNT VENV_ACTIVATE MKDT_INPUT_BATCH MKDT_SYNTH_ROOT TARGET_CACHE_ROOT

command -v jq >/dev/null || fail 'jq is required'
[[ -s $VENV_ACTIVATE ]] || fail "missing Python environment: $VENV_ACTIVATE"
[[ -s $epoch_file ]] || fail "missing epoch configuration: $epoch_file"
for size in 1 2; do
    epochs=$(jq -er --arg size "$size" \
        '.pretraining.mkdt_subset_epochs_by_data_percentage[$size]' "$epoch_file")
    [[ $epochs == 10000 ]] || fail "MKDT $size% epoch budget is $epochs, expected 10000"
done
for target in CIFAR10 CIFAR100 TinyImageNet Aircraft CUB2011 Dogs Flowers; do
    [[ -s $TARGET_CACHE_ROOT/$target.pt ]] || fail "missing target cache: $TARGET_CACHE_ROOT/$target.pt"
done

for generator in ConvNet VGG11; do
    for source in CIFAR10 CIFAR100 TinyImageNet; do
        token=$source
        final=5000
        if [[ $source == TinyImageNet ]]; then token=Tiny; final=10000; fi
        for size in 1 2; do
            output="$MKDT_SYNTH_ROOT/generated/$generator/$source/${token}_${size}pct"
            for kind in images labels synlr; do
                [[ -s $output/${kind}_${final}.pt ]] || \
                    fail "missing MKDT input: $output/${kind}_${final}.pt"
            done
        done
    done
done

for source_file in \
    benchmarks/benchmark_mkdt_multidataset.py \
    benchmarks/benchmark_mkdt_refresh.py \
    benchmarks/benchmark_cifar10.py \
    benchmarks/downstream_data.py \
    benchmarks/killarney_mkdt_refresh/environment.sh \
    benchmarks/killarney_mkdt_refresh/job.sh \
    benchmarks/killarney_mkdt_refresh/check.sh \
    OT-SSL-DD/utils.py \
    OT-SSL-DD/networks.py; do
    [[ -s $source_file ]] || fail "missing source file: $source_file"
done

if (( ! dry_run )); then
    command -v sbatch >/dev/null || fail 'sbatch is required'
    command -v sinfo >/dev/null || fail 'sinfo is required'
    command -v sacctmgr >/dev/null || fail 'sacctmgr is required'
    gpu_inventory=$(sinfo -h -o '%G' --Node | sort -u)
    grep -q 'gpu:l40s:' <<< "$gpu_inventory" || {
        printf 'Available GPU GRES:\n%s\n' "$gpu_inventory" >&2
        fail 'gpu:l40s is not advertised on this cluster'
    }
    accounts=$(sacctmgr show assoc user="$USER" format=Account -n -P | cut -d'|' -f1)
    grep -qx "$ACCOUNT" <<< "$accounts" || fail "account $ACCOUNT is not available to $USER"
fi

results_root="${MKDT_REFRESH_RESULTS_ROOT:-$PROJECT_ROOT/benchmarks/killarney_mkdt_refresh/results}"
mkdir -p "$results_root"
MKDT_REFRESH_BATCH="$(mktemp -d "$results_root/batch_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")"
MKDT_REFRESH_CODE_ROOT="$MKDT_REFRESH_BATCH/code"
export MKDT_REFRESH_BATCH MKDT_REFRESH_CODE_ROOT

mkdir -p \
    "$MKDT_REFRESH_BATCH/logs" \
    "$MKDT_REFRESH_BATCH/runs" \
    "$MKDT_REFRESH_CODE_ROOT/benchmarks/killarney_mkdt_refresh" \
    "$MKDT_REFRESH_CODE_ROOT/OT-SSL-DD"

cp benchmarks/benchmark_mkdt_multidataset.py \
    benchmarks/benchmark_mkdt_refresh.py \
    benchmarks/benchmark_cifar10.py \
    benchmarks/downstream_data.py \
    "$MKDT_REFRESH_CODE_ROOT/benchmarks/"
cp benchmarks/killarney_mkdt_refresh/environment.sh \
    benchmarks/killarney_mkdt_refresh/job.sh \
    benchmarks/killarney_mkdt_refresh/check.sh \
    "$MKDT_REFRESH_CODE_ROOT/benchmarks/killarney_mkdt_refresh/"
cp OT-SSL-DD/utils.py OT-SSL-DD/networks.py "$MKDT_REFRESH_CODE_ROOT/OT-SSL-DD/"
cp "$epoch_file" "$MKDT_REFRESH_BATCH/training_epochs.json"
printf '%s\n' 'mkdt_10k_probe1000_500_shared_encoder_v1' > "$MKDT_REFRESH_BATCH/protocol.txt"

printf 'config\tsource\tgenerator\tmodel\tsubset_percentage\tssl_method\tssl_epochs\tscope\tpretrain_job\tevaluate_job\n' \
    > "$MKDT_REFRESH_BATCH/configs.tsv"
printf 'table\tconfig\tsource\ttarget\tgenerator\tmodel\tsubset_percentage\tssl_method\tlabel_percentage\tresult_path\n' \
    > "$MKDT_REFRESH_BATCH/table_cells.tsv"
printf 'submitted_utc\tjob_id\taccount\tphase\tconfig\tdependency\n' \
    > "$MKDT_REFRESH_BATCH/submitted.tsv"

job_index=0
LAST_JOB_ID=''
submit_job() {
    local phase=$1 name=$2 limit=$3 dependency=$4
    shift 4
    job_index=$((job_index + 1))
    if (( dry_run )); then
        LAST_JOB_ID="DRY$(printf '%03d' "$job_index")"
        printf 'DRY-RUN %-46s phase=%-8s time=%s dependency=%s\n' \
            "$name" "$phase" "$limit" "${dependency:-none}"
    else
        args=(sbatch --parsable
            --account="$ACCOUNT"
            --job-name="$name"
            --nodes=1
            --ntasks=1
            --cpus-per-task=1
            --gres=gpu:l40s:1
            --time="$limit"
            --mem=8G
            --chdir="$PROJECT_ROOT"
            --output="$MKDT_REFRESH_BATCH/logs/%x-%j.out"
            --export="ALL,PROJECT_ROOT=$PROJECT_ROOT,MKDT_REFRESH_BATCH=$MKDT_REFRESH_BATCH,MKDT_REFRESH_CODE_ROOT=$MKDT_REFRESH_CODE_ROOT,MKDT_SYNTH_ROOT=$MKDT_SYNTH_ROOT,TARGET_CACHE_ROOT=$TARGET_CACHE_ROOT,VENV_ACTIVATE=$VENV_ACTIVATE")
        if [[ -n $dependency ]]; then
            args+=(--dependency="$dependency" --kill-on-invalid-dep=yes)
        fi
        submitted=$("${args[@]}" "$MKDT_REFRESH_CODE_ROOT/benchmarks/killarney_mkdt_refresh/job.sh" "$@")
        LAST_JOB_ID=${submitted%%;*}
        printf 'Submitted %s: %s\n' "$name" "$LAST_JOB_ID"
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$(date -u +%FT%TZ)" "$LAST_JOB_ID" "$ACCOUNT" "$phase" "$9" "$dependency" \
        >> "$MKDT_REFRESH_BATCH/submitted.tsv"
}

pretrain_limit() {
    local source=$1 model=$2 size=$3
    case "$source:$model:$size" in
        CIFAR10:ConvNet:1|CIFAR100:ConvNet:1) echo 0-01:00:00 ;;
        CIFAR10:ConvNet:2|CIFAR100:ConvNet:2) echo 0-01:15:00 ;;
        CIFAR10:VGG11:1|CIFAR100:VGG11:1) echo 0-01:05:00 ;;
        CIFAR10:VGG11:2|CIFAR100:VGG11:2) echo 0-01:20:00 ;;
        CIFAR10:ResNet18:1|CIFAR100:ResNet18:1) echo 0-01:30:00 ;;
        CIFAR10:ResNet18:2|CIFAR100:ResNet18:2) echo 0-02:00:00 ;;
        TinyImageNet:ConvNet:1) echo 0-01:10:00 ;;
        TinyImageNet:ConvNet:2) echo 0-01:35:00 ;;
        TinyImageNet:VGG11:1) echo 0-01:15:00 ;;
        TinyImageNet:VGG11:2) echo 0-01:35:00 ;;
        TinyImageNet:ResNet18:1) echo 0-01:40:00 ;;
        TinyImageNet:ResNet18:2) echo 0-02:40:00 ;;
        *) fail "no walltime for $source $model $size%" ;;
    esac
}

add_coverage() {
    local config=$1 source=$2 generator=$3 model=$4 size=$5 ssl=$6 scope=$7
    local token target labels table
    case "$source" in
        CIFAR10) token=cifar10 ;;
        CIFAR100) token=cifar100 ;;
        TinyImageNet) token=tinyimagenet ;;
    esac
    targets=("$source")
    if [[ $scope == all ]]; then
        case "$source" in
            CIFAR10) targets+=(CIFAR100 Aircraft CUB2011 Dogs Flowers TinyImageNet) ;;
            CIFAR100) targets+=(TinyImageNet CIFAR10 Aircraft CUB2011 Dogs Flowers) ;;
            TinyImageNet) targets+=(CIFAR10 CIFAR100 Aircraft CUB2011 Dogs Flowers) ;;
        esac
    fi
    for target in "${targets[@]}"; do
        table="${token}_downstream.tex"
        [[ $target == "$source" ]] && table="${token}_main.tex"
        for labels in 1 5; do
            printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                "$table" "$config" "$source" "$target" "$generator" "$model" \
                "$size" "$ssl" "$labels" \
                "runs/$config/evaluations/${target}_lbl${labels}/summary.json" \
                >> "$MKDT_REFRESH_BATCH/table_cells.tsv"
        done
    done
}

for source in CIFAR10 CIFAR100 TinyImageNet; do
    case "$source" in
        CIFAR10) source_short=c10 ;;
        CIFAR100) source_short=c100 ;;
        TinyImageNet) source_short=tiny ;;
    esac
    for model in ConvNet VGG11 ResNet18; do
        case "$model" in
            ConvNet) generator=ConvNet; model_short=cn ;;
            VGG11) generator=VGG11; model_short=vg ;;
            ResNet18) generator=ConvNet; model_short=r18 ;;
        esac
        scope=main
        if [[ $source == CIFAR10 || $model == ResNet18 ]]; then scope=all; fi
        for size in 1 2; do
            epochs=$(jq -er --arg size "$size" \
                '.pretraining.mkdt_subset_epochs_by_data_percentage[$size]' "$epoch_file")
            for ssl in simclr barlowtwins; do
                ssl_short=s
                [[ $ssl == barlowtwins ]] && ssl_short=b
                config="${source}_${generator}_${model}_size${size}_${ssl}_pt10000"
                name="kmr_${source_short}_${model_short}_${size}_${ssl_short}"
                limit=$(pretrain_limit "$source" "$model" "$size")
                submit_job pretrain "$name" "$limit" '' \
                    pretrain "$source" "$generator" "$model" "$size" "$ssl" "$epochs" "$scope" "$config"
                pretrain_id=$LAST_JOB_ID
                eval_limit=0-00:40:00
                [[ $scope == all ]] && eval_limit=0-01:30:00
                submit_job evaluate "${name}_ev" "$eval_limit" "afterok:$pretrain_id" \
                    evaluate "$source" "$generator" "$model" "$size" "$ssl" "$epochs" "$scope" "$config"
                evaluate_id=$LAST_JOB_ID
                printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                    "$config" "$source" "$generator" "$model" "$size" "$ssl" "$epochs" \
                    "$scope" "$pretrain_id" "$evaluate_id" >> "$MKDT_REFRESH_BATCH/configs.tsv"
                add_coverage "$config" "$source" "$generator" "$model" "$size" "$ssl" "$scope"
            done
        done
    done
done

config_count=$(( $(wc -l < "$MKDT_REFRESH_BATCH/configs.tsv") - 1 ))
cell_count=$(( $(wc -l < "$MKDT_REFRESH_BATCH/table_cells.tsv") - 1 ))
table_count=$(tail -n +2 "$MKDT_REFRESH_BATCH/table_cells.tsv" | cut -f1 | sort -u | wc -l)
unique_results=$(tail -n +2 "$MKDT_REFRESH_BATCH/table_cells.tsv" | cut -f10 | sort -u | wc -l)
[[ $job_index == 72 ]] || fail "internal error: $job_index jobs, expected 72"
[[ $config_count == 36 ]] || fail "internal error: $config_count configs, expected 36"
[[ $cell_count == 312 ]] || fail "internal error: $cell_count table cells, expected 312"
[[ $table_count == 6 ]] || fail "internal error: $table_count tables, expected 6"
[[ $unique_results == 312 ]] || fail "internal error: $unique_results results, expected 312"

mode=SUBMITTED
(( dry_run )) && mode=DRY-RUN
printf '%s batch: %s\nAccount: %s\nJobs: 72 (36 saved SSL checkpoints + 36 dependent evaluations)\nUnique evaluated cells: 312 across all 6 tables\nSSL epochs: 10000 for 1%% and 2%%\nProbe epochs: 1000 (1%% labels), 500 (5%% labels)\n' \
    "$mode" "$MKDT_REFRESH_BATCH" "$ACCOUNT"
