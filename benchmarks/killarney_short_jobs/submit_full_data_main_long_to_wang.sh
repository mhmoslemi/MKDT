#!/bin/bash
set -euo pipefail

# Fresh main-table full-data runs:
#   * CIFAR-10 and CIFAR-100
#   * ConvNet, VGG11, and ResNet18
#   * SimCLR: 1000 SSL epochs
#   * Barlow Twins: 1500 SSL epochs
#   * 10 encoder seeds, then 1%/5% same-dataset linear probes

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
PROJECT_ROOT=$(readlink -f "$SCRIPT_DIR/../..")
BUNDLE="$PROJECT_ROOT/benchmarks/killarney_short_jobs"
WORKER="$BUNDLE/full_data_main_long_job.sh"
ENVIRONMENT="$BUNDLE/full_data_main_long_environment.sh"
AUDIT="$BUNDLE/check_full_data_main_long.sh"
ACCOUNT=${ACCOUNT:-aip-boyuwang}

TARGET_SOURCE_BATCH=${TARGET_SOURCE_BATCH:-$PROJECT_ROOT/benchmarks/cross_downstream/results/batch_20261009T171449Z_o91BKP}
RESULTS_ROOT=${MAINLONG_RESULTS_ROOT:-$PROJECT_ROOT/benchmarks/full_data_main_long/results}
VENV_ACTIVATE=${VENV_ACTIVATE:-$HOME/ENV/bin/activate}

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
    *) fail "project root is not on scratch/project: $PROJECT_ROOT" ;;
esac

cd "$PROJECT_ROOT"
for path in \
    "$WORKER" \
    "$ENVIRONMENT" \
    "$AUDIT" \
    benchmarks/benchmark_full_data_main_long.py \
    benchmarks/benchmark_full_refresh.py \
    benchmarks/benchmark_cifar10.py \
    benchmarks/benchmark_dataset.py \
    benchmarks/downstream_data.py \
    OT-SSL-DD/utils.py \
    OT-SSL-DD/networks.py; do
    [[ -s $path ]] || fail "missing required file: $path"
done
[[ -s $VENV_ACTIVATE ]] || fail "missing Python environment: $VENV_ACTIVATE"

for target in CIFAR10 CIFAR100; do
    [[ -s $TARGET_SOURCE_BATCH/targets/$target.pt ]] || \
        fail "missing prepared target cache: $TARGET_SOURCE_BATCH/targets/$target.pt"
done

if (( ! dry_run )); then
    command -v sbatch >/dev/null 2>&1 || fail "sbatch is not available"
    command -v sinfo >/dev/null 2>&1 || fail "sinfo is not available"
    command -v squeue >/dev/null 2>&1 || fail "squeue is not available"

    gpu_inventory=$(sinfo -h -o '%G' --Node | sort -u)
    grep -q 'gpu:l40s:' <<< "$gpu_inventory" || {
        printf 'Discovered GPU inventory:\n%s\n' "$gpu_inventory" >&2
        fail 'gpu:l40s is not present in the discovered Killarney inventory'
    }

    queued_names=$(squeue -h -u "$USER" -o '%j')
    if grep -Eq '^kml_' <<< "$queued_names"; then
        printf 'Matching queued jobs:\n' >&2
        grep -E '^kml_' <<< "$queued_names" >&2
        fail 'long-run main-table work is already queued; refusing duplicate submission'
    fi
fi

if (( dry_run )); then
    MAINLONG_BATCH="$RESULTS_ROOT/<new-batch>"
    MAINLONG_CODE_ROOT="$MAINLONG_BATCH/code"
    TARGET_CACHE_ROOT="$MAINLONG_BATCH/targets"
    RUN_WORKER="$WORKER"
else
    mkdir -p "$RESULTS_ROOT"
    MAINLONG_BATCH=$(mktemp -d "$RESULTS_ROOT/batch_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")
    MAINLONG_CODE_ROOT="$MAINLONG_BATCH/code"
    TARGET_CACHE_ROOT="$MAINLONG_BATCH/targets"
    RUN_WORKER="$MAINLONG_CODE_ROOT/benchmarks/killarney_short_jobs/full_data_main_long_job.sh"

    mkdir -p \
        "$MAINLONG_BATCH/logs" \
        "$MAINLONG_BATCH/encoders" \
        "$MAINLONG_BATCH/evaluations" \
        "$MAINLONG_CODE_ROOT/benchmarks/killarney_short_jobs" \
        "$MAINLONG_CODE_ROOT/OT-SSL-DD" \
        "$TARGET_CACHE_ROOT"

    cp \
        benchmarks/benchmark_full_data_main_long.py \
        benchmarks/benchmark_full_refresh.py \
        benchmarks/benchmark_cifar10.py \
        benchmarks/benchmark_dataset.py \
        benchmarks/downstream_data.py \
        "$MAINLONG_CODE_ROOT/benchmarks/"
    cp OT-SSL-DD/utils.py OT-SSL-DD/networks.py "$MAINLONG_CODE_ROOT/OT-SSL-DD/"
    cp "$WORKER" "$ENVIRONMENT" "$AUDIT" \
        "$MAINLONG_CODE_ROOT/benchmarks/killarney_short_jobs/"

    for target in CIFAR10 CIFAR100; do
        ln -s "$TARGET_SOURCE_BATCH/targets/$target.pt" "$TARGET_CACHE_ROOT/$target.pt"
    done

    printf '%s\n' "$TARGET_SOURCE_BATCH" > "$MAINLONG_BATCH/target_source_batch.txt"
    printf '%s\n' 'full_data_main_simclr1000_barlow1500_v1' > "$MAINLONG_BATCH/protocol.txt"
    printf 'source\tmodel\tssl_method\tssl_epochs\tencoder_seeds\tlabels\tprobe_epochs\n' \
        > "$MAINLONG_BATCH/configurations.tsv"
    for source in CIFAR10 CIFAR100; do
        for model in ConvNet VGG11 ResNet18; do
            printf '%s\t%s\tsimclr\t1000\t0-9\t1,5\t500,300\n' "$source" "$model"
            printf '%s\t%s\tbarlowtwins\t1500\t0-9\t1,5\t500,300\n' "$source" "$model"
        done
    done >> "$MAINLONG_BATCH/configurations.tsv"
    printf 'submitted_utc\tjob_id\taccount\tkind\tconfiguration\tarray\tdependency\n' \
        > "$MAINLONG_BATCH/submitted.tsv"

    tar -czf "$MAINLONG_BATCH/source.tar.gz" \
        benchmarks/benchmark_full_data_main_long.py \
        benchmarks/benchmark_full_refresh.py \
        benchmarks/benchmark_cifar10.py \
        benchmarks/benchmark_dataset.py \
        benchmarks/downstream_data.py \
        benchmarks/killarney_short_jobs/full_data_main_long_job.sh \
        benchmarks/killarney_short_jobs/full_data_main_long_environment.sh \
        benchmarks/killarney_short_jobs/submit_full_data_main_long_to_wang.sh \
        benchmarks/killarney_short_jobs/check_full_data_main_long.sh \
        OT-SSL-DD/utils.py \
        OT-SSL-DD/networks.py
fi

export PROJECT_ROOT MAINLONG_BATCH MAINLONG_CODE_ROOT TARGET_CACHE_ROOT VENV_ACTIVATE

LAST_JOB_ID=
dry_id=998000
submitted_jobs=0
submitted_tasks=0

submit_pretrain_array() {
    local name=$1
    local limit=$2
    local memory=$3
    shift 3
    local args=(sbatch --parsable
        --account="$ACCOUNT"
        --job-name="$name"
        --array=0-9
        --nodes=1
        --ntasks=1
        --cpus-per-task=1
        --gres=gpu:l40s:1
        --time="$limit"
        --mem="$memory"
        --chdir="$PROJECT_ROOT"
        --output="$MAINLONG_BATCH/logs/%x-%A_%a.out")
    local submitted configuration
    configuration="$*"

    if (( dry_run )); then
        printf 'DRY RUN: '
        printf '%q ' "${args[@]}" "$RUN_WORKER" "$@"
        printf '\n'
        dry_id=$((dry_id + 1))
        LAST_JOB_ID=$dry_id
    else
        submitted=$("${args[@]}" "$RUN_WORKER" "$@")
        LAST_JOB_ID=${submitted%%;*}
        printf '%s\t%s\t%s\tpretrain\t%s\t0-9\t\n' \
            "$(date -u +%FT%TZ)" "$LAST_JOB_ID" "$ACCOUNT" "$configuration" \
            >> "$MAINLONG_BATCH/submitted.tsv"
        printf 'Submitted %-24s as array %s on %s\n' "$name" "$LAST_JOB_ID" "$ACCOUNT"
    fi
    submitted_jobs=$((submitted_jobs + 1))
    submitted_tasks=$((submitted_tasks + 10))
}

submit_evaluation() {
    local name=$1
    local dependency=$2
    shift 2
    local args=(sbatch --parsable
        --account="$ACCOUNT"
        --job-name="$name"
        --nodes=1
        --ntasks=1
        --cpus-per-task=1
        --gres=gpu:l40s:1
        --time=01:30:00
        --mem=8G
        --chdir="$PROJECT_ROOT"
        --output="$MAINLONG_BATCH/logs/%x-%j.out"
        --dependency="afterok:$dependency"
        --kill-on-invalid-dep=yes)
    local submitted configuration
    configuration="$*"

    if (( dry_run )); then
        printf 'DRY RUN: '
        printf '%q ' "${args[@]}" "$RUN_WORKER" "$@"
        printf '\n'
        dry_id=$((dry_id + 1))
        LAST_JOB_ID=$dry_id
    else
        submitted=$("${args[@]}" "$RUN_WORKER" "$@")
        LAST_JOB_ID=${submitted%%;*}
        printf '%s\t%s\t%s\tevaluate\t%s\t\tafterok:%s\n' \
            "$(date -u +%FT%TZ)" "$LAST_JOB_ID" "$ACCOUNT" "$configuration" "$dependency" \
            >> "$MAINLONG_BATCH/submitted.tsv"
        printf 'Submitted %-24s as %s after %s\n' "$name" "$LAST_JOB_ID" "$dependency"
    fi
    submitted_jobs=$((submitted_jobs + 1))
    submitted_tasks=$((submitted_tasks + 1))
}

submit_configuration() {
    local tag=$1
    local source=$2
    local model=$3
    local ssl=$4
    local limit=$5
    local pretrain_id

    submit_pretrain_array "kml_${tag}_pre" "$limit" 8G \
        pretrain "$source" "$model" "$ssl"
    pretrain_id=$LAST_JOB_ID
    submit_evaluation "kml_${tag}_eval" "$pretrain_id" \
        evaluate "$source" "$model" "$ssl"
}

for source in CIFAR10 CIFAR100; do
    if [[ $source == CIFAR10 ]]; then source_tag=c10; else source_tag=c100; fi

    submit_configuration "${source_tag}_cn_s" "$source" ConvNet simclr 02:00:00
    submit_configuration "${source_tag}_cn_b" "$source" ConvNet barlowtwins 03:00:00
    submit_configuration "${source_tag}_vg_s" "$source" VGG11 simclr 02:30:00
    submit_configuration "${source_tag}_vg_b" "$source" VGG11 barlowtwins 03:30:00
    submit_configuration "${source_tag}_r18_s" "$source" ResNet18 simclr 05:30:00
    submit_configuration "${source_tag}_r18_b" "$source" ResNet18 barlowtwins 07:30:00
done

printf '\nSubmission plan complete.\n'
printf 'Account: %s\n' "$ACCOUNT"
printf 'Submitted job records: %d (12 arrays + 12 dependent evaluators)\n' "$submitted_jobs"
printf 'GPU tasks: %d (120 pretraining + 12 evaluation)\n' "$submitted_tasks"
printf '%s\n' 'SimCLR: 1000 SSL epochs; Barlow Twins: 1500 SSL epochs'
printf '%s\n' 'Evaluation: source dataset only, labels 1%/5%, probes 500/300 epochs, 10 runs'
printf 'Results: %s\n' "$MAINLONG_BATCH"
printf '%s\n' 'CPU Slurm jobs: 0'
if (( dry_run )); then
    printf '%s\n' 'Dry run only: nothing was created or submitted.'
else
    printf 'After completion, audit with:\n  bash %q %q\n' "$AUDIT" "$MAINLONG_BATCH"
fi
