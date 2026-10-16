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

ACCOUNT="${ACCOUNT:-aip-yiweilu}"
VENV_ACTIVATE="${VENV_ACTIVATE:-$HOME/ENV/bin/activate}"
[[ $ACCOUNT == aip-yiweilu ]] || {
    echo "This workflow is pinned to the Lu account: aip-yiweilu" >&2
    exit 1
}
export ACCOUNT VENV_ACTIVATE

command -v jq >/dev/null || { echo "jq is required" >&2; exit 1; }
command -v sbatch >/dev/null || { echo "sbatch is required" >&2; exit 1; }
[[ -f $VENV_ACTIVATE ]] || { echo "Missing Python environment: $VENV_ACTIVATE" >&2; exit 1; }

source_batch="$PROJECT_ROOT/benchmarks/cross_downstream/results/batch_20261009T171449Z_o91BKP"
epoch_file="$PROJECT_ROOT/OT-SSL-DD/training_epochs.json"
[[ -f $epoch_file ]] || { echo "Missing epoch configuration: $epoch_file" >&2; exit 1; }

targets=(CIFAR10 CIFAR100 Aircraft CUB2011 Dogs Flowers TinyImageNet)
models=(ConvNet VGG11 ResNet18)
for target in "${targets[@]}"; do
    cache="$source_batch/targets/$target.pt"
    [[ -f $cache ]] || { echo "Missing target cache: $cache" >&2; exit 1; }
    for labels in 1 5; do
        actual=$(jq -er --arg target "$target" --arg labels "$labels" \
            '.evaluation_by_dataset[$target].no_pretraining_linear_probe_epochs_by_label_percentage[$labels]' \
            "$epoch_file")
        expected=1000
        [[ $labels == 5 ]] && expected=500
        [[ $actual == "$expected" ]] || {
            echo "$target $labels% no-pretraining probe budget is $actual; expected $expected" >&2
            exit 1
        }
    done
done

for source_file in \
    benchmarks/benchmark_downstream.py \
    benchmarks/benchmark_cifar10.py \
    benchmarks/downstream_data.py \
    benchmarks/vulcan_no_pretrain/environment.sh \
    benchmarks/vulcan_no_pretrain/job.sh \
    benchmarks/vulcan_no_pretrain/check.sh \
    OT-SSL-DD/utils.py \
    OT-SSL-DD/networks.py; do
    [[ -f $source_file ]] || { echo "Missing source file: $source_file" >&2; exit 1; }
done

if (( ! dry_run )); then
    active=$(squeue -h -u "$USER" -n vnp_all6 -o '%A' | sort -u)
    [[ -z $active ]] || {
        echo "Refusing duplicate submission; active vnp_all6 job(s): $active" >&2
        exit 1
    }
fi

results_root="$PROJECT_ROOT/benchmarks/vulcan_no_pretrain/results"
mkdir -p "$results_root"
NO_PRETRAIN_BATCH="$(mktemp -d "$results_root/batch_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")"
NO_PRETRAIN_CODE_ROOT="$NO_PRETRAIN_BATCH/code"
TARGET_CACHE_ROOT="$NO_PRETRAIN_BATCH/targets"
export NO_PRETRAIN_BATCH NO_PRETRAIN_CODE_ROOT TARGET_CACHE_ROOT

mkdir -p \
    "$NO_PRETRAIN_BATCH/logs" \
    "$NO_PRETRAIN_BATCH/evaluations" \
    "$NO_PRETRAIN_CODE_ROOT/benchmarks/vulcan_no_pretrain" \
    "$NO_PRETRAIN_CODE_ROOT/OT-SSL-DD" \
    "$TARGET_CACHE_ROOT"

for target in "${targets[@]}"; do
    ln -s "$source_batch/targets/$target.pt" "$TARGET_CACHE_ROOT/$target.pt"
done

cp benchmarks/benchmark_downstream.py benchmarks/benchmark_cifar10.py \
    benchmarks/downstream_data.py "$NO_PRETRAIN_CODE_ROOT/benchmarks/"
cp benchmarks/vulcan_no_pretrain/environment.sh \
    benchmarks/vulcan_no_pretrain/job.sh \
    benchmarks/vulcan_no_pretrain/check.sh \
    "$NO_PRETRAIN_CODE_ROOT/benchmarks/vulcan_no_pretrain/"
cp OT-SSL-DD/utils.py OT-SSL-DD/networks.py "$NO_PRETRAIN_CODE_ROOT/OT-SSL-DD/"
cp "$epoch_file" "$NO_PRETRAIN_BATCH/training_epochs.json"
printf '%s\n' 'no_pretrain_probe1000_500_all_six_tables_v1' > "$NO_PRETRAIN_BATCH/protocol.txt"

printf 'task_id\tconfig\ttarget\tmodel\n' > "$NO_PRETRAIN_BATCH/configs.tsv"
task_id=0
for target in "${targets[@]}"; do
    for model in "${models[@]}"; do
        printf '%s\t%s\t%s\t%s\n' \
            "$task_id" "${target}_${model}_no_pretrain" "$target" "$model" \
            >> "$NO_PRETRAIN_BATCH/configs.tsv"
        task_id=$((task_id + 1))
    done
done

# This explicitly maps every visible No Pre-Training cell in the six TeX files
# to one of the 42 unique target/model/label evaluations above.
printf 'table\ttarget\tmodel\tlabel_percentage\tresult_id\n' > "$NO_PRETRAIN_BATCH/table_cells.tsv"
add_cells() {
    local table=$1
    shift
    local target model labels
    while (( $# >= 2 )); do
        target=$1
        model=$2
        shift 2
        for labels in 1 5; do
            printf '%s\t%s\t%s\t%s\t%s\n' \
                "$table" "$target" "$model" "$labels" \
                "${target}_${model}_no_pretrain_lbl${labels}" \
                >> "$NO_PRETRAIN_BATCH/table_cells.tsv"
        done
    done
}

for model in "${models[@]}"; do
    add_cells cifar10_main.tex CIFAR10 "$model"
    add_cells cifar100_main.tex CIFAR100 "$model"
    add_cells tinyimagenet_main.tex TinyImageNet "$model"
done
for model in "${models[@]}"; do
    for target in CIFAR100 Aircraft CUB2011 Dogs Flowers TinyImageNet; do
        add_cells cifar10_downstream.tex "$target" "$model"
    done
done
for target in TinyImageNet CIFAR10 Aircraft CUB2011 Dogs Flowers; do
    add_cells cifar100_downstream.tex "$target" ResNet18
done
for target in CIFAR10 CIFAR100 Aircraft CUB2011 Dogs Flowers; do
    add_cells tinyimagenet_downstream.tex "$target" ResNet18
done

config_count=$(( $(wc -l < "$NO_PRETRAIN_BATCH/configs.tsv") - 1 ))
cell_count=$(( $(wc -l < "$NO_PRETRAIN_BATCH/table_cells.tsv") - 1 ))
table_count=$(tail -n +2 "$NO_PRETRAIN_BATCH/table_cells.tsv" | cut -f1 | sort -u | wc -l)
unique_results=$(tail -n +2 "$NO_PRETRAIN_BATCH/table_cells.tsv" | cut -f5 | sort -u | wc -l)
[[ $config_count == 21 ]] || { echo "Internal error: $config_count configs, expected 21" >&2; exit 1; }
[[ $cell_count == 78 ]] || { echo "Internal error: $cell_count cells, expected 78" >&2; exit 1; }
[[ $table_count == 6 ]] || { echo "Internal error: $table_count tables, expected 6" >&2; exit 1; }
[[ $unique_results == 42 ]] || { echo "Internal error: $unique_results results, expected 42" >&2; exit 1; }

printf 'submitted_utc\tjob_id\taccount\tarray_tasks\tunique_results\ttable_cells\n' \
    > "$NO_PRETRAIN_BATCH/submitted.tsv"

sbatch_args=(
    sbatch --parsable
    --account="$ACCOUNT"
    --job-name=vnp_all6
    --time=0-01:30:00
    --chdir="$PROJECT_ROOT"
    --array=0-20%12
    --output="$NO_PRETRAIN_BATCH/logs/%x-%A_%a.out"
    --export="ALL,PROJECT_ROOT=$PROJECT_ROOT,NO_PRETRAIN_BATCH=$NO_PRETRAIN_BATCH,NO_PRETRAIN_CODE_ROOT=$NO_PRETRAIN_CODE_ROOT,TARGET_CACHE_ROOT=$TARGET_CACHE_ROOT,VENV_ACTIVATE=$VENV_ACTIVATE"
    "$NO_PRETRAIN_CODE_ROOT/benchmarks/vulcan_no_pretrain/job.sh"
)

if (( dry_run )); then
    printf 'DRY-RUN: '
    printf '%q ' "${sbatch_args[@]}"
    printf '\n'
    job_id=DRY-RUN
else
    job_id=$("${sbatch_args[@]}")
    job_id=${job_id%%;*}
fi

printf '%s\t%s\t%s\t21\t42\t78\n' \
    "$(date -u +%FT%TZ)" "$job_id" "$ACCOUNT" >> "$NO_PRETRAIN_BATCH/submitted.tsv"

mode=SUBMITTED
(( dry_run )) && mode=DRY-RUN
printf '%s batch: %s\nJob: %s\nAccount: %s\nArray tasks: 21 (each runs both label budgets)\nUnique evaluations: 42\nCovered TeX cells: 78 across all 6 tables\n' \
    "$mode" "$NO_PRETRAIN_BATCH" "$job_id" "$ACCOUNT"
