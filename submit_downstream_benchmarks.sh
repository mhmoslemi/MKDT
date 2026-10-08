#!/bin/bash
# Each target configuration runs seeds 0..14; CIFAR-10 encoders are shared.
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
export PROJECT_ROOT
cd "$PROJECT_ROOT"
dry_run=0
TARGET_DATASETS=CIFAR100,Aircraft,CUB2011,Dogs,Flowers
while (( $# )); do
    case $1 in
        --dry-run) dry_run=1; shift ;;
        --datasets) [[ $# -ge 2 ]] || { echo '--datasets needs a comma-separated list.' >&2; exit 2; }; TARGET_DATASETS=$2; shift 2 ;;
        *) echo "Usage: bash $0 [--dry-run] [--datasets CIFAR100,Aircraft,CUB2011,Dogs,Flowers]" >&2; exit 2 ;;
    esac
done
IFS=, read -r -a targets <<< "$TARGET_DATASETS"
[[ ${#targets[@]} -gt 0 && $TARGET_DATASETS != *, ]] || { echo 'Empty target list.' >&2; exit 2; }
declare -A selected=()
for target in "${targets[@]}"; do
    case $target in CIFAR100|Aircraft|CUB2011|Dogs|Flowers) ;; *) echo "Unknown target: $target" >&2; exit 2 ;; esac
    [[ ! -v selected[$target] ]] || { echo "Duplicate target: $target" >&2; exit 2; }
    selected[$target]=1
done
SOURCE_CIFAR10_BATCH="${SOURCE_CIFAR10_BATCH:-$PROJECT_ROOT/benchmarks/cifar10/results/batch_20261008T052732Z_lb36yJ}"
export TARGET_DATASETS SOURCE_CIFAR10_BATCH
[[ -f benchmarks/downstream/configs.tsv && -f benchmarks/downstream/preflight.sh ]] || { echo 'Downstream job files have not been generated.' >&2; exit 1; }
if (( dry_run )); then
    DOWNSTREAM_RUN_ROOT="$PROJECT_ROOT/benchmarks/downstream/results/<new-batch>"
    DOWNSTREAM_CODE_ROOT="$DOWNSTREAM_RUN_ROOT/code"
else
    command -v sbatch >/dev/null
    [[ -f ${VENV_ACTIVATE:-$HOME/ENV/bin/activate} ]] || { echo 'Missing Python environment.' >&2; exit 1; }
    data_root="${DATA_ROOT:-$SCRATCH/data}"
    [[ -f $data_root/cifar-10-batches-py/data_batch_1 ]] || { echo "Missing CIFAR-10 under $data_root" >&2; exit 1; }
    # Cheap staging checks before submitting any jobs. The Slurm preflight checks metadata.
    for target in "${targets[@]}"; do
        case $target in
            CIFAR100) markers=("$data_root/cifar-100-python/train") ;;
            Aircraft) markers=("$data_root/fgvc-aircraft-2013b/data/variants.txt" "$data_root/aircraft/fgvc-aircraft-2013b/data/variants.txt") ;;
            CUB2011) markers=("$data_root/CUB_200_2011/images.txt" "$data_root/cub2011/CUB_200_2011/images.txt") ;;
            Dogs) markers=("$data_root/dogs/train_list.mat" "$data_root/stanford_dogs/train_list.mat" "$data_root/train_list.mat") ;;
            Flowers) markers=("$data_root/flowers-102/setid.mat" "$data_root/flowers/flowers-102/setid.mat") ;;
        esac
        present=0
        for marker in "${markers[@]}"; do [[ ! -f $marker ]] || present=1; done
        (( present )) || { echo "$target is not staged. Expected ${markers[0]}. See benchmarks/downstream/data_layout.txt." >&2; exit 1; }
    done
    mkdir -p benchmarks/downstream/results
    DOWNSTREAM_RUN_ROOT=$(mktemp -d "$PROJECT_ROOT/benchmarks/downstream/results/batch_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")
    DOWNSTREAM_CODE_ROOT="$DOWNSTREAM_RUN_ROOT/code"
    mkdir -p "$DOWNSTREAM_RUN_ROOT/logs" "$DOWNSTREAM_CODE_ROOT/OT-SSL-DD" "$DOWNSTREAM_CODE_ROOT/benchmarks/downstream"
    cp OT-SSL-DD/benchmark_downstream.py OT-SSL-DD/downstream_data.py OT-SSL-DD/benchmark_cifar10.py OT-SSL-DD/utils.py OT-SSL-DD/networks.py "$DOWNSTREAM_CODE_ROOT/OT-SSL-DD/"
    cp benchmarks/downstream/report.py benchmarks/downstream/configs.tsv benchmarks/downstream/pretraining.tsv benchmarks/downstream/protocol.txt benchmarks/downstream/data_layout.txt "$DOWNSTREAM_CODE_ROOT/benchmarks/downstream/"
    cp res.tex "$DOWNSTREAM_RUN_ROOT/res_template.tex"
    cp benchmarks/downstream/configs.tsv benchmarks/downstream/pretraining.tsv "$DOWNSTREAM_RUN_ROOT/"
    printf '%s\n' "$TARGET_DATASETS" > "$DOWNSTREAM_RUN_ROOT/targets.txt"
    if [[ -f $SOURCE_CIFAR10_BATCH/tables.csv ]]; then cp "$SOURCE_CIFAR10_BATCH/tables.csv" "$DOWNSTREAM_RUN_ROOT/source_cifar10.csv"; fi
fi
export DOWNSTREAM_RUN_ROOT DOWNSTREAM_CODE_ROOT
dry_id=900000
submit() {
    local script=$1
    shift
    local args=(sbatch --parsable --chdir="$PROJECT_ROOT" --output="$DOWNSTREAM_RUN_ROOT/logs/%x-%j.out" "$@" "$script")
    if (( dry_run )); then
        printf '%q ' "${args[@]}" >&2
        printf '\n' >&2
        SUBMITTED_ID=$((++dry_id))
    else
        SUBMITTED_ID=$("${args[@]}")
        SUBMITTED_ID=${SUBMITTED_ID%%;*}
        printf '%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "$SUBMITTED_ID" "$script" >> "$DOWNSTREAM_RUN_ROOT/submitted.tsv"
        printf 'Submitted %s: %s\n' "$(basename "$script")" "$SUBMITTED_ID" >&2
    fi
}
submit "$PROJECT_ROOT/benchmarks/downstream/preflight.sh"
preflight_id=$SUBMITTED_ID
submit "$PROJECT_ROOT/benchmarks/downstream/prepare_selections.sh" --dependency="afterok:$preflight_id" --kill-on-invalid-dep=yes
selection_id=$SUBMITTED_ID
declare -A packing_ids=() encoder_ids=()
for target in "${targets[@]}"; do
    submit "$PROJECT_ROOT/benchmarks/downstream/jobs/prepare/$target.sh" --dependency="afterok:$preflight_id" --kill-on-invalid-dep=yes
    packing_ids[$target]=$SUBMITTED_ID
done
for script in "$PROJECT_ROOT"/benchmarks/downstream/jobs/pretrain/*.sh; do
    dependency=$preflight_id
    [[ $script != *_kmeans_* ]] || dependency=$selection_id
    submit "$script" --dependency="afterok:$dependency" --kill-on-invalid-dep=yes
    encoder_ids[$(basename "$script" .sh)]=$SUBMITTED_ID
done
evaluation_ids=()
while IFS=$'\t' read -r config target model ssl method size labels encoder epochs runs policy updates estimate limit script; do
    [[ $config != config_id ]] || continue
    [[ -v selected[$target] ]] || continue
    script=${script%$'\r'}
    dependency=${packing_ids[$target]}
    [[ $encoder == none ]] || dependency+=":${encoder_ids[$encoder]}"
    submit "$PROJECT_ROOT/$script" --dependency="afterok:$dependency" --kill-on-invalid-dep=yes
    evaluation_ids+=("$SUBMITTED_ID")
done < benchmarks/downstream/configs.tsv
dependency=$(IFS=:; echo "${evaluation_ids[*]}")
submit "$PROJECT_ROOT/benchmarks/downstream/report.sh" --dependency="afterany:$dependency"
printf '%s downstream configurations x 15 runs, 42 shared source jobs, and preparation/report jobs.\nResults: %s\n' "${#evaluation_ids[@]}" "$DOWNSTREAM_RUN_ROOT"
