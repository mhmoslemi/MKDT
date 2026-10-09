#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
export PROJECT_ROOT
cd "$PROJECT_ROOT"

if (( $# < 1 || $# > 2 )); then
    echo "Usage: bash $0 <CIFAR100|TinyImageNet> [--dry-run]" >&2
    exit 2
fi
DATASET=$1
dry_run=0
if (( $# == 2 )); then
    [[ $2 == --dry-run ]] || { echo "Unknown option: $2" >&2; exit 2; }
    dry_run=1
fi

case "$DATASET" in
    CIFAR100)
        slug=cifar100
        cache_name=CIFAR100.pt
        ;;
    TinyImageNet)
        slug=tinyimagenet
        cache_name=TinyImageNet.pt
        ;;
    *)
        echo "Dataset must be CIFAR100 or TinyImageNet." >&2
        exit 2
        ;;
esac

export DATASET
export TARGET_CACHE="${TARGET_CACHE:-$PROJECT_ROOT/benchmarks/downstream/results/batch_20261009T043709Z_zC8EPD/targets/$cache_name}"
JOB_ARCHIVE="$PROJECT_ROOT/benchmarks/$slug/jobs.tar"
CONFIGS="$PROJECT_ROOT/benchmarks/$slug/configs.tsv"
[[ -f $TARGET_CACHE ]] || { echo "Missing validated target cache: $TARGET_CACHE" >&2; exit 1; }
[[ -f $JOB_ARCHIVE && -f $CONFIGS && -f benchmarks/$slug/prepare.sh && -f benchmarks/$slug/report.sh ]] || {
    echo "Missing generated $DATASET jobs; run benchmarks/main/generate.sh through Slurm first." >&2
    exit 1
}
mapfile -t jobs < <(tar -tf "$JOB_ARCHIVE")

declare -A accounts
while IFS=$'\t' read -r script account; do
    accounts["$script"]=$account
done < <(awk -F'\t' 'NR==1 {for(i=1;i<=NF;i++){gsub(/\r/, "", $i); if($i=="job_script")j=i; if($i=="account")a=i} next} {gsub(/\r/, "", $j); gsub(/\r/, "", $a); print $j "\t" $a}' "$CONFIGS")

if (( dry_run )); then
    MAIN_RUN_ROOT="$PROJECT_ROOT/benchmarks/$slug/results/<new-batch>"
    MAIN_CODE_ROOT="$PROJECT_ROOT"
else
    command -v sbatch >/dev/null
    [[ -f ${VENV_ACTIVATE:-$HOME/ENV/bin/activate} ]] || { echo 'Python environment is missing.' >&2; exit 1; }
    mkdir -p "benchmarks/$slug/results"
    MAIN_RUN_ROOT=$(mktemp -d "$PROJECT_ROOT/benchmarks/$slug/results/batch_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")
    mkdir "$MAIN_RUN_ROOT/logs"
    cp "$CONFIGS" "$MAIN_RUN_ROOT/configs.tsv"
    MAIN_CODE_ROOT="$MAIN_RUN_ROOT/code"
    mkdir -p "$MAIN_CODE_ROOT/OT-SSL-DD" "$MAIN_CODE_ROOT/benchmarks/main" "$MAIN_CODE_ROOT/benchmarks/$slug"
    cp OT-SSL-DD/utils.py OT-SSL-DD/networks.py OT-SSL-DD/training_epochs.json "$MAIN_CODE_ROOT/OT-SSL-DD/"
    cp benchmarks/benchmark_cifar10.py benchmarks/benchmark_dataset.py benchmarks/downstream_data.py "$MAIN_CODE_ROOT/benchmarks/"
    cp benchmarks/main/report.py "$MAIN_CODE_ROOT/benchmarks/main/"
    cp "$CONFIGS" "$MAIN_CODE_ROOT/benchmarks/$slug/configs.tsv"
    tar -czf "$MAIN_RUN_ROOT/source.tar.gz" \
        benchmarks/benchmark_cifar10.py benchmarks/benchmark_dataset.py benchmarks/downstream_data.py \
        OT-SSL-DD/utils.py OT-SSL-DD/networks.py OT-SSL-DD/training_epochs.json \
        "benchmarks/$slug/jobs.tar" benchmarks/main/environment.sh benchmarks/main/report.py \
        "benchmarks/$slug/protocol.txt"
fi
export MAIN_RUN_ROOT MAIN_CODE_ROOT
export SELECTION_DIR="$MAIN_RUN_ROOT/selections"

submit() {
    local script=$1
    local account=$2
    shift 2
    local args=(sbatch --parsable --account="$account" --chdir="$PROJECT_ROOT" --output="$MAIN_RUN_ROOT/logs/%x-%j.out" "$@")
    if (( dry_run )); then
        printf '%q ' "${args[@]}" >&2
        printf '%s\n' "$script" >&2
        printf 'DRY_JOB_ID\n'
    else
        local id
        if [[ $script == benchmarks/$slug/jobs/* ]]; then
            id=$(tar -xOf "$JOB_ARCHIVE" "$script" | "${args[@]}")
        else
            id=$("${args[@]}" "$script")
        fi
        printf '%s\t%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "$id" "$account" "$script" >> "$MAIN_RUN_ROOT/submitted.tsv"
        printf 'Submitted %s on %s: %s\n' "$(basename "$script")" "$account" "$id" >&2
        printf '%s\n' "${id%%;*}"
    fi
}

prepare_id=$(submit "$PROJECT_ROOT/benchmarks/$slug/prepare.sh" aip-boyuwang)
ids=()
for script in "${jobs[@]}"; do
    dependencies=(--dependency="afterok:$prepare_id" --kill-on-invalid-dep=yes)
    ids+=("$(submit "$script" "${accounts[$script]}" "${dependencies[@]}")")
done
dependency=$(IFS=:; echo "${ids[*]}")
report_id=$(submit "$PROJECT_ROOT/benchmarks/$slug/report.sh" aip-boyuwang --dependency="afterany:$dependency")
wang=$(awk -F'\t' 'NR>1 && $15=="aip-boyuwang"{n++} END{print n+0}' "$CONFIGS")
lu=$(awk -F'\t' 'NR>1 && $15=="aip-yiweilu"{n++} END{print n+0}' "$CONFIGS")
printf '%s: %s configurations (%s Wang, %s Lu), plus preparation and report jobs.\nResults: %s\nReport job: %s\n' \
    "$DATASET" "${#jobs[@]}" "$wang" "$lu" "$MAIN_RUN_ROOT" "$report_id"
