#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
export PROJECT_ROOT
cd "$PROJECT_ROOT"

SYNTHETIC_INPUT_ROOT="${SYNTHETIC_INPUT_ROOT:-$PROJECT_ROOT/synthetic_new}"
TARGET_CACHE_ROOT="${TARGET_CACHE_ROOT:-$PROJECT_ROOT/benchmarks/downstream/results/batch_20261009T043709Z_zC8EPD/targets}"
CIFAR10_CACHE="${CIFAR10_CACHE:-$PROJECT_ROOT/benchmarks/cross_downstream/results/batch_20261009T171449Z_o91BKP/targets/CIFAR10.pt}"
VENV_ACTIVATE="${VENV_ACTIVATE:-$HOME/ENV/bin/activate}"
export SYNTHETIC_INPUT_ROOT TARGET_CACHE_ROOT CIFAR10_CACHE VENV_ACTIVATE

for source in CIFAR10 CIFAR100; do
    for size in 1 2 5; do
        file="$SYNTHETIC_INPUT_ROOT/res_IIC_${source}_ConvNet_${size}percent.pt"
        [[ -f $file ]] || { echo "Missing synthetic input: $file" >&2; exit 1; }
    done
done
for target in CIFAR100 Aircraft CUB2011 Dogs Flowers TinyImageNet; do
    [[ -f $TARGET_CACHE_ROOT/$target.pt ]] || {
        echo "Missing target cache: $TARGET_CACHE_ROOT/$target.pt" >&2
        exit 1
    }
done
[[ -f $CIFAR10_CACHE ]] || { echo "Missing target cache: $CIFAR10_CACHE" >&2; exit 1; }
[[ -f $VENV_ACTIVATE ]] || { echo "Missing Python environment: $VENV_ACTIVATE" >&2; exit 1; }

mkdir -p benchmarks/synthetic_new/results
SYNTH_BATCH_ROOT=$(mktemp -d "$PROJECT_ROOT/benchmarks/synthetic_new/results/batch_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")
SYNTH_CODE_ROOT="$SYNTH_BATCH_ROOT/code"
export SYNTH_BATCH_ROOT SYNTH_CODE_ROOT
mkdir -p "$SYNTH_BATCH_ROOT/logs" "$SYNTH_BATCH_ROOT/table_templates" \
    "$SYNTH_CODE_ROOT/benchmarks" "$SYNTH_CODE_ROOT/OT-SSL-DD"
cp benchmarks/benchmark_synthetic_new.py benchmarks/benchmark_cifar10.py \
    benchmarks/benchmark_dataset.py benchmarks/downstream_data.py \
    "$SYNTH_CODE_ROOT/benchmarks/"
cp OT-SSL-DD/utils.py OT-SSL-DD/networks.py OT-SSL-DD/training_epochs.json \
    "$SYNTH_CODE_ROOT/OT-SSL-DD/"
cp CVPR_tables/cifar10_main.tex CVPR_tables/cifar100_main.tex \
    CVPR_tables/cifar10_downstream.tex CVPR_tables/cifar100_downstream.tex \
    "$SYNTH_BATCH_ROOT/table_templates/"
cp benchmarks/synthetic_new/make_jobs.py benchmarks/synthetic_new/environment.sh \
    benchmarks/synthetic_new/generate.sh "$SYNTH_BATCH_ROOT/"

generation_id=$(sbatch --wait --parsable --account=aip-boyuwang \
    --chdir="$PROJECT_ROOT" --output="$SYNTH_BATCH_ROOT/logs/%x-%j.out" \
    benchmarks/synthetic_new/generate.sh)
printf '%s\t%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "${generation_id%%;*}" \
    aip-boyuwang benchmarks/synthetic_new/generate.sh > "$SYNTH_BATCH_ROOT/setup.tsv"

[[ -f $SYNTH_BATCH_ROOT/jobs.tsv && -f $SYNTH_BATCH_ROOT/configs.tsv ]] || {
    echo "Generation job did not create the manifests; inspect $SYNTH_BATCH_ROOT/logs" >&2
    exit 1
}

declare -A job_ids
evaluation_ids=()
while IFS=$'\t' read -r stage key dependency account walltime script; do
    script=${script%$'\r'}
    [[ $stage == stage ]] && continue
    args=(sbatch --parsable --account="$account" --chdir="$PROJECT_ROOT"
          --output="$SYNTH_BATCH_ROOT/logs/%x-%j.out")
    if [[ $dependency != - ]]; then
        dependency_id=${job_ids[$dependency]:-}
        [[ -n $dependency_id ]] || { echo "Unknown dependency: $dependency" >&2; exit 1; }
        args+=(--dependency="afterok:$dependency_id" --kill-on-invalid-dep=yes)
    fi
    id=$("${args[@]}" "$SYNTH_BATCH_ROOT/$script")
    id=${id%%;*}
    job_ids["$key"]=$id
    if [[ $stage != pretrain ]]; then
        evaluation_ids+=("$id")
    fi
    printf '%s\t%s\t%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "$id" "$account" "$stage" "$script" \
        >> "$SYNTH_BATCH_ROOT/submitted.tsv"
done < "$SYNTH_BATCH_ROOT/jobs.tsv"

dependency=$(IFS=:; echo "${evaluation_ids[*]}")
report_id=$(sbatch --parsable --account=aip-boyuwang --chdir="$PROJECT_ROOT" \
    --output="$SYNTH_BATCH_ROOT/logs/%x-%j.out" \
    --dependency="afterany:$dependency" "$SYNTH_BATCH_ROOT/jobs/report.sh")
report_id=${report_id%%;*}
printf '%s\t%s\t%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "$report_id" aip-boyuwang report jobs/report.sh \
    >> "$SYNTH_BATCH_ROOT/submitted.tsv"

wang=$(awk -F'\t' '$3=="aip-boyuwang"{n++} END{print n+0}' "$SYNTH_BATCH_ROOT/submitted.tsv")
lu=$(awk -F'\t' '$3=="aip-yiweilu"{n++} END{print n+0}' "$SYNTH_BATCH_ROOT/submitted.tsv")
printf 'Submitted 84 GPU jobs plus one report job: Wang=%s, Lu=%s.\n' "$wang" "$lu"
printf 'Batch: %s\nReport job: %s\n' "$SYNTH_BATCH_ROOT" "$report_id"
