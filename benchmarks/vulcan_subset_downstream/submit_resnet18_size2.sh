#!/bin/bash
set -euo pipefail

# Evaluate only the eight 2% ResNet18 Random/K-means checkpoint sets needed by
# the active CIFAR-10 and CIFAR-100 downstream tables. This never retrains SSL.

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
cd "$PROJECT_ROOT"

ACCOUNT=aip-boyuwang
UPSTREAM_BATCH="$PROJECT_ROOT/benchmarks/vulcan_subset_refresh/results/batch_20261016T163046Z_utW9z3"
VENV_ACTIVATE="${VENV_ACTIVATE:-$HOME/ENV/bin/activate}"
cache_root="$PROJECT_ROOT/benchmarks/cross_downstream/results/batch_20261009T171449Z_o91BKP/targets"

fail() {
    echo "FATAL: $*" >&2
    exit 1
}

[[ -s $UPSTREAM_BATCH/configs.tsv ]] || fail "missing upstream manifest: $UPSTREAM_BATCH/configs.tsv"
[[ -s $VENV_ACTIVATE ]] || fail "missing Python environment: $VENV_ACTIVATE"
command -v jq >/dev/null || fail "jq is required"

for target in CIFAR10 CIFAR100 Aircraft CUB2011 Dogs Flowers TinyImageNet; do
    [[ -s $cache_root/$target.pt ]] || fail "missing target cache: $cache_root/$target.pt"
done

queued=$(squeue -h -u "$USER" -n vsd_r18_pt2 -o '%i' || true)
[[ -z $queued ]] || fail "vsd_r18_pt2 is already queued as $queued"

gpu_inventory=$(sinfo -h -o '%G' --Node | sort -u)
grep -q 'gpu:l40s:' <<< "$gpu_inventory" || fail "gpu:l40s is not available"

results_root="$PROJECT_ROOT/benchmarks/vulcan_subset_downstream/results"
mkdir -p "$results_root"
DOWNSTREAM_BATCH="$(mktemp -d "$results_root/batch_$(date -u +%Y%m%dT%H%M%SZ)_r18pt2_XXXXXX")"
DOWNSTREAM_CODE_ROOT="$DOWNSTREAM_BATCH/code"
TARGET_CACHE_ROOT="$DOWNSTREAM_BATCH/targets"
export PROJECT_ROOT DOWNSTREAM_BATCH DOWNSTREAM_CODE_ROOT TARGET_CACHE_ROOT VENV_ACTIVATE

mkdir -p "$DOWNSTREAM_BATCH/logs" "$DOWNSTREAM_BATCH/runs" \
    "$DOWNSTREAM_CODE_ROOT/benchmarks/vulcan_subset_downstream" \
    "$DOWNSTREAM_CODE_ROOT/OT-SSL-DD" "$TARGET_CACHE_ROOT"
for target in CIFAR10 CIFAR100 Aircraft CUB2011 Dogs Flowers TinyImageNet; do
    ln -s "$cache_root/$target.pt" "$TARGET_CACHE_ROOT/$target.pt"
done

upstream_code="$UPSTREAM_BATCH/code"
for source_file in \
    benchmarks/benchmark_cifar10.py \
    benchmarks/benchmark_dataset.py \
    benchmarks/benchmark_subset_refresh.py \
    benchmarks/downstream_data.py \
    OT-SSL-DD/utils.py \
    OT-SSL-DD/networks.py; do
    [[ -s $upstream_code/$source_file ]] || fail "missing upstream snapshot file: $upstream_code/$source_file"
done

cp "$upstream_code/benchmarks/benchmark_cifar10.py" \
    "$upstream_code/benchmarks/benchmark_dataset.py" \
    "$upstream_code/benchmarks/benchmark_subset_refresh.py" \
    "$upstream_code/benchmarks/downstream_data.py" \
    "$DOWNSTREAM_CODE_ROOT/benchmarks/"
cp benchmarks/vulcan_subset_downstream/environment.sh \
    benchmarks/vulcan_subset_downstream/job.sh \
    "$DOWNSTREAM_CODE_ROOT/benchmarks/vulcan_subset_downstream/"
cp "$upstream_code/OT-SSL-DD/utils.py" "$upstream_code/OT-SSL-DD/networks.py" \
    "$DOWNSTREAM_CODE_ROOT/OT-SSL-DD/"
cp OT-SSL-DD/training_epochs.json "$DOWNSTREAM_CODE_ROOT/OT-SSL-DD/training_epochs.json"
cp OT-SSL-DD/training_epochs.json "$DOWNSTREAM_BATCH/training_epochs.json"
printf '%s\n' "$UPSTREAM_BATCH" > "$DOWNSTREAM_BATCH/upstream_batch.txt"

printf 'config\tsource\tmethod\tmodel\tssl_method\tsubset_percentage\tssl_epochs\tencoder_dir\n' \
    > "$DOWNSTREAM_BATCH/configs.tsv"

ready=0
while IFS=$'\t' read -r config source method model ssl subset epochs runs probe1 probe5 pretrain_job evaluate_job; do
    [[ $config == config ]] && continue
    [[ $model == ResNet18 && $subset == 2 ]] || continue
    [[ $source == CIFAR10 || $source == CIFAR100 ]] || continue

    encoder_dir="$UPSTREAM_BATCH/runs/$config/encoders"
    for seed in 000 001 002 003 004 005 006 007 008 009; do
        [[ -s $encoder_dir/seed_$seed.pt ]] || fail "missing checkpoint: $encoder_dir/seed_$seed.pt"
    done
    [[ -s $encoder_dir/config.json ]] || fail "missing encoder metadata: $encoder_dir/config.json"
    jq -e \
        --arg source "$source" --arg method "$method" --arg ssl "$ssl" \
        --argjson subset "$subset" --argjson epochs "$epochs" \
        '.source_dataset == $source and .method == $method and
         .model == "ResNet18" and .ssl_method == $ssl and
         .subset_percentage == $subset and .ssl_epochs == $epochs' \
        "$encoder_dir/config.json" >/dev/null || fail "encoder metadata mismatch: $config"

    printf '%s\t%s\t%s\tResNet18\t%s\t%s\t%s\t%s\n' \
        "$config" "$source" "$method" "$ssl" "$subset" "$epochs" "$encoder_dir" \
        >> "$DOWNSTREAM_BATCH/configs.tsv"
    ready=$((ready + 1))
done < "$UPSTREAM_BATCH/configs.tsv"

(( ready == 8 )) || fail "expected exactly 8 ready ResNet18 2% configurations; found $ready"

cp benchmarks/vulcan_subset_downstream/submit_resnet18_size2.sh "$DOWNSTREAM_BATCH/submit.sh"
tar -czf "$DOWNSTREAM_BATCH/source.tar.gz" -C "$DOWNSTREAM_BATCH" \
    code submit.sh training_epochs.json upstream_batch.txt configs.tsv

job_id=$(sbatch --parsable \
    --account="$ACCOUNT" \
    --job-name=vsd_r18_pt2 \
    --time=01:30:00 \
    --chdir="$PROJECT_ROOT" \
    --array=0-7%8 \
    --output="$DOWNSTREAM_BATCH/logs/%x-%A_%a.out" \
    "$DOWNSTREAM_CODE_ROOT/benchmarks/vulcan_subset_downstream/job.sh")
job_id=${job_id%%;*}

printf 'submitted_utc\tjob_id\taccount\tarray\tconfigs\tevaluations\n' > "$DOWNSTREAM_BATCH/submitted.tsv"
printf '%s\t%s\t%s\t0-7%%8\t8\t96\n' \
    "$(date -u +%FT%TZ)" "$job_id" "$ACCOUNT" >> "$DOWNSTREAM_BATCH/submitted.tsv"

printf 'Submitted %s\nBatch: %s\nAccount: %s\nScope: 8 ResNet18 2%% configs, 96 downstream cells, no SSL training\n' \
    "$job_id" "$DOWNSTREAM_BATCH" "$ACCOUNT"
