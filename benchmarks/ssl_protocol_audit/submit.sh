#!/bin/bash
set -euo pipefail
# One command on either server. Diagnose only checkpoints present there.
# Three paired seeds; frozen encoders; never pretrains SSL or updates tables.
PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)
cd "$PROJECT_ROOT"
fail() { echo "FATAL: $*" >&2; exit 1; }
ACCOUNT=aip-boyuwang
VENV_ACTIVATE=${VENV_ACTIVATE:-$HOME/ENV/bin/activate}
TARGET_CACHE="$PROJECT_ROOT/benchmarks/cross_downstream/results/batch_20261009T171449Z_o91BKP/targets/CIFAR100.pt"
FULL_BATCH="$PROJECT_ROOT/benchmarks/full_data_main_long/results/batch_20261016T042912Z_MLpgLE"
SUBSET_BATCH="$PROJECT_ROOT/benchmarks/vulcan_subset_refresh/results/batch_20261016T163046Z_utW9z3"
[[ -s $VENV_ACTIVATE && -s $TARGET_CACHE ]] || fail 'missing environment or CIFAR100 cache'
gpu_inventory=$(sinfo -h -o '%G' --Node | sort -u)
grep -q 'gpu:l40s:' <<< "$gpu_inventory" || fail 'gpu:l40s unavailable'
accounts=$(sacctmgr -nP show assoc user="$USER" format=Account)
grep -Fxq "$ACCOUNT" <<< "$accounts" || fail "$ACCOUNT not available"
shopt -s nullglob
submitted=0
for AUDIT_KIND in full random; do
    if [[ $AUDIT_KIND == full ]]; then
        upstream="$FULL_BATCH/code"
        ENCODER_DIR="$FULL_BATCH/encoders/cifar100_ResNet18_barlowtwins_full_size100_1500ep"
    else
        upstream="$SUBSET_BATCH/code"
        ENCODER_DIR="$SUBSET_BATCH/runs/CIFAR100_random_ResNet18_barlowtwins_pt0p2/encoders"
    fi
    ready=1
    for seed in 000 001 002; do
        matches=("$ENCODER_DIR"/shard_*/seed_"$seed".pt)
        [[ ! -s $ENCODER_DIR/seed_$seed.pt ]] || matches+=("$ENCODER_DIR/seed_$seed.pt")
        (( ${#matches[@]} <= 1 )) || fail "$AUDIT_KIND seed $seed has duplicate checkpoints"
        if (( ${#matches[@]} != 1 )) || [[ ! -s ${matches[0]} ]]; then ready=0; fi
    done
    if (( ! ready )); then
        echo "SKIP $AUDIT_KIND: seeds 0-2 not present on this server"
        continue
    fi
    name="bt_audit_$AUDIT_KIND"
    queued=$(squeue -h -u "$USER" -n "$name" -o '%i')
    if [[ -n $queued ]]; then echo "SKIP $name: queued as $queued"; continue; fi
    for path in benchmarks/benchmark_cifar10.py benchmarks/benchmark_dataset.py \
        benchmarks/downstream_data.py OT-SSL-DD/utils.py OT-SSL-DD/networks.py; do
        [[ -s $upstream/$path ]] || fail "missing snapshot $upstream/$path"
    done
    mkdir -p benchmarks/ssl_protocol_audit/results
    batch=$(mktemp -d "$PROJECT_ROOT/benchmarks/ssl_protocol_audit/results/${AUDIT_KIND}_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")
    AUDIT_CODE="$batch/code"
    AUDIT_OUTPUT="$batch/diagnostic"
    mkdir -p "$AUDIT_CODE/benchmarks/ssl_protocol_audit" "$AUDIT_CODE/OT-SSL-DD"
    for path in benchmarks/benchmark_cifar10.py benchmarks/benchmark_dataset.py \
        benchmarks/downstream_data.py OT-SSL-DD/utils.py OT-SSL-DD/networks.py; do
        cp "$upstream/$path" "$AUDIT_CODE/$path"
    done
    cp benchmarks/audit_cifar100_barlow.py benchmarks/benchmark_full_refresh.py "$AUDIT_CODE/benchmarks/"
    cp benchmarks/ssl_protocol_audit/job.sh "$AUDIT_CODE/benchmarks/ssl_protocol_audit/"
    cp OT-SSL-DD/training_epochs.json "$AUDIT_CODE/OT-SSL-DD/"
    export AUDIT_CODE AUDIT_OUTPUT AUDIT_KIND ENCODER_DIR TARGET_CACHE VENV_ACTIVATE
    job=$(sbatch --parsable --account="$ACCOUNT" --job-name="$name" \
        --nodes=1 --ntasks=1 --cpus-per-task=1 --gres=gpu:l40s:1 --mem=8G \
        --time=00:20:00 --chdir="$PROJECT_ROOT" --output="$batch/%x-%j.out" \
        "$AUDIT_CODE/benchmarks/ssl_protocol_audit/job.sh")
    printf '%s\n' "$job" > "$batch/job_id.txt"
    printf 'Submitted %s as %s (%s): evaluation-only diagnostic, 3 seeds\n' "$AUDIT_KIND" "$job" "$ACCOUNT"
    submitted=$((submitted + 1))
done
printf 'Submitted %d diagnostic jobs; no SSL training.\n' "$submitted"
