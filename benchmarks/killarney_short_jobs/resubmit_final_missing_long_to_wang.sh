#!/bin/bash
set -euo pipefail

# Submit only the two unfinished long-epoch Full Data configurations:
#   CIFAR100 / VGG11 / Barlow Twins: seed 9
#   CIFAR100 / ResNet18 / SimCLR:    seeds 0,1,2
# Each evaluator waits for its replacement training array and writes both the
# 1% (500-epoch probe) and 5% (300-epoch probe) summaries through the frozen
# worker stored in the original batch.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
PROJECT_ROOT=$(readlink -f "$SCRIPT_DIR/../..")
MAINLONG_BATCH=${MAINLONG_BATCH:-$PROJECT_ROOT/benchmarks/full_data_main_long/results/batch_20261016T042912Z_MLpgLE}
ACCOUNT=${ACCOUNT:-aip-boyuwang}
VENV_ACTIVATE=${VENV_ACTIVATE:-$HOME/ENV/bin/activate}
EXCLUDED_NODES=${EXCLUDED_NODES:-kn072,kn088,kn117,kn140,kn161}
EXPECTED_PROTOCOL=full_data_main_simclr1000_barlow1500_v1

fail() {
    echo "FATAL: $*" >&2
    exit 1
}

case "$PROJECT_ROOT" in
    /scratch/*|/project/*) ;;
    *) fail "project root is not on scratch/project: $PROJECT_ROOT" ;;
esac

[[ -d $MAINLONG_BATCH ]] || fail "missing batch: $MAINLONG_BATCH"
[[ -s $MAINLONG_BATCH/protocol.txt ]] || fail "missing batch protocol"
[[ $(<"$MAINLONG_BATCH/protocol.txt") == "$EXPECTED_PROTOCOL" ]] || \
    fail "unexpected protocol in $MAINLONG_BATCH/protocol.txt"

MAINLONG_CODE_ROOT="$MAINLONG_BATCH/code"
WORKER="$MAINLONG_CODE_ROOT/benchmarks/killarney_short_jobs/full_data_main_long_job.sh"
AUDIT="$PROJECT_ROOT/benchmarks/killarney_short_jobs/check_full_data_main_long.sh"
MANIFEST="$MAINLONG_BATCH/final_missing_submitted.tsv"

[[ -s $WORKER ]] || fail "missing immutable batch worker: $WORKER"
[[ -s $VENV_ACTIVATE ]] || fail "missing Python environment: $VENV_ACTIVATE"

if [[ -s $MAINLONG_BATCH/targets/CIFAR100.pt ]]; then
    TARGET_CACHE_ROOT="$MAINLONG_BATCH/targets"
elif [[ -s $MAINLONG_BATCH/target_source_batch.txt ]]; then
    target_source_batch=$(<"$MAINLONG_BATCH/target_source_batch.txt")
    TARGET_CACHE_ROOT="$target_source_batch/targets"
else
    fail "cannot locate target cache for $MAINLONG_BATCH"
fi
[[ -s $TARGET_CACHE_ROOT/CIFAR100.pt ]] || \
    fail "missing prepared target cache: $TARGET_CACHE_ROOT/CIFAR100.pt"

command -v sbatch >/dev/null 2>&1 || fail "sbatch is not available"
command -v sinfo >/dev/null 2>&1 || fail "sinfo is not available"
command -v squeue >/dev/null 2>&1 || fail "squeue is not available"

gpu_inventory=$(sinfo -h -o '%G' --Node | sort -u)
grep -q 'gpu:l40s:' <<< "$gpu_inventory" || {
    printf 'Discovered GPU inventory:\n%s\n' "$gpu_inventory" >&2
    fail 'gpu:l40s is not present in the discovered Killarney inventory'
}

queued_names=$(squeue -h -u "$USER" -o '%j')
if grep -Eq '^kmlf_c100_(vg_b|r18_s)_(pre|eval)$' <<< "$queued_names"; then
    printf '%s\n' 'Matching final-retry jobs are already queued:' >&2
    grep -E '^kmlf_c100_(vg_b|r18_s)_(pre|eval)$' <<< "$queued_names" >&2
    fail 'refusing duplicate submission'
fi

mkdir -p "$MAINLONG_BATCH/logs"
if [[ ! -e $MANIFEST ]]; then
    printf 'submitted_utc\tjob_id\taccount\tkind\tmodel\tssl_method\tseeds\tdependency\texcluded_nodes\n' \
        > "$MANIFEST"
fi

export PROJECT_ROOT MAINLONG_BATCH MAINLONG_CODE_ROOT TARGET_CACHE_ROOT VENV_ACTIVATE

valid_evaluation_count() {
    local model=$1
    local ssl=$2
    local epochs=$3
    local label=$4
    local probe=$5
    local identifier="cifar100_${model}_${ssl}_full_size100_${epochs}ep"
    local eval_dir="$MAINLONG_BATCH/evaluations/${identifier}__CIFAR100_lbl${label}"
    local summary
    local valid=0
    local seed_files=()

    shopt -s nullglob
    for summary in "$eval_dir"/job_*/summary.json; do
        [[ -s $summary ]] || continue
        grep -Eq '"protocol"[[:space:]]*:[[:space:]]*"'"$EXPECTED_PROTOCOL"'"' "$summary" || continue
        grep -Eq '"source"[[:space:]]*:[[:space:]]*"CIFAR100"' "$summary" || continue
        grep -Eq '"target"[[:space:]]*:[[:space:]]*"CIFAR100"' "$summary" || continue
        grep -Eq '"model"[[:space:]]*:[[:space:]]*"'"$model"'"' "$summary" || continue
        grep -Eq '"ssl_method"[[:space:]]*:[[:space:]]*"'"$ssl"'"' "$summary" || continue
        grep -Eq '"ssl_epochs"[[:space:]]*:[[:space:]]*'"$epochs"'([,[:space:]]|$)' "$summary" || continue
        grep -Eq '"label_percentage"[[:space:]]*:[[:space:]]*'"$label"'([,[:space:]]|$)' "$summary" || continue
        grep -Eq '"probe_epochs"[[:space:]]*:[[:space:]]*'"$probe"'([,[:space:]]|$)' "$summary" || continue
        grep -Eq '"complete"[[:space:]]*:[[:space:]]*true' "$summary" || continue
        grep -Eq '"completed_runs"[[:space:]]*:[[:space:]]*10([,[:space:]]|$)' "$summary" || continue
        grep -Eq '"expected_runs"[[:space:]]*:[[:space:]]*10([,[:space:]]|$)' "$summary" || continue
        seed_files=("${summary%/summary.json}"/seed_*.json)
        (( ${#seed_files[@]} == 10 )) || continue
        valid=$((valid + 1))
    done
    printf '%d\n' "$valid"
}

missing_candidate_seeds() {
    local model=$1
    local ssl=$2
    local epochs=$3
    local candidates=$4
    local identifier="cifar100_${model}_${ssl}_full_size100_${epochs}ep"
    local encoder_root="$MAINLONG_BATCH/encoders/$identifier"
    local seed checkpoint
    local selected=()
    local matches=()

    shopt -s nullglob
    for seed in {0..9}; do
        printf -v checkpoint 'seed_%03d.pt' "$seed"
        matches=("$encoder_root"/shard_*/"$checkpoint")
        if (( ${#matches[@]} == 1 )) && [[ -s ${matches[0]} ]]; then
            continue
        fi
        (( ${#matches[@]} == 0 )) || \
            fail "$identifier seed $seed has ${#matches[@]} ambiguous/invalid checkpoints"
        case ",$candidates," in
            *",$seed,"*) selected+=("$seed") ;;
            *) fail "$identifier unexpectedly lacks previously successful seed $seed" ;;
        esac
    done

    if (( ${#selected[@]} > 0 )); then
        (IFS=,; printf '%s' "${selected[*]}")
    fi
}

submit_configuration() {
    local tag=$1
    local model=$2
    local ssl=$3
    local epochs=$4
    local candidates=$5
    local train_limit=$6
    local label1_count label5_count seed_spec pretrain_id evaluation_id
    local dependency=

    label1_count=$(valid_evaluation_count "$model" "$ssl" "$epochs" 1 500)
    label5_count=$(valid_evaluation_count "$model" "$ssl" "$epochs" 5 300)
    (( label1_count <= 1 && label5_count <= 1 )) || \
        fail "$tag has duplicate completed evaluations"
    if (( label1_count == 1 && label5_count == 1 )); then
        printf 'SKIP already complete: %s\n' "$tag"
        return
    fi
    (( label1_count == 0 && label5_count == 0 )) || \
        fail "$tag has only one completed label setting; refusing to duplicate it"

    seed_spec=$(missing_candidate_seeds "$model" "$ssl" "$epochs" "$candidates")
    if [[ -n $seed_spec ]]; then
        pretrain_id=$(sbatch --parsable \
            --account="$ACCOUNT" \
            --job-name="kmlf_${tag}_pre" \
            --array="$seed_spec" \
            --nodes=1 \
            --ntasks=1 \
            --cpus-per-task=1 \
            --gres=gpu:l40s:1 \
            --exclude="$EXCLUDED_NODES" \
            --time="$train_limit" \
            --mem=8G \
            --chdir="$PROJECT_ROOT" \
            --output="$MAINLONG_BATCH/logs/%x-%A_%a.out" \
            "$WORKER" pretrain CIFAR100 "$model" "$ssl")
        pretrain_id=${pretrain_id%%;*}
        dependency="afterok:$pretrain_id"
        printf '%s\t%s\t%s\tpretrain\t%s\t%s\t%s\t\t%s\n' \
            "$(date -u +%FT%TZ)" "$pretrain_id" "$ACCOUNT" "$model" "$ssl" \
            "$seed_spec" "$EXCLUDED_NODES" >> "$MANIFEST"
        printf 'Submitted %-22s as %s (seeds %s)\n' \
            "kmlf_${tag}_pre" "$pretrain_id" "$seed_spec"
    else
        printf 'All ten checkpoints already exist for %s; submitting evaluation only.\n' "$tag"
    fi

    eval_args=(sbatch --parsable
        --account="$ACCOUNT"
        --job-name="kmlf_${tag}_eval"
        --nodes=1
        --ntasks=1
        --cpus-per-task=1
        --gres=gpu:l40s:1
        --exclude="$EXCLUDED_NODES"
        --time=01:30:00
        --mem=8G
        --chdir="$PROJECT_ROOT"
        --output="$MAINLONG_BATCH/logs/%x-%j.out")
    if [[ -n $dependency ]]; then
        eval_args+=(--dependency="$dependency" --kill-on-invalid-dep=yes)
    fi
    evaluation_id=$("${eval_args[@]}" "$WORKER" evaluate CIFAR100 "$model" "$ssl")
    evaluation_id=${evaluation_id%%;*}
    printf '%s\t%s\t%s\tevaluate\t%s\t%s\t-\t%s\t%s\n' \
        "$(date -u +%FT%TZ)" "$evaluation_id" "$ACCOUNT" "$model" "$ssl" \
        "${dependency:--}" "$EXCLUDED_NODES" >> "$MANIFEST"
    printf 'Submitted %-22s as %s' "kmlf_${tag}_eval" "$evaluation_id"
    [[ -z $dependency ]] || printf ' (%s)' "$dependency"
    printf '\n'
}

submit_configuration c100_vg_b VGG11 barlowtwins 1500 9 03:30:00
submit_configuration c100_r18_s ResNet18 simclr 1000 0,1,2 05:30:00

printf '\nSubmitted the final missing long-run work on %s.\n' "$ACCOUNT"
printf 'Excluded nodes: %s\n' "$EXCLUDED_NODES"
printf 'Submission record: %s\n' "$MANIFEST"
printf 'After completion, check with:\n  bash %q %q\n' "$AUDIT" "$MAINLONG_BATCH"
