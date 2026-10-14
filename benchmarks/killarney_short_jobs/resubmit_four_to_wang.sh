#!/bin/bash
set -euo pipefail

# Replace the four cancelled Yiwei Lu jobs and rebuild the complete downstream
# Tiny VGG dependency graph. Every replacement is charged to aip-boyuwang.

PROJECT_ROOT=$(readlink -f /home/mmoslem3/scratch/MKDT)
BUNDLE="$PROJECT_ROOT/benchmarks/killarney_short_jobs"
WORKER="$BUNDLE/mkdt_job.sh"
TINY_WORKER="$BUNDLE/synthesis_part3_job.sh"
LOG_ROOT="$BUNDLE/logs/wang_replacements"
LEGACY_MANIFEST="$BUNDLE/submitted_wang_replacements.tsv"
MANIFEST="$BUNDLE/submitted_wang_full_rebuild.tsv"
ACCOUNT=aip-boyuwang

# Old sort -> distill -> pretrain -> evaluation graph. These jobs must be
# removed because they ultimately depend on the cancelled expert job 6214274.
OLD_CHAIN=6214275,6214276,6214277,6214278,6214279,6214280,6214281,6214282,6214283,6214284,6214285,6214286,6214287,6214288,6214289,6214290

fail() {
    echo "FATAL: $*" >&2
    exit 1
}

case "$PROJECT_ROOT" in
    /scratch/*|/project/*) ;;
    *) fail "project root is not on scratch/project: $PROJECT_ROOT" ;;
esac

cd "$PROJECT_ROOT"
command -v sbatch >/dev/null 2>&1 || fail "sbatch is not available"
command -v scancel >/dev/null 2>&1 || fail "scancel is not available"
command -v squeue >/dev/null 2>&1 || fail "squeue is not available"
[[ -s $WORKER ]] || fail "missing worker: $WORKER"
[[ -s $TINY_WORKER ]] || fail "missing worker: $TINY_WORKER"

# Cancel only the obsolete downstream graph. Missing/already-cancelled IDs are
# ignored; active IDs are resolved first so scancel receives valid targets.
active_old=$(squeue -h -j "$OLD_CHAIN" -o '%A' 2>/dev/null || true)
if [[ -n $active_old ]]; then
    mapfile -t old_job_ids <<< "$active_old"
    scancel "${old_job_ids[@]}"
    echo "Cancelled obsolete downstream jobs: ${old_job_ids[*]}"
else
    echo "Obsolete downstream jobs are already absent from the queue."
fi

mkdir -p "$LOG_ROOT"
if [[ ! -e $MANIFEST ]]; then
    printf 'submitted_utc\treplacement_job_id\taccount\tclass\tname\ttime\tmem\tdependency\n' > "$MANIFEST"
fi

# Find a previously submitted replacement by name. This supports both the
# earlier four-job manifest and this script's resumable full-graph manifest.
lookup_job_id() {
    local wanted=$1
    local file id
    for file in "$MANIFEST" "$LEGACY_MANIFEST"; do
        [[ -s $file ]] || continue
        id=$(awk -F '\t' -v wanted="$wanted" '
            NR == 1 {
                for (i = 1; i <= NF; i++) {
                    if ($i == "replacement_job_id") job_col = i
                    if ($i == "name") name_col = i
                }
                next
            }
            job_col && name_col && $name_col == wanted { found = $job_col }
            END { if (found != "") print found }
        ' "$file")
        if [[ -n $id ]]; then
            printf '%s\n' "$id"
            return 0
        fi
    done
    return 1
}

LAST_JOB_ID=
submit_gpu() {
    local class=$1
    local name=$2
    local limit=$3
    local memory=$4
    local dependency=$5
    local script=$6
    shift 6

    local existing
    if existing=$(lookup_job_id "$name"); then
        LAST_JOB_ID=$existing
        echo "Reusing existing $name as $LAST_JOB_ID"
        return 0
    fi

    local args=(sbatch --parsable
        --account="$ACCOUNT"
        --job-name="$name"
        --nodes=1
        --ntasks=1
        --cpus-per-task=1
        --gres=gpu:l40s:1
        --time="$limit"
        --mem="$memory"
        --chdir="$PROJECT_ROOT"
        --output="$LOG_ROOT/%x-%j.out")
    if [[ -n $dependency ]]; then
        args+=(--dependency="$dependency" --kill-on-invalid-dep=yes)
    fi

    local submitted
    submitted=$("${args[@]}" "$script" "$@")
    LAST_JOB_ID=${submitted%%;*}
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$(date -u +%FT%TZ)" "$LAST_JOB_ID" "$ACCOUNT" "$class" \
        "$name" "$limit" "$memory" "$dependency" >> "$MANIFEST"
    echo "Submitted $name as $LAST_JOB_ID on $ACCOUNT"
}

# Replacement for cancelled expert shard 80-99.
submit_gpu mkdt_experts kil_wang_exp_Tiny_VGG_80 02:30:00 20G "" \
    "$TINY_WORKER" experts Tiny VGG11 80
tiny_expert=$LAST_JOB_ID

# Complete replacement Tiny graph:
# expert 80-99 -> sort -> {1%,2%,5% distill} -> {SimCLR,Barlow} -> evaluate.
submit_gpu mkdt_sort kil_wang_sort_Tiny_VGG 01:10:00 26G \
    "afterok:$tiny_expert" "$TINY_WORKER" sort Tiny VGG11
tiny_sort=$LAST_JOB_ID

declare -A tiny_synth
for size in 1 2 5; do
    submit_gpu mkdt_distill "kil_wang_syn_Tiny_VGG_${size}" 02:10:00 12G \
        "afterok:$tiny_sort" "$TINY_WORKER" distill Tiny VGG11 "$size" VGG11
    tiny_synth[$size]=$LAST_JOB_ID
done

declare -A sim_epochs=( [1]=5000 [2]=4000 [5]=3000 )
declare -A bt_epochs=( [1]=10000 [2]=8000 [5]=6000 )
for size in 1 2 5; do
    for ssl in simclr barlowtwins; do
        if [[ $ssl == simclr ]]; then
            epochs=${sim_epochs[$size]}
            short_ssl=simclr
            case "$size" in
                1) pre_limit=00:55:00 ;;
                2) pre_limit=01:00:00 ;;
                5) pre_limit=01:05:00 ;;
            esac
        else
            epochs=${bt_epochs[$size]}
            short_ssl=barlow
            case "$size" in
                1) pre_limit=01:00:00 ;;
                2) pre_limit=01:05:00 ;;
                5) pre_limit=01:20:00 ;;
            esac
        fi

        submit_gpu mkdt_pretrain "kil_wang_pre_Tiny_VGG_${size}_${short_ssl}" \
            "$pre_limit" 4G "afterok:${tiny_synth[$size]}" "$WORKER" \
            pretrain TinyImageNet VGG11 VGG11 "$size" "$ssl" "$epochs" main
        pretrain_id=$LAST_JOB_ID

        submit_gpu mkdt_evaluate "kil_wang_ev_Tiny_VGG_${size}_${short_ssl}" \
            00:15:00 4G "afterok:$pretrain_id" "$WORKER" \
            evaluate TinyImageNet VGG11 VGG11 "$size" "$ssl" "$epochs" main
    done
done

# Three standalone evaluation replacements from the original Part 2 batch.
submit_gpu mkdt_evaluate kil_wang_ev_CIFAR10_VGG_1_barlow 01:00:00 4G "" \
    "$WORKER" evaluate CIFAR10 VGG11 VGG11 1 barlowtwins 10000 all

submit_gpu mkdt_evaluate kil_wang_ev_CIFAR100_VGG_1_simclr 01:00:00 4G "" \
    "$WORKER" evaluate CIFAR100 VGG11 VGG11 1 simclr 5000 main

submit_gpu mkdt_evaluate kil_wang_ev_CIFAR100_VGG_5_barlow 01:00:00 4G "" \
    "$WORKER" evaluate CIFAR100 VGG11 VGG11 5 barlowtwins 6000 main

echo "The complete 20-job Wang-funded replacement graph is now submitted."
echo "Manifest: $MANIFEST"
