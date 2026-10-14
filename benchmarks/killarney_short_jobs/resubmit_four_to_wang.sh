#!/bin/bash
set -euo pipefail

# Resubmit the four jobs that were originally charged to aip-yiweilu.
#
# Before running this script, use the cancellation sequence printed by Codex.
# It holds job 6214275 and removes its dependency on the old expert job before
# cancelling 6214274, preventing the rest of the Tiny VGG chain from being
# killed by an invalid dependency.

PROJECT_ROOT=$(readlink -f /home/mmoslem3/scratch/MKDT)
BUNDLE="$PROJECT_ROOT/benchmarks/killarney_short_jobs"
WORKER="$BUNDLE/mkdt_job.sh"
TINY_WORKER="$BUNDLE/synthesis_part3_job.sh"
LOG_ROOT="$BUNDLE/logs/wang_replacements"
MANIFEST="$BUNDLE/submitted_wang_replacements.tsv"
ACCOUNT=aip-boyuwang
SORT_JOB=6214275
REMAINING_EXPERTS=6214270:6214271:6214272:6214273
OLD_JOBS=6196445,6196455,6196465,6214274

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
command -v scontrol >/dev/null 2>&1 || fail "scontrol is not available"
command -v squeue >/dev/null 2>&1 || fail "squeue is not available"
[[ -s $WORKER ]] || fail "missing worker: $WORKER"
[[ -s $TINY_WORKER ]] || fail "missing worker: $TINY_WORKER"
[[ ! -e $MANIFEST ]] || fail "$MANIFEST already exists; refusing duplicate submissions"

# Refuse to submit duplicates if any of the old jobs are still in the queue.
if [[ -n $(squeue -h -j "$OLD_JOBS" 2>/dev/null) ]]; then
    fail "one or more old jobs are still queued; run the cancellation sequence first"
fi

REWIRE_SORT=0
RELEASE_SORT=0
if sort_details=$(scontrol show job "$SORT_JOB" -o 2>/dev/null); then
    if [[ $sort_details == *"JobState=PENDING"* ]]; then
        REWIRE_SORT=1
        if [[ $sort_details == *"Reason=JobHeldUser"* ]]; then
            RELEASE_SORT=1
        fi
    else
        echo "Notice: sort job $SORT_JOB is not pending; submitting the four replacements without rewiring it."
    fi
else
    echo "Notice: sort job $SORT_JOB is no longer known to Slurm; submitting the four replacements only."
fi

mkdir -p "$LOG_ROOT"
printf 'submitted_utc\toriginal_job_id\treplacement_job_id\taccount\tname\ttime\tmem\n' > "$MANIFEST"

LAST_JOB_ID=
submit_gpu() {
    local original_job=$1
    local name=$2
    local limit=$3
    local memory=$4
    local script=$5
    shift 5

    local submitted
    submitted=$(sbatch --parsable \
        --account="$ACCOUNT" \
        --job-name="$name" \
        --nodes=1 \
        --ntasks=1 \
        --cpus-per-task=1 \
        --gres=gpu:l40s:1 \
        --time="$limit" \
        --mem="$memory" \
        --chdir="$PROJECT_ROOT" \
        --output="$LOG_ROOT/%x-%j.out" \
        "$script" "$@")
    LAST_JOB_ID=${submitted%%;*}

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$(date -u +%FT%TZ)" "$original_job" "$LAST_JOB_ID" "$ACCOUNT" \
        "$name" "$limit" "$memory" >> "$MANIFEST"
    echo "Submitted $name as $LAST_JOB_ID on $ACCOUNT"
}

# Replace the cancelled Tiny VGG expert job first, then reconnect and release
# the already-submitted downstream chain.
submit_gpu 6214274 kil_wang_exp_Tiny_VGG_80 02:30:00 20G \
    "$TINY_WORKER" experts Tiny VGG11 80
new_expert=$LAST_JOB_ID

if [[ $REWIRE_SORT == 1 ]]; then
    scontrol update JobId="$SORT_JOB" \
        Dependency="afterok:${REMAINING_EXPERTS}:${new_expert}"
    if [[ $RELEASE_SORT == 1 ]]; then
        scontrol release "$SORT_JOB"
        echo "Rewired and released sort job $SORT_JOB; new dependency includes $new_expert"
    else
        echo "Rewired sort job $SORT_JOB; new dependency includes $new_expert"
    fi
fi

# These evaluation jobs had already cleared their pretraining dependencies and
# were pending only for priority, so their input encoders are already present.
submit_gpu 6196445 kil_wang_ev_CIFAR10_VGG_1_barlow 01:00:00 4G \
    "$WORKER" evaluate CIFAR10 VGG11 VGG11 1 barlowtwins 10000 all

submit_gpu 6196455 kil_wang_ev_CIFAR100_VGG_1_simclr 01:00:00 4G \
    "$WORKER" evaluate CIFAR100 VGG11 VGG11 1 simclr 5000 main

submit_gpu 6196465 kil_wang_ev_CIFAR100_VGG_5_barlow 01:00:00 4G \
    "$WORKER" evaluate CIFAR100 VGG11 VGG11 5 barlowtwins 6000 main

echo "All four replacements submitted. Manifest: $MANIFEST"
