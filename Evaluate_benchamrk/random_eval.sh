#!/bin/bash
# Run with bash on the cluster holding the retained encoders. No job arrays.
set -euo pipefail
bench=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
scripts="$bench/slurm/random_eval"
mode=${1:---submit}
ensure_node() {
    if ! command -v node >/dev/null 2>&1; then
        module spider nodejs/20.16.0 >/dev/null 2>&1
        module load StdEnv/2023 nodejs/20.16.0
    fi
}
case "$mode" in
    --dry-run) ensure_node; exec node "$scripts/plan.js" preview "$bench" ;;
    --prepare) ensure_node; exec node "$scripts/plan.js" prepare "$bench" ;;
    --submit|--resume) ;;
    --help|-h)
        cat <<'HELP'
bash Evaluate_benchamrk/random_eval.sh  # prepare and submit all 40 jobs
bash Evaluate_benchamrk/random_eval.sh --dry-run
bash Evaluate_benchamrk/random_eval.sh --prepare
bash Evaluate_benchamrk/random_eval.sh --submit BATCH
bash Evaluate_benchamrk/random_eval.sh --resume BATCH

40 ordinary evaluation jobs total: 38 ready + 2 dependent TinyImageNet jobs.
540 retained encoders; 3240 probes; 324 active random-subset table cells.
No SSL training, encoder deletion, 5%-pretraining rows, or table overwrites.
With no arguments, prepare and submit a new batch on this cluster.
--prepare prints the batch path; run it on Killarney after syncing with git.
--submit requires an unsubmitted batch. --resume only submits unfinished jobs
whose previous submission has left squeue; completed probes are verified/skipped.
BENCH_VENV defaults to $HOME/ENV, BENCH_DATA_ROOT to <MKDT parent>/data.
BENCH_CHECKPOINT_ROOT defaults to Evaluate_benchamrk/random_ssl_checkpoints.
BENCH_EVAL_TIME=06:00:00 and BENCH_EVAL_TAIL_TIME=12:00:00 at preparation.
BENCH_ACCOUNT defaults to aip-boyuwang; BENCH_SECONDARY_ACCOUNT=aip-yiweilu
receives four of the 40 jobs. Set both to the same account if appropriate.
GPU_TYPE defaults to l40s and is checked against destination sinfo at submission.
HELP
        exit 0 ;;
    *) echo 'Unknown option; use --help' >&2; exit 2 ;;
esac
if [[ $mode == --submit && $# -le 1 ]]; then
    [[ -z ${SLURM_JOB_ID:-} ]] || {
        echo 'Run this launcher with bash on the login node.' >&2; exit 2;
    }
    ensure_node
    prepared_batch=$(node "$scripts/plan.js" prepare "$bench")
    printf 'Prepared batch: %s\n' "$prepared_batch"
    set -- --submit "$prepared_batch"
fi
[[ $# == 2 && -z ${SLURM_JOB_ID:-} ]] || {
    echo 'Use --submit/--resume BATCH with bash on the login node.' >&2; exit 2;
}
batch=$(cd -- "$2" && pwd -P)
ensure_node
[[ -s $batch/plan.json && -r $batch/plan.js ]] || exit 2
exec 9>"$batch/.submission.lock"
flock -n 9 || { echo 'Another submission is in progress' >&2; exit 2; }
if [[ $mode == --submit && -e $batch/submitted.tsv ]]; then
    echo 'This batch was already submitted. Use --resume for unfinished jobs.' >&2
    exit 2
fi
# Explicit discovery on the destination; dependencies cannot cross clusters.
module spider python/3.11.5 >/dev/null 2>&1
sinfo -s
sinfo -h -o '%G' --Node > "$batch/gres.txt"
sacctmgr -nP show assoc "user=$USER" format=Account > "$batch/accounts.txt"
squeue -u "$USER" -h -o '%i|%j|%T' > "$batch/queue.txt"
cluster=$(scontrol show config | awk '$1 == "ClusterName" {print $3; exit}')
[[ -n $cluster ]] || { echo 'Cannot identify the destination cluster' >&2; exit 2; }
node "$batch/plan.js" check "$batch" "$cluster" "${GPU_TYPE:-l40s}"
# Check ALL inputs/dependencies before submitting the first job. The worker
# repeats strict checkpoint/data validation inside its GPU allocation.
if [[ ! -e $batch/submitted.tsv ]]; then
    printf 'task\tjob_id\tdependency\tcluster\n' > "$batch/submitted.tsv"
fi
while IFS=$'\t' read -r task account walltime dependency gpu; do
    [[ $task != task ]] || continue
    args=()
    [[ $dependency == - ]] || args+=("--dependency=$dependency" --kill-on-invalid-dep=yes)
    job_id=$(sbatch --parsable --job-name="random_eval_${task}" --account="$account" \
        --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=6G --gres="gpu:$gpu:1" \
        --time="$walltime" --chdir="$batch" --output="$batch/logs/%x-%j.out" \
        --error="$batch/logs/%x-%j.err" "${args[@]}" "$batch/worker.sh" "$batch" "$task")
    job_id=${job_id%%;*}
    [[ $job_id =~ ^[0-9]+$ ]] || { echo "Unexpected sbatch output: $job_id" >&2; exit 2; }
    printf '%s\t%s\t%s\t%s\n' "$task" "$job_id" "$dependency" "$cluster" >> "$batch/submitted.tsv"
    printf 'Submitted task %s as job %s (%s)\n' "$task" "$job_id" "$dependency"
done < "$batch/submit.tsv"
printf 'Submission record: %s/submitted.tsv\n' "$batch"
