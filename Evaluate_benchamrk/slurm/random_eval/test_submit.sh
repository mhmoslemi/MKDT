#!/bin/bash
# Exercise the real Bash launcher with fake Slurm commands, inside an allocation.
set -euo pipefail
: "${SLURM_JOB_ID:?Run this test in Slurm}"
: "${SLURM_TMPDIR:?}"
scripts=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
test_root=$(mktemp -d "$SLURM_TMPDIR/random_eval_submit.XXXXXX")
trap 'rm -rf -- "$test_root"' EXIT
fixture="$test_root/MKDT/Evaluate_benchamrk"
mkdir -p "$fixture/slurm/random_eval" "$fixture/random_ssl_checkpoints" "$test_root/venv/bin" "$test_root/data"
cp "$scripts/../../random_eval.sh" "$fixture/"
cp "$scripts/tasks.tsv" "$fixture/slurm/random_eval/"
for name in slurm/random_eval/worker.py slurm/random_eval/report.py table_coverage.tsv evaluate.py protocol.json; do
    printf 'fixture\n' > "$fixture/$name"
done
printf '# fixture\n' > "$test_root/venv/bin/activate"
export BENCH_VENV="$test_root/venv" BENCH_DATA_ROOT="$test_root/data"
export BENCH_CHECKPOINT_ROOT="$fixture/random_ssl_checkpoints"
export MOCK_CALLS="$test_root/calls" MOCK_QUEUE="$test_root/queue" MOCK_CLUSTER=killarney
unset BENCH_ACCOUNT BENCH_SECONDARY_ACCOUNT GPU_TYPE
: > "$MOCK_QUEUE"
module() { :; }
sinfo() { printf 'gpu:l40s:4\n'; }
sacctmgr() { printf 'aip-boyuwang|\naip-yiweilu|\n'; }
squeue() { cat "$MOCK_QUEUE"; }
scontrol() { printf 'ClusterName = %s\n' "$MOCK_CLUSTER"; }
sbatch() {
    printf '%s ' "$@" >> "$MOCK_CALLS"
    printf '\n' >> "$MOCK_CALLS"
    printf '%s\n' "$((900000 + $(wc -l < "$MOCK_CALLS")))"
}
export -f module sinfo sacctmgr squeue scontrol sbatch
make_checkpoints() {
    local which=$1 task source_name model subset ssl seed seed_name folder
    while IFS=$'\t' read -r task source_name model subset ssl seed; do
        [[ $task != task ]] || continue
        if [[ $which == ready && ( $task == 39 || $task == 40 ) ]]; then continue; fi
        printf -v seed_name 'seed_%03d' "$seed"
        folder="$BENCH_CHECKPOINT_ROOT/$source_name/$model/random/pt$subset/$ssl/$seed_name"
        mkdir -p "$folder"
        for name in encoder.pt config.json ssl_complete.json; do printf 'fixture\n' > "$folder/$name"; done
    done < "$scripts/tasks.tsv"
}
launch() { (unset SLURM_JOB_ID; bash "$fixture/random_eval.sh") > "$test_root/output" 2>&1; }
make_checkpoints ready
if launch; then echo 'Missing checkpoints unexpectedly passed' >&2; exit 1; fi
[[ ! -s $MOCK_CALLS ]]
cat > "$MOCK_QUEUE" <<'QUEUE'
6334332|random_TinyImageNet_ResNet18_1_simclr|RUNNING
6334333|random_TinyImageNet_ResNet18_1_barlowtwins|RUNNING
6334334|random_TinyImageNet_ResNet18_2_simclr|RUNNING
6334335|random_TinyImageNet_ResNet18_2_barlowtwins|RUNNING
QUEUE
export MOCK_CLUSTER=vulcan
if launch; then echo 'Cross-cluster dependencies unexpectedly passed' >&2; exit 1; fi
[[ ! -s $MOCK_CALLS ]]
export MOCK_CLUSTER=killarney
launch || { cat "$test_root/output"; exit 1; }
[[ $(wc -l < "$MOCK_CALLS") == 40 ]]
[[ $(grep -c -- '--dependency=afterok:' "$MOCK_CALLS") == 2 ]]
grep -- '--job-name=random_eval_39 ' "$MOCK_CALLS" | grep -qE -- '--dependency=afterok:(6334332:6334333|6334333:6334332) '
grep -- '--job-name=random_eval_40 ' "$MOCK_CALLS" | grep -qE -- '--dependency=afterok:(6334334:6334335|6334335:6334334) '
[[ $(grep -c -- '--account=aip-yiweilu ' "$MOCK_CALLS") == 4 ]]
[[ $(grep -c -- '--gres=gpu:l40s:1 ' "$MOCK_CALLS") == 40 ]]
[[ $(grep -c -- '--worker ' "$MOCK_CALLS") == 40 ]]
make_checkpoints all
: > "$MOCK_QUEUE"
: > "$MOCK_CALLS"
launch || { cat "$test_root/output"; exit 1; }
[[ $(wc -l < "$MOCK_CALLS") == 40 ]]
! grep -q -- '--dependency=' "$MOCK_CALLS"
mkdir -p "$fixture/random_eval_results/jobs"
printf '{"complete": true}\n' > "$fixture/random_eval_results/jobs/01.json"
printf '900002|random_eval_02|RUNNING\n' > "$MOCK_QUEUE"
: > "$MOCK_CALLS"
launch || { cat "$test_root/output"; exit 1; }
[[ $(wc -l < "$MOCK_CALLS") == 38 ]]
! grep -qE -- '--job-name=random_eval_0[12] ' "$MOCK_CALLS"
echo 'PASS Bash submission: 40 jobs, both dependencies, completed parents, resume, missing/cross-cluster guards'
