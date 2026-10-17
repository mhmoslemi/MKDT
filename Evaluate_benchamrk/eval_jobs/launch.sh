#!/bin/bash
# Shared implementation for the two NEW evaluation-only launchers.
set -euo pipefail
method=${1:?Use eval_kmeans_vulcan.sh or eval_random_killarney.sh}
shift
case $method in
    kmeans) cluster=vulcan; seed_count=6; expected_probes=1944
        checkpoint_name=kmeans_ssl_checkpoints_cifar10k8k6k_tiny6k5k4k_6seeds ;;
    random) cluster=killarney; seed_count=10; expected_probes=3240
        checkpoint_name=random_ssl_checkpoints_cifar10k8k6k_tiny6k5k4k ;;
    *) echo 'Unsupported selection method' >&2; exit 2 ;;
esac
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
bench_dir=$(cd -- "$script_dir/.." && pwd -P)

if [[ ${1:-} == --worker ]]; then
    shift
    [[ $# == 5 ]] || { echo 'Invalid worker arguments' >&2; exit 2; }
    : "${SLURM_JOB_ID:?This worker must run in a Slurm job}"
    [[ ${SLURM_CLUSTER_NAME:-${CC_CLUSTER:-}} == "$cluster" ]] || {
        echo "This worker belongs on $cluster" >&2; exit 2;
    }
    task=$1 venv=$2 data_root=$3 checkpoint_root=$4 output_root=$5
    module --force purge
    module load StdEnv/2023 python/3.11.5
    source "$venv/bin/activate"
    export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
    export HF_HOME="$output_root/cache/hf" XDG_CACHE_HOME="$output_root/cache/xdg"
    export TORCH_HOME="$output_root/cache/torch" TMPDIR="${SLURM_TMPDIR:?}"
    cd "$bench_dir"
    exec python "$script_dir/worker.py" "$method" "$task" "$data_root" "$checkpoint_root" "$output_root"
fi

case ${1:---submit} in
    --submit) dry_run=0 ;;
    --dry-run) dry_run=1 ;;
    --help|-h)
        printf '%s\n' \
            "Evaluation only: $method on $cluster, 120 ordinary jobs, $seed_count seeds/configuration." \
            'Use bash on the correct login node AFTER all SSL checkpoints are complete.' \
            'Options: --dry-run (offline preview; no writes), --submit (default), --help.' \
            '1 L40S, 1 CPU, 6G; no arrays, dependencies, SSL training, or checkpoint deletion.' \
            '100-epoch frozen-encoder probes at 1% and 5% labels; main and downstream tables.' \
            'Walltimes are approximate projections + exactly 15 minutes, NOT measured probe times.' \
            'BENCH_EVAL_TIME overrides all walltimes, e.g. BENCH_EVAL_TIME=06:00:00.' \
            'Optional paths: BENCH_VENV, BENCH_DATA_ROOT, BENCH_CHECKPOINT_ROOT, BENCH_EVAL_OUTPUT_ROOT.' \
            'Rerunning skips active/completed jobs; finished probes survive an interrupted job.' \
            'An interrupted individual probe restarts its head; the SSL encoder is never retrained.'
        exit 0 ;;
    *) echo 'Use --submit, --dry-run, or --help' >&2; exit 2 ;;
esac
[[ $# -le 1 && -z ${SLURM_JOB_ID:-} ]] || { echo 'Run with bash on the login node, not sbatch.' >&2; exit 2; }
tasks="$script_dir/${method}_tasks.tsv"
jobs="$script_dir/${method}_jobs.tsv"
checkpoint_root=${BENCH_CHECKPOINT_ROOT:-$bench_dir/$checkpoint_name}
output_root=${BENCH_EVAL_OUTPUT_ROOT:-$bench_dir/${method}_eval_results_current_${seed_count}seeds_120jobs}
data_root=${BENCH_DATA_ROOT:-$bench_dir/../../data}
venv=${BENCH_VENV:-$HOME/ENV}
for path in "$tasks" "$jobs" "$script_dir/worker.py" "$script_dir/report.py" \
    "$bench_dir/evaluate.py" "$bench_dir/protocol.json" "$bench_dir/table_coverage.tsv" "$bench_dir/${method}_ssl.sh"; do
    [[ -r $path ]] || { echo "Missing $path; sync the new evaluation files and current code." >&2; exit 2; }
done
# Verify manifest counts and per-job time arithmetic without running Python.
awk -F '\t' -v ns="$seed_count" -v np="$expected_probes" '
    NR==FNR {if(FNR==1)next; c[$1]++; p[$1]+=($3=="ResNet18"?14:2); t[$1]+=$7;
        key=$2 FS $3 FS $4 FS $5 FS $6;
        if(seen[key]++ || $6<0 || $6>=ns)bad=1; next}
    FNR==1 {next}
    {j++; n+=$3; probes+=$4; split($6,h,":");
     if(done[$1]++ || c[$1]!=$3 || p[$1]!=$4 || t[$1]!=$5 || h[1]*3600+h[2]*60+h[3]!=$5+900)bad=1}
    END {if(bad || j!=120 || n!=54*ns || probes!=np)exit 2}
' "$tasks" "$jobs" || { echo 'Invalid evaluation task manifest.' >&2; exit 2; }

printf '%s on %s: 120 jobs / %s encoders / %s probes.\n' "$method" "$cluster" "$((54*seed_count))" "$expected_probes"
printf 'task\taccount\tencoders\tprobes\testimated_seconds\trequested_walltime\n'
if (( dry_run )); then
    while IFS=$'\t' read -r task account encoders probes seconds walltime; do
        [[ $task != task ]] || continue
        printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$task" "$account" "$encoders" "$probes" "$seconds" "${BENCH_EVAL_TIME:-$walltime}"
    done < "$jobs"
    printf 'Checkpoint input: %s\nResults: %s\nDry run: nothing submitted or written.\n' "$checkpoint_root" "$output_root"
    exit 0
fi
actual_cluster=$(scontrol show config | awk '$1 == "ClusterName" {print $3}')
[[ $actual_cluster == "$cluster" ]] || { echo "Run this launcher on $cluster, not $actual_cluster." >&2; exit 2; }
[[ -r $venv/bin/activate && -d $data_root && -d $checkpoint_root ]] || {
    echo 'Missing venv, data, or checkpoint root. Set the BENCH_* path overrides if needed.' >&2; exit 2;
}
venv=$(cd -- "$venv" && pwd -P)
data_root=$(cd -- "$data_root" && pwd -P)
checkpoint_root=$(cd -- "$checkpoint_root" && pwd -P)
output_root=$(realpath -m -- "$output_root")
[[ $output_root == "$bench_dir/"* && $output_root != "$checkpoint_root" && $output_root != "$checkpoint_root/"* ]] || {
    echo 'Results must be in a separate directory inside Evaluate_benchamrk.' >&2; exit 2;
}
module spider python/3.11.5 >/dev/null 2>&1
inventory=$(sinfo -h -o '%G' --Node)
printf '%s\n' "$inventory" | tr ',' '\n' | grep -Eq '^gpu:l40s:' || {
    echo 'No L40S GRES found; these timing estimates target L40S.' >&2; exit 2;
}
accounts=$(sacctmgr -nP show assoc "user=$(id -un)" format=Account)
for account in aip-boyuwang aip-yiweilu; do
    printf '%s\n' "$accounts" | cut -d'|' -f1 | grep -Fxq -- "$account" || {
        echo "Account unavailable: $account" >&2; exit 2;
    }
done
# Check ALL inputs before submitting the first job. Final markers are validated
# cryptographically by the worker; a rolling training.pt is not a final encoder.
missing=0
while IFS=$'\t' read -r task source_name model subset ssl seed seconds; do
    [[ $task != task ]] || continue
    printf -v seed_name 'seed_%03d' "$seed"
    folder="$checkpoint_root/$source_name/$model/$method/pt$subset/$ssl/$seed_name"
    for name in config.json ssl_complete.json encoder.pt; do
        if [[ ! -s $folder/$name ]]; then
            printf 'MISSING %s\n' "$folder/$name" >&2
            missing=$((missing+1))
        fi
    done
done < "$tasks"
(( missing == 0 )) || { echo "$missing missing files: finish SSL first. No jobs submitted." >&2; exit 2; }
mkdir -p "$output_root/logs"
exec 9>"$output_root/.submit.lock"
flock -n 9 || { echo 'Another evaluation launcher is running.' >&2; exit 2; }
declare -A active
queue=$(squeue -u "$(id -un)" -h -o '%j')
while read -r name; do [[ -z $name ]] || active[$name]=1; done <<< "$queue"
receipt="$output_root/submitted_$(date -u +%Y%m%dT%H%M%SZ)_$$.tsv"
printf 'job_id\ttask\taccount\tencoders\tprobes\twalltime\n' > "$receipt"
submitted=0
while IFS=$'\t' read -r task account encoders probes seconds walltime; do
    [[ $task != task ]] || continue
    name="${method}_probe_current_120_$task"
    if [[ -n ${active[$name]:-} || -s $output_root/jobs/$task.json ]]; then
        printf 'SKIP %s: already active or completed.\n' "$task"
        continue
    fi
    walltime=${BENCH_EVAL_TIME:-$walltime}
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$task" "$account" "$encoders" "$probes" "$seconds" "$walltime"
    job_id=$(sbatch --parsable --job-name="$name" --account="$account" \
        --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=6G --gres=gpu:l40s:1 \
        --time="$walltime" --chdir="$bench_dir" \
        --output="$output_root/logs/%x-%j.out" --error="$output_root/logs/%x-%j.err" \
        "$script_dir/launch.sh" "$method" --worker "$task" "$venv" "$data_root" "$checkpoint_root" "$output_root")
    job_id=${job_id%%;*}
    [[ $job_id =~ ^[0-9]+$ ]] || { echo "Unexpected sbatch response: $job_id" >&2; exit 2; }
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$job_id" "$task" "$account" "$encoders" "$probes" "$walltime" >> "$receipt"
    submitted=$((submitted+1))
done < "$jobs"
printf 'Submitted %s jobs. IDs: %s\nResults: %s\n' "$submitted" "$receipt" "$output_root"
