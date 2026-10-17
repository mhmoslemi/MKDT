#!/bin/bash
# Retry ONLY the nine CUDA-startup failures from the 2026-10-17 random batch.
# Run with bash on Killarney. No cancellation, arrays, or changes to training.
set -euo pipefail

case ${1:---submit} in
    --submit) dry_run=0 ;;
    --dry-run) dry_run=1 ;;
    --help|-h)
        printf '%s\n' \
            'bash Evaluate_benchamrk/retry_random_failed_killarney.sh --dry-run' \
            'bash Evaluate_benchamrk/retry_random_failed_killarney.sh  # submit on Killarney' \
            'Exactly nine original job configurations, two seeds each; all originally Wang.' \
            'Keeps original epochs, walltimes, 1 L40S, 1 CPU, 6G, and output directory.' \
            'Looks up the FAILED originals with sacct and excludes their recorded host(s).' \
            'Skips matching active jobs and seed pairs with both completed encoders.' \
            'Uses random_ssl.sh --worker, including its unchanged resume/checkpoint behavior.' \
            'BENCH_VENV and BENCH_DATA_ROOT override the existing environment and data paths.' \
            'Dry-run is offline: no scheduler calls, files, or submissions.'
        exit 0 ;;
    *) echo 'Use --submit (default), --dry-run, or --help.' >&2; exit 2 ;;
esac
[[ $# -le 1 ]] || { echo 'Unexpected extra arguments.' >&2; exit 2; }
[[ -z ${SLURM_JOB_ID:-} ]] || { echo 'Run with bash on the login node, not sbatch.' >&2; exit 2; }

# Columns: original job, source, network, subset %, SSL, first seed, epochs, walltime.
# Verified against random_ssl_checkpoints_cifar10k8k6k_tiny6k5k4k/
# submitted_20261017T050226Z_3146085.tsv. All nine had the same failing GPU UUID:
# GPU-831939f3-68ca-e0a0-b705-d7729ba0f2d8. Resolve its host from accounting, not a guess.
tasks=(
    '6341202 CIFAR10 VGG11 1 simclr 6 8000 02:08:40'
    '6341206 CIFAR10 VGG11 1 barlowtwins 4 8000 02:02:07'
    '6341225 CIFAR10 ResNet18 0.2 barlowtwins 0 10000 01:18:47'
    '6341228 CIFAR10 ResNet18 0.2 barlowtwins 6 10000 01:18:47'
    '6341236 CIFAR10 ResNet18 1 barlowtwins 2 8000 02:15:59'
    '6341268 CIFAR100 ConvNet 1 barlowtwins 6 8000 01:55:58'
    '6341285 CIFAR100 VGG11 0.2 barlowtwins 0 10000 01:12:21'
    '6341294 CIFAR100 VGG11 1 simclr 6 8000 02:03:22'
    '6341312 CIFAR100 ResNet18 0.2 simclr 2 10000 01:18:39'
)
account=aip-boyuwang
printf 'original_job\tdataset\tmodel\tsubset_percent\tssl\tseeds\tepochs_per_seed\taccount\twalltime\n'
for task in "${tasks[@]}"; do
    read -r old source_name model subset ssl seed_start epochs walltime <<< "$task"
    printf '%s\t%s\t%s\t%s\t%s\t%s..%s\t%s\t%s\t%s\n' \
        "$old" "$source_name" "$model" "$subset" "$ssl" "$seed_start" "$((seed_start+1))" \
        "$epochs" "$account" "$walltime"
done
if (( dry_run )); then
    echo 'Dry run: nine jobs / 18 seeds. Failed hosts will be resolved and excluded at submission.'
    exit 0
fi

cluster=$(scontrol show config | awk '$1 == "ClusterName" {print $3}')
[[ $cluster == killarney ]] || { echo 'Submit this script on Killarney only.' >&2; exit 2; }
bench_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
launcher="$bench_dir/random_ssl.sh"
output_root="$bench_dir/random_ssl_checkpoints_cifar10k8k6k_tiny6k5k4k"
venv=${BENCH_VENV:-$HOME/ENV}
data_root=${BENCH_DATA_ROOT:-$bench_dir/../../data}
for required in random_ssl.sh common.py data.py train_ssl.py run.py evaluate.py protocol.json \
                ../networks.py ../benchmarks/downstream_data.py; do
    [[ -r $bench_dir/$required ]] || { echo "Missing $bench_dir/$required" >&2; exit 2; }
done
[[ -r $venv/bin/activate && -d $data_root ]] || {
    echo 'Set BENCH_VENV and BENCH_DATA_ROOT to the existing Killarney paths.' >&2; exit 2;
}
venv=$(cd -- "$venv" && pwd -P)
data_root=$(cd -- "$data_root" && pwd -P)
for marker in cifar-10-batches-py/data_batch_{1..5} cifar-10-batches-py/batches.meta \
              cifar-100-python/train cifar-100-python/meta; do
    [[ -r $data_root/$marker ]] || { echo "Missing training data: $data_root/$marker" >&2; exit 2; }
done
inventory=$(sinfo -h -o '%G' --Node)
gpu_types=$(printf '%s\n' "$inventory" | tr ',' '\n' | sed -nE 's/^gpu:([^:]+):.*/\1/p' | sort -u)
grep -Fxq l40s <<< "$gpu_types" || { echo 'L40S GPUs are unavailable here.' >&2; exit 2; }
accounts=$(sacctmgr -nP show assoc "user=$(id -un)" format=Account)
printf '%s\n' "$accounts" | cut -d'|' -f1 | grep -Fxq "$account" || {
    echo "Account not available: $account" >&2; exit 2;
}

ids=6341202,6341206,6341225,6341228,6341236,6341268,6341285,6341294,6341312
records=$(sacct -X -nP -u "$(id -un)" -j "$ids" --starttime=2026-10-16 \
    --format=JobIDRaw,State,NodeList)
declare -A states=() hosts=() excluded=()
while IFS='|' read -r job state nodes rest; do
    [[ $job =~ ^[0-9]+$ ]] || continue
    states[$job]=$state
    hosts[$job]=$nodes
done <<< "$records"
for task in "${tasks[@]}"; do
    read -r old rest <<< "$task"
    [[ ${states[$old]:-} == FAILED ]] || {
        echo "Original $old is not confirmed FAILED in Killarney accounting; refusing submission." >&2
        exit 2
    }
    node=${hosts[$old]:-}
    [[ $node =~ ^[[:alnum:]][[:alnum:]_-]*$ && $node != None && $node != Unknown ]] || {
        echo "Cannot resolve a single failed host for $old: $node; refusing submission." >&2; exit 2;
    }
    excluded[$node]=1
done
exclude_nodes=$(printf '%s\n' "${!excluded[@]}" | sort | paste -sd, -)
echo "Excluding hosts recorded for the nine CUDA failures: $exclude_nodes"

mkdir -p "$output_root/logs"
exec 9>"$output_root/.retry_cuda_failures.lock"
flock -n 9 || { echo 'Another copy of this retry launcher is running.' >&2; exit 2; }
queue=$(squeue -h -u "$(id -un)" -o '%i|%j')
receipt="$output_root/retried_cuda_failures_$(date -u +%Y%m%dT%H%M%SZ)_$$.tsv"
printf 'original_job\tnew_job\taccount\tdataset\tmodel\tsubset_percent\tssl\tseeds\tepochs_per_seed\twalltime\texcluded_nodes\n' > "$receipt"
submitted=0
for task in "${tasks[@]}"; do
    read -r old source_name model subset ssl seed_start epochs walltime <<< "$task"
    seed_end=$((seed_start + 1))
    original_name="random_${source_name}_${model}_${subset/./p}_${ssl}_e${epochs}_s${seed_start}-${seed_end}"
    retry_name="random_retry_${old}"
    if awk -F '|' -v original="$original_name" -v retry="$retry_name" \
        '$2 == original || $2 == retry {found=1} END {exit !found}' <<< "$queue"; then
        echo "SKIP $old: a matching original/retry job is already queued or running."
        continue
    fi
    complete=1
    for seed in "$seed_start" "$seed_end"; do
        printf -v seed_dir 'seed_%03d' "$seed"
        folder="$output_root/$source_name/$model/random/pt$subset/$ssl/$seed_dir"
        if [[ ! -s $folder/encoder.pt || ! -s $folder/ssl_complete.json ]]; then complete=0; fi
    done
    if (( complete )); then
        echo "SKIP $old: both final encoder files and completion markers already exist."
        continue
    fi
    job_id=$(sbatch --parsable --job-name="$retry_name" --account="$account" \
        --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=6G --gres=gpu:l40s:1 \
        --time="$walltime" --exclude="$exclude_nodes" --chdir="$bench_dir" \
        --output="$output_root/logs/%x-%j.out" --error="$output_root/logs/%x-%j.err" \
        "$launcher" --worker "$bench_dir" "$source_name" "$model" "$subset" "$ssl" \
        "$venv" "$data_root" "$output_root" "$seed_start" "$epochs" 2)
    job_id=${job_id%%;*}
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s..%s\t%s\t%s\t%s\n' \
        "$old" "$job_id" "$account" "$source_name" "$model" "$subset" "$ssl" \
        "$seed_start" "$seed_end" "$epochs" "$walltime" "$exclude_nodes" >> "$receipt"
    printf 'Submitted retry %s for original %s\n' "$job_id" "$old"
    submitted=$((submitted + 1))
done
printf 'Submitted %s of 9 retries. Job mapping: %s\n' "$submitted" "$receipt"
