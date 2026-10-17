#!/bin/bash
# Run with bash on Vulcan. Submits only the 14 audited unsuccessful jobs.
# 2 timeout retries each contain ONLY their unfinished seed; 12 startup retries
# retain their original seed grouping. Total: 20 missing encoders, no evaluation.
set -euo pipefail
[[ $# == 0 && -z ${SLURM_JOB_ID:-} ]] || {
    echo 'Run: bash Evaluate_benchamrk/retry_kmeans_failed_vulcan.sh (not sbatch)' >&2
    exit 2
}
bench_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
launcher="$bench_dir/kmeans_ssl.sh"
output_root="$bench_dir/kmeans_ssl_checkpoints_cifar10k8k6k_tiny6k5k4k_6seeds"
venv=${BENCH_VENV:-$HOME/ENV}
data_root=${BENCH_DATA_ROOT:-$bench_dir/../../data}
[[ -r $launcher && -r $venv/bin/activate && -d $data_root && -d $output_root ]] || {
    echo 'Missing existing worker, environment, dataset directory, or run directory.' >&2
    exit 2
}
venv=$(cd -- "$venv" && pwd -P)
data_root=$(cd -- "$data_root" && pwd -P)
cluster=$(scontrol show config | awk '$1 == "ClusterName" {print $3}')
[[ $cluster == vulcan ]] || { echo 'Vulcan only.' >&2; exit 2; }
inventory=$(sinfo -h -o '%G' --Node)
[[ $inventory == *gpu:l40s:* ]] || { echo 'L40S GPU type not found.' >&2; exit 2; }
accounts=$(sacctmgr -nP show assoc "user=$(id -un)" format=Account)
for account in aip-boyuwang aip-yiweilu; do
    awk -F '|' -v a="$account" '$1==a {found=1} END {exit !found}' <<< "$accounts" || {
        echo "Account unavailable: $account" >&2; exit 2;
    }
done

# Failed job, original job, account, source, model, %, SSL, first seed, count, epochs, walltime.
# Walltimes = maximum measured completed sibling seed time (training + selection),
# multiplied by seed count, plus 900 seconds; rounded UP to the next minute.
# Includes the slow rack05-06 timings for both timeout retries, not just faster siblings.
# Measured per-seed seconds: 7451,7297,4502,4352,1929,6130,6465,3156 respectively.
tasks=(
    '1814410 1814095 aip-yiweilu CIFAR10 VGG11 1 simclr 5 1 8000 02:20:00'
    '1814411 1814096 aip-yiweilu CIFAR10 VGG11 1 barlowtwins 1 1 8000 02:17:00'
    '1814453 1814153 aip-yiweilu CIFAR100 VGG11 2 simclr 0 2 6000 02:46:00'
    '1814456 1814156 aip-yiweilu CIFAR100 VGG11 2 barlowtwins 0 2 6000 02:41:00'
    '1814180 1814180 aip-boyuwang TinyImageNet ConvNet 0.2 barlowtwins 0 2 6000 01:20:00'
    '1814181 1814181 aip-boyuwang TinyImageNet ConvNet 0.2 barlowtwins 2 2 6000 01:20:00'
    '1814189 1814189 aip-boyuwang TinyImageNet ConvNet 1 barlowtwins 0 1 5000 01:58:00'
    '1814190 1814190 aip-boyuwang TinyImageNet ConvNet 1 barlowtwins 1 1 5000 01:58:00'
    '1814219 1814219 aip-boyuwang TinyImageNet VGG11 1 barlowtwins 0 1 5000 02:03:00'
    '1814220 1814220 aip-boyuwang TinyImageNet VGG11 1 barlowtwins 1 1 5000 02:03:00'
    '1814223 1814223 aip-boyuwang TinyImageNet VGG11 1 barlowtwins 4 1 5000 02:03:00'
    '1814224 1814224 aip-boyuwang TinyImageNet VGG11 1 barlowtwins 5 1 5000 02:03:00'
    '1814240 1814240 aip-boyuwang TinyImageNet ResNet18 0.2 barlowtwins 0 2 6000 02:01:00'
    '1814241 1814241 aip-boyuwang TinyImageNet ResNet18 0.2 barlowtwins 2 2 6000 02:01:00'
)
exec 9>"$output_root/.retry_failed_vulcan.lock"
flock -n 9 || { echo 'Another retry launcher is running.' >&2; exit 2; }
queue=$(squeue -h -u "$(id -un)" -o '%i|%j')
# Avoid overlapping writes from any existing k-means SSL launcher.
# Allow this script's own queued retries so rerunning can submit remaining tasks.
if awk -F '|' '$2 ~ /^kmeans_/ && $2 !~ /^kmeans_retry_failed_/ && $2 !~ /^kmeans_probe_/ {found=1} END {exit !found}' <<< "$queue"; then
    echo 'Other k-means jobs are active. Wait for them to finish before retrying.' >&2
    exit 2
fi
mkdir -p "$output_root/logs"
receipt="$output_root/retried_failed_vulcan_$(date -u +%Y%m%dT%H%M%SZ)_$$.tsv"
printf 'failed_job\tnew_job\taccount\tsource\tmodel\tsubset\tssl\tseed_start\tseed_count\tepochs\twalltime\n' > "$receipt"
submitted=0
for task in "${tasks[@]}"; do
    read -r failed original account source_name model subset ssl seed_start seed_count epochs walltime <<< "$task"
    name="kmeans_retry_failed_$failed"
    if awk -F '|' -v name="$name" '$2==name {found=1} END {exit !found}' <<< "$queue"; then
        echo "SKIP $failed: retry already queued/running."
        continue
    fi
    complete=1
    for ((seed=seed_start; seed<seed_start+seed_count; seed++)); do
        printf -v seed_dir 'seed_%03d' "$seed"
        folder="$output_root/$source_name/$model/kmeans/pt$subset/$ssl/$seed_dir"
        if [[ ! -s $folder/encoder.pt || ! -s $folder/ssl_complete.json ]]; then complete=0; fi
    done
    if (( complete )); then
        echo "SKIP $failed: requested encoders already saved."
        continue
    fi
    job=$(sbatch --parsable --job-name="$name" --account="$account" \
        --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=6G --gres=gpu:l40s:1 \
        --exclude=rack02-06 --time="$walltime" --chdir="$bench_dir" \
        --output="$output_root/logs/%x-%j.out" --error="$output_root/logs/%x-%j.err" \
        "$launcher" --worker "$bench_dir" "$source_name" "$model" "$subset" "$ssl" \
        "$venv" "$data_root" "$output_root" "$seed_start" "$epochs" "$seed_count")
    job=${job%%;*}
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$failed" "$job" "$account" "$source_name" "$model" "$subset" "$ssl" \
        "$seed_start" "$seed_count" "$epochs" "$walltime" >> "$receipt"
    printf 'SUBMITTED %s: %s/%s/%s%%/%s seeds=%s count=%s time=%s account=%s\n' \
        "$job" "$source_name" "$model" "$subset" "$ssl" "$seed_start" "$seed_count" "$walltime" "$account"
    submitted=$((submitted + 1))
done
printf 'DONE: submitted %s jobs (maximum 14). Receipt: %s\n' "$submitted" "$receipt"
