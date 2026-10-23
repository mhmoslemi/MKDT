#!/bin/bash
# Remaining RANDOM SSL only, on Killarney, all Wang. Run with bash, not sbatch.
# Audit: 33 cancelled pending + 16 CUDA failures + 3 missing-log jobs = 52 jobs / 76 seeds.
# Explicitly EXCLUDED at the user's request: timeout jobs 6341234,6341239,6341277,
# 6341333,6341334,6341336,6341390. Their files/recorded epochs are not modified.
# No MKDT, evaluation, cancellation, deletion, or training-code changes.
set -euo pipefail
[[ $# == 0 && -z ${SLURM_JOB_ID:-} ]] || {
    echo 'Run: bash Evaluate_benchamrk/submit_remaining_random_wang_killarney.sh' >&2
    exit 2
}
bench_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
launcher="$bench_dir/random_ssl.sh"
output_root="$bench_dir/random_ssl_checkpoints_cifar10k8k6k_tiny6k5k4k"
account=aip-boyuwang
venv=${BENCH_VENV:-$HOME/ENV}
data_root=${BENCH_DATA_ROOT:-$bench_dir/../../data}
[[ -r $launcher && -r $venv/bin/activate && -d $data_root && -d $output_root ]] || {
    echo 'Missing existing worker, environment, data directory, or random output directory.' >&2
    exit 2
}
venv=$(cd -- "$venv" && pwd -P)
data_root=$(cd -- "$data_root" && pwd -P)
cluster=$(scontrol show config | awk '$1 == "ClusterName" {print $3}')
[[ $cluster == killarney ]] || { echo 'Submit on Killarney only.' >&2; exit 2; }
inventory=$(sinfo -h -o '%G' --Node)
[[ $inventory == *gpu:l40s:* ]] || { echo 'L40S GPU type not found.' >&2; exit 2; }
accounts=$(sacctmgr -nP show assoc "user=$(id -un)" format=Account)
awk -F '|' -v a="$account" '$1==a {found=1} END {exit !found}' <<< "$accounts" || {
    echo "Account unavailable: $account" >&2; exit 2;
}
for marker in cifar-10-batches-py/data_batch_{1..5} cifar-10-batches-py/batches.meta \
    cifar-100-python/train cifar-100-python/meta tinyimagenet.pt; do
    [[ -r $data_root/$marker ]] || { echo "Missing training data: $data_root/$marker" >&2; exit 2; }
done

# Original ID, dataset, model, subset %, SSL, first seed, seed count, epochs, walltime.
# Frozen from submitted_20261017T050226Z_3146085.tsv; original walltimes unchanged.
tasks=(
    '6341183 CIFAR10 ConvNet 2 simclr 8 2 6000 04:02:33'
    '6341184 CIFAR10 ConvNet 2 barlowtwins 0 2 6000 03:57:08'
    '6341185 CIFAR10 ConvNet 2 barlowtwins 2 2 6000 03:57:08'
    '6341186 CIFAR10 ConvNet 2 barlowtwins 4 2 6000 03:57:08'
    '6341188 CIFAR10 ConvNet 2 barlowtwins 8 2 6000 03:57:08'
    '6341213 CIFAR10 VGG11 2 simclr 8 2 6000 04:26:46'
    '6341218 CIFAR10 VGG11 2 barlowtwins 8 2 6000 04:10:01'
    '6341244 CIFAR10 ResNet18 2 simclr 8 2 6000 04:47:05'
    '6341249 CIFAR10 ResNet18 2 barlowtwins 8 2 6000 04:42:57'
    '6341274 CIFAR100 ConvNet 2 simclr 8 2 6000 04:01:37'
    '6341305 CIFAR100 VGG11 2 simclr 8 2 6000 04:16:26'
    '6341310 CIFAR100 VGG11 2 barlowtwins 8 2 6000 04:08:04'
    '6341326 CIFAR100 ResNet18 1 barlowtwins 0 2 8000 02:16:34'
    '6341327 CIFAR100 ResNet18 1 barlowtwins 2 2 8000 02:16:34'
    '6341328 CIFAR100 ResNet18 1 barlowtwins 4 2 8000 02:16:34'
    '6341329 CIFAR100 ResNet18 1 barlowtwins 6 2 8000 02:16:34'
    '6341330 CIFAR100 ResNet18 1 barlowtwins 8 2 8000 02:16:34'
    '6341335 CIFAR100 ResNet18 2 simclr 8 2 6000 04:47:24'
    '6341340 CIFAR100 ResNet18 2 barlowtwins 8 2 6000 04:41:20'
    '6341351 TinyImageNet ConvNet 1 simclr 0 1 5000 02:56:59'
    '6341355 TinyImageNet ConvNet 1 simclr 4 1 5000 02:56:59'
    '6341360 TinyImageNet ConvNet 1 simclr 9 1 5000 02:56:59'
    '6341364 TinyImageNet ConvNet 1 barlowtwins 3 1 5000 02:46:36'
    '6341365 TinyImageNet ConvNet 1 barlowtwins 4 1 5000 02:46:36'
    '6341370 TinyImageNet ConvNet 1 barlowtwins 8 1 5000 02:46:36'
    '6341371 TinyImageNet ConvNet 1 barlowtwins 9 1 5000 02:46:36'
    '6341376 TinyImageNet ConvNet 2 simclr 4 1 4000 05:14:15'
    '6341381 TinyImageNet ConvNet 2 simclr 9 1 4000 05:14:15'
    '6341386 TinyImageNet ConvNet 2 barlowtwins 4 1 4000 04:55:36'
    '6341391 TinyImageNet ConvNet 2 barlowtwins 9 1 4000 04:55:36'
    '6341398 TinyImageNet VGG11 0.2 barlowtwins 2 2 6000 01:55:38'
    '6341407 TinyImageNet VGG11 1 simclr 4 1 5000 03:23:52'
    '6341412 TinyImageNet VGG11 1 simclr 9 1 5000 03:23:52'
    '6341417 TinyImageNet VGG11 1 barlowtwins 4 1 5000 02:53:41'
    '6341421 TinyImageNet VGG11 1 barlowtwins 8 1 5000 02:53:41'
    '6341422 TinyImageNet VGG11 1 barlowtwins 9 1 5000 02:53:41'
    '6341427 TinyImageNet VGG11 2 simclr 4 1 4000 05:26:34'
    '6341432 TinyImageNet VGG11 2 simclr 9 1 4000 05:26:34'
    '6341438 TinyImageNet VGG11 2 barlowtwins 4 1 4000 05:10:58'
    '6341443 TinyImageNet VGG11 2 barlowtwins 9 1 4000 05:10:58'
    '6341447 TinyImageNet ResNet18 0.2 simclr 6 2 6000 02:38:10'
    '6341448 TinyImageNet ResNet18 0.2 simclr 8 2 6000 02:38:10'
    '6341450 TinyImageNet ResNet18 0.2 barlowtwins 2 2 6000 02:33:15'
    '6341453 TinyImageNet ResNet18 0.2 barlowtwins 8 2 6000 02:33:15'
    '6341458 TinyImageNet ResNet18 1 simclr 4 1 5000 04:02:06'
    '6341464 TinyImageNet ResNet18 1 simclr 9 1 5000 04:02:06'
    '6341469 TinyImageNet ResNet18 1 barlowtwins 4 1 5000 03:51:32'
    '6341474 TinyImageNet ResNet18 1 barlowtwins 9 1 5000 03:51:32'
    '6341483 TinyImageNet ResNet18 2 simclr 4 1 4000 07:17:40'
    '6341488 TinyImageNet ResNet18 2 simclr 9 1 4000 07:17:40'
    '6341493 TinyImageNet ResNet18 2 barlowtwins 4 1 4000 06:56:49'
    '6341498 TinyImageNet ResNet18 2 barlowtwins 9 1 4000 06:56:49'
)
ids=()
for task in "${tasks[@]}"; do read -r old rest <<< "$task"; ids+=("$old"); done
id_list=$(IFS=,; echo "${ids[*]}")
records=$(sacct -X -nP -u "$(id -un)" -j "$id_list" --starttime=2026-10-16 \
    --format=JobIDRaw,State%40,NodeList)
declare -A states=() hosts=() excluded=()
while IFS='|' read -r job state nodes rest; do
    [[ $job =~ ^[0-9]+$ ]] || continue
    states[$job]=$state
    hosts[$job]=$nodes
done <<< "$records"
# Resolve the actual hosts for the 16 audited CUDA failures, rather than guess names.
for old in 6341184 6341185 6341186 6341326 6341327 6341328 6341329 6341330 \
    6341351 6341364 6341370 6341398 6341421 6341447 6341450 6341453; do
    node=${hosts[$old]:-}
    [[ $node =~ ^[[:alnum:]][[:alnum:]_-]*$ && $node != None && $node != Unknown ]] || {
        echo "Cannot resolve the CUDA-failure node for $old; nothing submitted." >&2; exit 2;
    }
    excluded[$node]=1
done
exclude_nodes=$(printf '%s\n' "${!excluded[@]}" | sort | paste -sd, -)
printf 'All Wang; maximum 52 jobs / 76 seeds. Excluded GPU-failure nodes: %s\n' "$exclude_nodes"
exec 9>"$output_root/.remaining_random_wang.lock"
flock -n 9 || { echo 'Another copy of this launcher is running.' >&2; exit 2; }
queue=$(squeue -h -u "$(id -un)" -o '%i|%j')
mkdir -p "$output_root/logs"
receipt="$output_root/resubmitted_remaining_wang_$(date -u +%Y%m%dT%H%M%SZ)_$$.tsv"
printf 'original_job\tnew_job\taccount\tdataset\tmodel\tsubset\tssl\tseed_start\tseed_count\tepochs\twalltime\texcluded_nodes\n' > "$receipt"
submitted=0
for task in "${tasks[@]}"; do
    read -r old source_name model subset ssl start count epochs walltime <<< "$task"
    end=$((start + count - 1))
    tag=$start
    if (( count == 2 )); then tag="$start-$end"; fi
    original_name="random_${source_name}_${model}_${subset/./p}_${ssl}_e${epochs}_s${tag}"
    name="random_remaining_wang_$old"
    if awk -F '|' -v old="$old" -v original="$original_name" -v name="$name" -v retry="random_retry_$old" \
        '$1==old || $2==original || $2==name || $2==retry {found=1} END {exit !found}' <<< "$queue"; then
        echo "SKIP $old: matching job still queued/running."
        continue
    fi
    complete=1
    for ((seed=start; seed<=end; seed++)); do
        printf -v seed_dir 'seed_%03d' "$seed"
        folder="$output_root/$source_name/$model/random/pt$subset/$ssl/$seed_dir"
        if [[ ! -s $folder/encoder.pt || ! -s $folder/ssl_complete.json ]]; then complete=0; fi
    done
    if (( complete )); then echo "SKIP $old: encoders already complete."; continue; fi
    case ${states[$old]:-unknown} in
        CANCELLED*|FAILED|COMPLETED|TIMEOUT|NODE_FAIL|OUT_OF_MEMORY|PREEMPTED|BOOT_FAIL|DEADLINE) ;;
        *) echo "SKIP $old: original state '${states[$old]:-unknown}' is not confirmed terminal; rerun after it settles."; continue ;;
    esac
    job=$(sbatch --parsable --job-name="$name" --account="$account" \
        --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=6G --gres=gpu:l40s:1 \
        --time="$walltime" --exclude="$exclude_nodes" --chdir="$bench_dir" \
        --output="$output_root/logs/%x-%j.out" --error="$output_root/logs/%x-%j.err" \
        "$launcher" --worker "$bench_dir" "$source_name" "$model" "$subset" "$ssl" \
        "$venv" "$data_root" "$output_root" "$start" "$epochs" "$count")
    job=${job%%;*}
    [[ $job =~ ^[0-9]+$ ]] || { echo "Unexpected sbatch response: $job" >&2; exit 2; }
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$old" "$job" "$account" "$source_name" "$model" "$subset" "$ssl" \
        "$start" "$count" "$epochs" "$walltime" "$exclude_nodes" >> "$receipt"
    printf 'SUBMITTED %s: %s/%s/%s%%/%s seeds=%s time=%s account=%s\n' \
        "$job" "$source_name" "$model" "$subset" "$ssl" "$tag" "$walltime" "$account"
    submitted=$((submitted + 1))
done
printf 'DONE: submitted %s of 52 remaining random jobs. Receipt: %s\n' "$submitted" "$receipt"
echo 'The seven accepted timeout cases were not submitted or modified.'
