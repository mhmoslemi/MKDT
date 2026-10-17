#!/bin/bash
# Resubmit only the 69 Wang jobs in the user's pasted list, using Lu on Vulcan.
# Does NOT cancel anything. First cancel those exact originals, then run with bash.
set -euo pipefail

case ${1:---submit} in
    --submit) dry_run=0 ;;
    --dry-run) dry_run=1 ;;
    --help|-h)
        printf '%s\n' \
            'bash Evaluate_benchamrk/resubmit_selected_kmeans_lu.sh --dry-run' \
            'bash Evaluate_benchamrk/resubmit_selected_kmeans_lu.sh  # submit on Vulcan after cancellation' \
            'Only the 69 pasted original jobs / 133 seed runs; every replacement uses aip-yiweilu.' \
            'Keeps original datasets, networks, subsets, SSL methods, seeds, epochs and walltimes.' \
            'Keeps 1 L40S GPU, 1 CPU, 6G, shared selection cache and checkpoint directory.' \
            'Calls unchanged kmeans_ssl.sh --worker; resumes the latest rolling 500-epoch checkpoint.' \
            'Skips active originals/replacements and existing completed seed groups.' \
            'Submits only originals confirmed CANCELLED in accounting; rerun if cancellation is still settling.' \
            'No cancellation, arrays, dependencies, checkpoint deletion, or changes to other jobs.' \
            'BENCH_VENV and BENCH_DATA_ROOT may override the existing environment/data paths.'
        exit 0 ;;
    *) echo 'Use --submit (default), --dry-run, or --help.' >&2; exit 2 ;;
esac
[[ $# -le 1 ]] || { echo 'Unexpected extra arguments.' >&2; exit 2; }
[[ -z ${SLURM_JOB_ID:-} ]] || { echo 'Run with bash on the login node, not sbatch.' >&2; exit 2; }

# Original job, dataset, network, subset %, SSL, first seed, seed count, epochs, walltime.
# Frozen from submitted_20261017T051102Z_35858.tsv, filtered to the exact pasted IDs.
tasks=(
    '1814080 CIFAR10 ConvNet 2 simclr 0 2 6000 04:02:39'
    '1814081 CIFAR10 ConvNet 2 simclr 2 2 6000 04:02:39'
    '1814082 CIFAR10 ConvNet 2 simclr 4 2 6000 04:02:39'
    '1814083 CIFAR10 ConvNet 2 barlowtwins 0 2 6000 03:57:14'
    '1814084 CIFAR10 ConvNet 2 barlowtwins 2 2 6000 03:57:14'
    '1814085 CIFAR10 ConvNet 2 barlowtwins 4 2 6000 03:57:14'
    '1814093 CIFAR10 VGG11 1 simclr 0 2 8000 02:08:46'
    '1814094 CIFAR10 VGG11 1 simclr 2 2 8000 02:08:46'
    '1814095 CIFAR10 VGG11 1 simclr 4 2 8000 02:08:46'
    '1814096 CIFAR10 VGG11 1 barlowtwins 0 2 8000 02:02:13'
    '1814097 CIFAR10 VGG11 1 barlowtwins 2 2 8000 02:02:13'
    '1814099 CIFAR10 VGG11 2 simclr 0 2 6000 04:26:52'
    '1814100 CIFAR10 VGG11 2 simclr 2 2 6000 04:26:52'
    '1814101 CIFAR10 VGG11 2 simclr 4 2 6000 04:26:52'
    '1814102 CIFAR10 VGG11 2 barlowtwins 0 2 6000 04:10:07'
    '1814103 CIFAR10 VGG11 2 barlowtwins 2 2 6000 04:10:07'
    '1814104 CIFAR10 VGG11 2 barlowtwins 4 2 6000 04:10:07'
    '1814109 CIFAR10 ResNet18 0.2 barlowtwins 2 2 10000 01:18:53'
    '1814110 CIFAR10 ResNet18 0.2 barlowtwins 4 2 10000 01:18:53'
    '1814111 CIFAR10 ResNet18 1 simclr 0 2 8000 02:16:42'
    '1814112 CIFAR10 ResNet18 1 simclr 2 2 8000 02:16:42'
    '1814113 CIFAR10 ResNet18 1 simclr 4 2 8000 02:16:42'
    '1814114 CIFAR10 ResNet18 1 barlowtwins 0 2 8000 02:16:05'
    '1814115 CIFAR10 ResNet18 1 barlowtwins 2 2 8000 02:16:05'
    '1814116 CIFAR10 ResNet18 1 barlowtwins 4 2 8000 02:16:05'
    '1814117 CIFAR10 ResNet18 2 simclr 0 2 6000 04:47:11'
    '1814119 CIFAR10 ResNet18 2 simclr 4 2 6000 04:47:11'
    '1814120 CIFAR10 ResNet18 2 barlowtwins 0 2 6000 04:43:03'
    '1814121 CIFAR10 ResNet18 2 barlowtwins 2 2 6000 04:43:03'
    '1814122 CIFAR10 ResNet18 2 barlowtwins 4 2 6000 04:43:03'
    '1814124 CIFAR100 ConvNet 0.2 simclr 2 2 10000 01:05:24'
    '1814125 CIFAR100 ConvNet 0.2 simclr 4 2 10000 01:05:24'
    '1814126 CIFAR100 ConvNet 0.2 barlowtwins 0 2 10000 01:10:09'
    '1814127 CIFAR100 ConvNet 0.2 barlowtwins 2 2 10000 01:10:09'
    '1814129 CIFAR100 ConvNet 1 simclr 0 2 8000 01:59:06'
    '1814130 CIFAR100 ConvNet 1 simclr 2 2 8000 01:59:06'
    '1814131 CIFAR100 ConvNet 1 simclr 4 2 8000 01:59:06'
    '1814132 CIFAR100 ConvNet 1 barlowtwins 0 2 8000 01:56:06'
    '1814133 CIFAR100 ConvNet 1 barlowtwins 2 2 8000 01:56:06'
    '1814134 CIFAR100 ConvNet 1 barlowtwins 4 2 8000 01:56:06'
    '1814135 CIFAR100 ConvNet 2 simclr 0 2 6000 04:01:45'
    '1814136 CIFAR100 ConvNet 2 simclr 2 2 6000 04:01:45'
    '1814137 CIFAR100 ConvNet 2 simclr 4 2 6000 04:01:45'
    '1814139 CIFAR100 ConvNet 2 barlowtwins 2 2 6000 03:57:37'
    '1814140 CIFAR100 ConvNet 2 barlowtwins 4 2 6000 03:57:37'
    '1814141 CIFAR100 VGG11 0.2 simclr 0 2 10000 01:11:44'
    '1814142 CIFAR100 VGG11 0.2 simclr 2 2 10000 01:11:44'
    '1814143 CIFAR100 VGG11 0.2 simclr 4 2 10000 01:11:44'
    '1814144 CIFAR100 VGG11 0.2 barlowtwins 0 2 10000 01:12:29'
    '1814145 CIFAR100 VGG11 0.2 barlowtwins 2 2 10000 01:12:29'
    '1814146 CIFAR100 VGG11 0.2 barlowtwins 4 2 10000 01:12:29'
    '1814153 CIFAR100 VGG11 2 simclr 0 2 6000 04:16:34'
    '1814154 CIFAR100 VGG11 2 simclr 2 2 6000 04:16:34'
    '1814155 CIFAR100 VGG11 2 simclr 4 2 6000 04:16:34'
    '1814156 CIFAR100 VGG11 2 barlowtwins 0 2 6000 04:08:12'
    '1814157 CIFAR100 VGG11 2 barlowtwins 2 2 6000 04:08:12'
    '1814159 CIFAR100 ResNet18 0.2 simclr 0 2 10000 01:18:47'
    '1814160 CIFAR100 ResNet18 0.2 simclr 2 2 10000 01:18:47'
    '1814171 CIFAR100 ResNet18 2 simclr 0 2 6000 04:47:32'
    '1814172 CIFAR100 ResNet18 2 simclr 2 2 6000 04:47:32'
    '1814173 CIFAR100 ResNet18 2 simclr 4 2 6000 04:47:32'
    '1814174 CIFAR100 ResNet18 2 barlowtwins 0 2 6000 04:41:28'
    '1814175 CIFAR100 ResNet18 2 barlowtwins 2 2 6000 04:41:28'
    '1814176 CIFAR100 ResNet18 2 barlowtwins 4 2 6000 04:41:28'
    '1814213 TinyImageNet VGG11 1 simclr 0 1 5000 03:24:01'
    '1814214 TinyImageNet VGG11 1 simclr 1 1 5000 03:24:01'
    '1814215 TinyImageNet VGG11 1 simclr 2 1 5000 03:24:01'
    '1814216 TinyImageNet VGG11 1 simclr 3 1 5000 03:24:01'
    '1814218 TinyImageNet VGG11 1 simclr 5 1 5000 03:24:01'
)
account=aip-yiweilu
printf 'original_job\tdataset\tmodel\tsubset_percent\tssl\tseed_start\tseed_count\tepochs_per_seed\taccount\twalltime\n'
ids=()
for task in "${tasks[@]}"; do
    read -r old source_name model subset ssl seed_start seed_count epochs walltime <<< "$task"
    ids+=("$old")
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$old" "$source_name" "$model" "$subset" "$ssl" "$seed_start" "$seed_count" \
        "$epochs" "$account" "$walltime"
done
if (( dry_run )); then
    echo 'Dry run: 69 selected jobs / 133 seed runs, all Lu. No scheduler calls or files written.'
    exit 0
fi

cluster=$(scontrol show config | awk '$1 == "ClusterName" {print $3}')
[[ $cluster == vulcan ]] || { echo 'Submit this script on Vulcan only.' >&2; exit 2; }
bench_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
launcher="$bench_dir/kmeans_ssl.sh"
output_root="$bench_dir/kmeans_ssl_checkpoints_cifar10k8k6k_tiny6k5k4k_6seeds"
venv=${BENCH_VENV:-$HOME/ENV}
data_root=${BENCH_DATA_ROOT:-$bench_dir/../../data}
for required in kmeans_ssl.sh common.py data.py train_ssl.py run.py evaluate.py protocol.json \
                ../networks.py ../benchmarks/downstream_data.py; do
    [[ -r $bench_dir/$required ]] || { echo "Missing $bench_dir/$required" >&2; exit 2; }
done
[[ -r $venv/bin/activate && -d $data_root ]] || {
    echo 'Set BENCH_VENV and BENCH_DATA_ROOT to the existing Vulcan paths.' >&2; exit 2;
}
venv=$(cd -- "$venv" && pwd -P)
data_root=$(cd -- "$data_root" && pwd -P)
for marker in cifar-10-batches-py/data_batch_{1..5} cifar-10-batches-py/batches.meta \
              cifar-100-python/train cifar-100-python/meta tinyimagenet.pt; do
    [[ -r $data_root/$marker ]] || { echo "Missing training data: $data_root/$marker" >&2; exit 2; }
done
inventory=$(sinfo -h -o '%G' --Node)
gpu_types=$(printf '%s\n' "$inventory" | tr ',' '\n' | sed -nE 's/^gpu:([^:]+):.*/\1/p' | sort -u)
grep -Fxq l40s <<< "$gpu_types" || { echo 'L40S GPUs are unavailable here.' >&2; exit 2; }
accounts=$(sacctmgr -nP show assoc "user=$(id -un)" format=Account)
printf '%s\n' "$accounts" | cut -d'|' -f1 | grep -Fxq "$account" || {
    echo "Account not available: $account" >&2; exit 2;
}
id_list=$(IFS=,; echo "${ids[*]}")
records=$(sacct -X -nP -u "$(id -un)" -j "$id_list" --starttime=2026-10-16 \
    --format=JobIDRaw,Account,State%40)
declare -A states=() old_accounts=()
while IFS='|' read -r job old_account state rest; do
    [[ $job =~ ^[0-9]+$ ]] || continue
    states[$job]=$state
    old_accounts[$job]=$old_account
done <<< "$records"
for old in "${ids[@]}"; do
    [[ ${old_accounts[$old]:-} == aip-boyuwang ]] || {
        echo "Original $old is not confirmed as your Wang job in Vulcan accounting; refusing submission." >&2
        exit 2
    }
done

mkdir -p "$output_root/logs"
exec 9>"$output_root/.selected_lu_resubmit.lock"
flock -n 9 || { echo 'Another copy of this Lu resubmission launcher is running.' >&2; exit 2; }
queue=$(squeue -h -u "$(id -un)" -o '%i|%j')
receipt="$output_root/resubmitted_lu_$(date -u +%Y%m%dT%H%M%SZ)_$$.tsv"
printf 'original_job\tnew_job\taccount\tdataset\tmodel\tsubset_percent\tssl\tseed_start\tseed_count\tepochs_per_seed\twalltime\n' > "$receipt"
submitted=0
for task in "${tasks[@]}"; do
    read -r old source_name model subset ssl seed_start seed_count epochs walltime <<< "$task"
    seed_end=$((seed_start + seed_count - 1))
    seed_tag=$seed_start
    if (( seed_count == 2 )); then seed_tag="$seed_start-$seed_end"; fi
    original_name="kmeans_${source_name}_${model}_${subset/./p}_${ssl}_e${epochs}_s${seed_tag}"
    retry_name="kmeans_lu_${old}"
    if awk -F '|' -v old="$old" -v original="$original_name" -v retry="$retry_name" \
        '$1 == old || $2 == original || $2 == retry {found=1} END {exit !found}' <<< "$queue"; then
        echo "SKIP $old: matching original/replacement is still queued or running."
        continue
    fi
    complete=1
    for ((seed=seed_start; seed<=seed_end; seed++)); do
        printf -v seed_dir 'seed_%03d' "$seed"
        folder="$output_root/$source_name/$model/kmeans/pt$subset/$ssl/$seed_dir"
        if [[ ! -s $folder/encoder.pt || ! -s $folder/ssl_complete.json ]]; then complete=0; fi
    done
    if (( complete )); then
        echo "SKIP $old: all final encoders and completion markers already exist."
        continue
    fi
    if [[ ${states[$old]:-} != CANCELLED* ]]; then
        echo "SKIP $old: accounting state '${states[$old]:-unknown}' is not CANCELLED; rerun after cancellation settles."
        continue
    fi
    job_id=$(sbatch --parsable --job-name="$retry_name" --account="$account" \
        --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=6G --gres=gpu:l40s:1 \
        --time="$walltime" --chdir="$bench_dir" \
        --output="$output_root/logs/%x-%j.out" --error="$output_root/logs/%x-%j.err" \
        "$launcher" --worker "$bench_dir" "$source_name" "$model" "$subset" "$ssl" \
        "$venv" "$data_root" "$output_root" "$seed_start" "$epochs" "$seed_count")
    job_id=${job_id%%;*}
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$old" "$job_id" "$account" "$source_name" "$model" "$subset" "$ssl" \
        "$seed_start" "$seed_count" "$epochs" "$walltime" >> "$receipt"
    printf 'Submitted Lu job %s for original Wang job %s\n' "$job_id" "$old"
    submitted=$((submitted + 1))
done
printf 'Submitted %s of 69 selected jobs on Lu. Job mapping: %s\n' "$submitted" "$receipt"
