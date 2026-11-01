#!/bin/bash
set -euo pipefail
[[ $# == 0 ]] || { echo "Usage: bash $0" >&2; exit 2; }
bench=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
root=$(dirname -- "$bench")
run="$bench/mipa_components"
[[ $(scontrol show config | awk '$1=="ClusterName" {print $3}') == killarney ]] || { echo 'Run this submitter on Killarney.' >&2; exit 2; }
inventory=$(sinfo -h -o '%G' --Node | sort -u)
[[ $inventory == *gpu:l40s:* ]] || { echo 'L40S GPUs were not found on Killarney.' >&2; exit 2; }
accounts=$(sacctmgr -nP show assoc "user=$(id -un)" format=Account)
for account in aip-yiweilu aip-boyuwang; do
    awk -F '|' -v account="$account" '$1==account {found=1} END {exit !found}' <<< "$accounts" || { echo "Account unavailable: $account" >&2; exit 2; }
done
module spider python/3.11.5 >/dev/null 2>&1
for file in "$run/job.sh" "$run/worker.py" /home/mmoslem3/ENV/bin/activate \
    "$bench/mipa_cvpr_color_crop/manifest.json" \
    "$bench/mipa_cvpr_color_crop/snapshot/method1_cvpr.py" \
    "$bench/mipa_cvpr_color_crop/snapshot/worker.py" \
    "$bench/mipa_cvpr_color_crop/results/CIFAR10/ResNet18BN/pt1/simclr/ssl_seed_000/config.json" \
    "$bench/full_ssl_checkpoints_1000ep_3seeds/CIFAR10/ResNet18/full/pt100/simclr/seed_000/encoder.pt" \
    "$root/OT-SSL-DD/ours_syntehtic_cvpr_color_crop/CIFAR10/ResNet18BN/pt1/res_IIC_CIFAR10_ResNet18BN_1percent_final.pt" \
    /scratch/mmoslem3/data/cifar-10-batches-py/{data_batch_1,data_batch_2,data_batch_3,data_batch_4,data_batch_5,test_batch,batches.meta}; do
    [[ -s $file ]] || { echo "Sync the required input first: $file" >&2; exit 2; }
done
logs="$run/logs/killarney"
mkdir -p "$logs"
exec 9>"$run/.submit_killarney.lock"
flock -n 9 || { echo 'Another component submitter is running.' >&2; exit 2; }
queued=$(squeue -h -u "$(id -un)" -o '%j|%i')
for account in aip-yiweilu aip-boyuwang; do
    tag=lu
    suffix=
    [[ $account != aip-boyuwang ]] || { tag=wang; suffix=_wang; }
    for variant in independent_agree shared_agree; do
        output="$run/runs/pt1/${variant}${suffix}"
        name="mipac1_${variant}_${tag}"
        active=$(awk -F '|' -v name="$name" '$1==name {print $2}' <<< "$queued")
        if [[ -n $active ]]; then
            printf 'Already queued: %s (%s)\n' "$name" "$active"
            continue
        fi
        if [[ -s "$output/evaluation/complete.json" ]]; then
            printf 'Already completed: %s\n' "$output"
            continue
        fi
        job=$(sbatch --parsable --job-name="$name" --account="$account" \
            --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=6G --gres=gpu:l40s:1 \
            --time=00:40:00 --chdir="$root" --output="$logs/%x-%j.out" --error="$logs/%x-%j.err" \
            "$run/job.sh" "$variant" 1)
        job=${job%%;*}
        [[ $job =~ ^[0-9]+$ ]] || { echo "Invalid submitted job ID: $job" >&2; exit 2; }
        printf '%s\t%s\t%s\t00:40:00\t%s\n' "$job" "$account" "$variant" "$output" >> "$run/submitted_killarney.tsv"
        printf 'Submitted %s: %s (%s), CIFAR10 1%%, 40 minutes\nLog: %s/%s-%s.out\n' "$job" "$variant" "$tag" "$logs" "$name" "$job"
    done
done
