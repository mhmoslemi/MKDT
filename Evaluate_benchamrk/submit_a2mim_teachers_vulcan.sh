#!/bin/bash
set -euo pipefail
bench=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
run="$bench/a2mim_teacher"
venv=${BENCH_VENV:-/home/mmoslem3/ENV}
data=${BENCH_DATA_ROOT:-/scratch/mmoslem3/data}
[[ -r $venv/bin/activate ]] || { echo "Missing environment: $venv" >&2; exit 2; }
[[ $(scontrol show config | awk '$1=="ClusterName" {print $3}') == vulcan ]] || exit 2
inventory=$(sinfo -h -o '%G' --Node)
[[ $inventory == *gpu:l40s:* ]] || { echo 'L40S unavailable.' >&2; exit 2; }
accounts=$(sacctmgr -nP show assoc user=mmoslem3 format=Account)
for account in aip-boyuwang aip-yiweilu; do
    awk -F '|' -v account="$account" '$1==account {found=1} END {exit !found}' <<< "$accounts"
done
module spider python/3.11.5 >/dev/null 2>&1
module spider cuda/12.6 >/dev/null 2>&1
for marker in cifar-10-batches-py/data_batch_1 cifar-100-python/train tinyimagenet.pt; do
    [[ -r $data/$marker ]] || { echo "Missing dataset: $data/$marker" >&2; exit 2; }
done
mkdir -p "$run/logs"
exec 9>"$run/.submit.lock"
flock -n 9 || { echo 'Another copy is submitting.' >&2; exit 2; }
queued=$(squeue -h -u mmoslem3 -o '%100j|%i')
receipt="$run/submitted_$(date -u +%Y%m%dT%H%M%SZ)_$$.tsv"
printf 'dataset\tjob\taccount\twalltime\n' > "$receipt"
for dataset in CIFAR10 CIFAR100 TinyImageNet; do
    name="a2mim_${dataset}_ResNet18_s0_300ep"
    if [[ -s $run/checkpoints/$dataset/ResNet18/full/pt100/a2mim/seed_000/ssl_complete.json ]]; then
        echo "SKIP completed: $dataset"
        continue
    fi
    if awk -F '|' -v name="$name" '$1~("^"name" *$") {found=1} END {exit !found}' <<< "$queued"; then
        echo "SKIP already queued/running: $dataset"
        continue
    fi
    account=aip-boyuwang
    walltime=05:00:00
    if [[ $dataset == TinyImageNet ]]; then account=aip-yiweilu; walltime=23:00:00; fi
    job=$(sbatch --parsable --job-name="$name" --account="$account" \
        --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=6G --gres=gpu:l40s:1 \
        --time="$walltime" --signal=B:USR1@90 --chdir="$bench" \
        --output="$run/logs/%x-%j.out" --error="$run/logs/%x-%j.err" \
        "$run/worker.sh" "$bench" "$dataset" "$venv" "$data")
    job=${job%%;*}
    [[ $job =~ ^[0-9]+$ ]] || { echo "Unexpected job ID: $job" >&2; exit 2; }
    printf '%s\t%s\t%s\t%s\n' "$dataset" "$job" "$account" "$walltime" | tee -a "$receipt"
done
echo "Receipt: $receipt"
