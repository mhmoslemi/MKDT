#!/bin/bash
# 12 ordinary jobs; each trains one encoder and then its ten main-table probes.
set -euo pipefail
if [[ ${1:-} == --worker ]]; then
    shift
    [[ $# == 7 ]] || exit 2
    bench=$1 venv=$2 data=$3 source_name=$4 method=$5 subset=$6 worker_sha=$7
    : "${SLURM_JOB_ID:?Submit through Slurm}"
    module --force purge
    module load StdEnv/2023 python/3.11.5
    source "$venv/bin/activate"
    export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
    export HF_HOME="$bench/digits_subset_simclr2k/cache/hf" XDG_CACHE_HOME="$bench/digits_subset_simclr2k/cache"
    export TORCH_HOME="$bench/digits_subset_simclr2k/cache/torch" TMPDIR="${SLURM_TMPDIR:?}"
    cd "$bench/digits_subset_simclr2k"
    exec python "$bench/digits_subset_simclr2k/worker.py" --source "$source_name" --method "$method" \
        --subset "$subset" --data-root "$data" --worker-sha256 "$worker_sha"
fi
[[ $# == 0 && -z ${SLURM_JOB_ID:-} ]] || { echo 'Run with bash on the Vulcan login node.' >&2; exit 2; }
bench=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
root="$bench/digits_subset_simclr2k"
launcher="$bench/submit_digits_random_kmeans_simclr2k_vulcan.sh"
venv=${BENCH_VENV:-$HOME/ENV}
data=${BENCH_DATA_ROOT:-$bench/../../data}
[[ -r $venv/bin/activate && -d $data && -s $bench/digits_simclr/prepared.json ]] || exit 2
venv=$(cd -- "$venv" && pwd -P)
data=$(cd -- "$data" && pwd -P)
[[ $(scontrol show config | awk '$1=="ClusterName"{print $3}') == vulcan ]] || exit 2
[[ $(sinfo -h -o '%G' --Node) == *gpu:l40s:* ]] || exit 2
module spider python/3.11.5 >/dev/null 2>&1
accounts=$(sacctmgr -nP show assoc "user=$(id -un)" format=Account)
for account in aip-boyuwang aip-yiweilu; do
    cut -d'|' -f1 <<< "$accounts" | grep -Fxq "$account" || exit 2
done
small_log="$bench/cifar_small_simclr2k_1ssl5probes/results/CIFAR10/ResNet18/random/pt0.2/simclr/ssl_seed_000/ssl.jsonl"
batch_log="$bench/cifar_small_simclr2k_1ssl5probes/results/CIFAR10/ResNet18/random/pt1/simclr/ssl_seed_000/ssl.jsonl"
[[ -s $small_log && -s $batch_log && -s $bench/digits_simclr/timing.json ]] || exit 2
small_rate=$(tail -n 200 "$small_log" | jq -s 'map(.seconds)|sort|.[(length*0.95|ceil)-1]')
batch_rate=$(tail -n 200 "$batch_log" | jq -s 'map(.seconds)|sort|.[(length*0.95|ceil)-1]')
worker_sha=$(sha256sum "$root/worker.py" | awk '{print $1}')
mkdir -p "$root/logs"
exec 9>"$root/.submit.lock"
flock -n 9 || { echo 'A digit subset submitter is already active.' >&2; exit 2; }
queue=$(squeue -h -u "$(id -un)" -o '%i|%j')
receipt="$root/submitted_$(date -u +%Y%m%dT%H%M%SZ)_$$.tsv"
printf 'job_id\tdataset\tmethod\tsubset\timages\taccount\twalltime\n' > "$receipt"
for dataset in SVHN MNIST; do
    count=$(jq -r --arg s "$dataset" '.datasets[$s].train_count' "$bench/digits_simclr/prepared.json")
    probe_time=$(jq -r --arg s "$dataset" '.[$s].probe_seconds' "$bench/digits_simclr/timing.json")
    account=aip-boyuwang
    [[ $dataset != MNIST ]] || account=aip-yiweilu
    for method in random kmeans; do
        for subset in 0.2 1 2; do
            name="dsub_${dataset}_${method}_p${subset/./p}"
            if grep -q "|$name$" <<< "$queue"; then echo "SKIP active $name"; continue; fi
            output="$root/results/$dataset/ResNet18/$method/pt$subset/simclr/ssl_seed_000"
            if [[ -s $output/complete.json ]]; then echo "SKIP completed $name"; continue; fi
            # Measured 2k CIFAR rates; digit probe times are already measured on these splits.
            read -r images seconds < <(jq -n -r --argjson n "$count" --argjson p "$subset" \
                --argjson small "$small_rate" --argjson batch "$batch_rate" --argjson probe "$probe_time" \
                --arg method "$method" '
                ($n*$p/100|floor) as $count |
                (if $count<256 then $small*$count/100 else ($count/256|floor)*$batch end) as $epoch |
                [$count, (2000*$epoch+5*$probe+20+60+(if $method=="kmeans" then 120 else 0 end)|ceil)] | @tsv')
            [[ $seconds =~ ^[0-9]+$ && $seconds -gt 60 && $seconds -le 604800 ]] || exit 2
            printf -v wall '%d:%02d:%02d' "$((seconds/3600))" "$((seconds/60%60))" "$((seconds%60))"
            job=$(sbatch --parsable --job-name="$name" --account="$account" \
                --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=6G --gres=gpu:l40s:1 --time="$wall" \
                --chdir="$bench" --output="$root/logs/%x-%j.out" --error="$root/logs/%x-%j.err" \
                "$launcher" --worker "$bench" "$venv" "$data" "$dataset" "$method" "$subset" "$worker_sha")
            job=${job%%;*}
            [[ $job =~ ^[0-9]+$ ]] || exit 2
            printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$job" "$dataset" "$method" "$subset" "$images" "$account" "$wall" >> "$receipt"
            printf 'SUBMITTED %s: %s, %s images, %s, %s\n' "$job" "$name" "$images" "$account" "$wall"
        done
    done
done
echo "Receipt: $receipt"
