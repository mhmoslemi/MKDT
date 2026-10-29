#!/bin/bash
# FashionMNIST only: full SimCLR, Random/K-means, and original MKDT prerequisites.
set -euo pipefail

if [[ ${1:-} == --worker ]]; then
    shift
    [[ $# -ge 5 ]] || exit 2
    bench=$1 venv=$2 data=$3 worker_sha=$4 stage=$5
    shift 5
    : "${SLURM_JOB_ID:?Submit through Slurm}"
    module --force purge
    module load StdEnv/2023 python/3.11.5
    source "$venv/bin/activate"
    export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 WANDB_MODE=offline
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
    export XDG_CACHE_HOME="$bench/fashionmnist/cache" HF_HOME="$bench/fashionmnist/cache/hf"
    export TORCH_HOME="$bench/fashionmnist/cache/torch" TMPDIR="${SLURM_TMPDIR:?}"
    cd "$bench/fashionmnist"
    # Syntax checks stay on the allocated node, not the login node.
    python -c 'import ast,pathlib; [ast.parse(p.read_text(), filename=str(p)) for p in pathlib.Path(".").glob("*.py")]'
    exec python "$bench/fashionmnist/worker.py" "$stage" --data-root "$data" --worker-sha256 "$worker_sha" "$@"
fi

[[ $# == 0 && -z ${SLURM_JOB_ID:-} ]] || { echo 'Run with bash on the Vulcan login node.' >&2; exit 2; }
bench=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
launcher="$bench/submit_fashionmnist_vulcan.sh"
root="$bench/fashionmnist"
venv=${BENCH_VENV:-$HOME/ENV}
data=${BENCH_DATA_ROOT:-$bench/../../data}
[[ -r $venv/bin/activate && -d $data ]] || exit 2
venv=$(cd -- "$venv" && pwd -P)
data=$(cd -- "$data" && pwd -P)
[[ $(scontrol show config | awk '$1=="ClusterName"{print $3}') == vulcan ]] || exit 2
[[ $(sinfo -h -o '%G' --Node) == *gpu:l40s:* ]] || exit 2
module spider python/3.11.5 >/dev/null 2>&1
accounts=$(sacctmgr -nP show assoc "user=$(id -un)" format=Account)
for account in aip-boyuwang aip-yiweilu; do
    cut -d'|' -f1 <<< "$accounts" | grep -Fxq "$account" || exit 2
done
[[ -s $bench/digits_simclr/timing.json && -s $bench/digits_mkdt/timing.json ]] || exit 2
mkdir -p "$root/logs"
exec 9>"$root/.submit.lock"
flock -n 9 || { echo 'Another FashionMNIST submitter is active.' >&2; exit 2; }
queue=$(squeue -h -u "$(id -un)" -o '%i|%j')
worker_sha=$(sha256sum "$root/worker.py" | awk '{print $1}')
receipt="$root/submitted_$(date -u +%Y%m%dT%H%M%SZ)_$$.tsv"
printf 'job_id\tstage\taccount\twalltime\tdependencies\n' > "$receipt"
LAST_JOB='' index=0

submit() {
    local key=$1 stage=$2 seconds=$3 dependency=$4 marker=$5 account wall job
    shift 5
    account=aip-boyuwang
    if (( index % 2 )); then account=aip-yiweilu; fi
    index=$((index+1))
    if [[ -s $marker ]] && jq -e '.complete == true' "$marker" >/dev/null; then
        LAST_JOB=''; echo "SKIP completed fashion_$key"; return
    fi
    job=$(awk -F'|' -v name="fashion_$key" '$2==name {print $1}' <<< "$queue")
    if [[ -n $job ]]; then LAST_JOB=$job; echo "SKIP active fashion_$key ($job)"; return; fi
    [[ $seconds =~ ^[0-9]+$ && $seconds -ge 60 && $seconds -le 604800 ]] || exit 2
    printf -v wall '%d:%02d:%02d' "$((seconds/3600))" "$((seconds/60%60))" "$((seconds%60))"
    local command=(sbatch --parsable --job-name="fashion_$key" --account="$account"
        --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=6G --time="$wall" --chdir="$bench"
        --output="$root/logs/%x-%j.out" --error="$root/logs/%x-%j.err")
    [[ $stage == prepare || $stage == init ]] || command+=(--gres=gpu:l40s:1)
    [[ -z $dependency ]] || command+=(--dependency="afterok:$dependency" --kill-on-invalid-dep=yes)
    job=$("${command[@]}" "$launcher" --worker "$bench" "$venv" "$data" "$worker_sha" "$stage" "$@")
    job=${job%%;*}
    [[ $job =~ ^[0-9]+$ ]] || exit 2
    LAST_JOB=$job
    printf '%s\t%s\t%s\t%s\t%s\n' "$job" "$key" "$account" "$wall" "$dependency" >> "$receipt"
    printf 'SUBMITTED %s: fashion_%s, %s, %s, after=%s\n' "$job" "$key" "$account" "$wall" "${dependency:-none}"
}

data_dependency=''
missing=0
for name in train-images-idx3-ubyte train-labels-idx1-ubyte t10k-images-idx3-ubyte t10k-labels-idx1-ubyte; do
    [[ -s $data/FashionMNIST/raw/$name && -s $data/FashionMNIST/raw/$name.gz ]] || missing=1
done
if (( missing )); then
    data_dependency=$(awk -F'|' '$2=="fashion_data" {print $1}' <<< "$queue")
    if [[ -z $data_dependency ]]; then
        data_dependency=$(sbatch --parsable --job-name=fashion_data --account=aip-boyuwang \
            --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=1G --time=00:05:00 --chdir="$root" \
            --output="$root/logs/%x-%j.out" --error="$root/logs/%x-%j.err" "$root/stage_data.sh" "$data")
        data_dependency=${data_dependency%%;*}
        printf '%s\tdata\taip-boyuwang\t00:05:00\t\n' "$data_dependency" >> "$receipt"
    fi
fi
submit prepare prepare 180 "$data_dependency" "$root/ready.json"
prepared=$LAST_JOB
index=0

# Same image count, preprocessing and recipes as MNIST; reuse its measured timing.
probe_time=$(jq '.MNIST.probe_seconds' "$bench/digits_simclr/timing.json")
full_rate=$(jq '.MNIST.epoch_seconds' "$bench/digits_simclr/timing.json")
full_seconds=$(jq -n --argjson rate "$full_rate" --argjson probe "$probe_time" '1000*$rate+$probe+120+60|ceil')
for seed in 0 1 2; do
    printf -v seed_dir 'seed_%03d' "$seed"
    submit "full_s$seed" full "$full_seconds" "$prepared" \
        "$root/full/checkpoints/FashionMNIST/ResNet18/full/pt100/simclr/$seed_dir/complete.json" --seed "$seed"
done
for method in random kmeans; do
    for size in 0.2 1 2; do
        log="$bench/digits_subset_simclr2k/results/MNIST/ResNet18/$method/pt$size/simclr/ssl_seed_000/ssl.jsonl"
        [[ -s $log ]] || { echo "Missing matched MNIST timing: $log" >&2; exit 2; }
        rate=$(tail -n 200 "$log" | jq -s 'map(.seconds)|sort|.[(length*0.95|ceil)-1]')
        seconds=$(jq -n --argjson rate "$rate" --argjson probe "$probe_time" --arg method "$method" \
            '2000*$rate+5*$probe+20+60+(if $method=="kmeans" then 120 else 0 end)|ceil')
        submit "${method}_p${size/./p}" subset "$seconds" "$prepared" \
            "$root/subset/results/FashionMNIST/ResNet18/$method/pt$size/simclr/ssl_seed_000/complete.json" \
            --method "$method" --subset "$size"
    done
done
teacher_rate=$(jq '.MNIST.teacher_epoch_seconds' "$bench/digits_mkdt/timing.json")
teacher_time=$(jq -n --argjson rate "$teacher_rate" '1000*$rate+120+60|ceil')
submit teacher teacher "$teacher_time" "$prepared" "$root/mkdt/teachers/FashionMNIST/complete.json"
dependency=${LAST_JOB:-$prepared}
submit features features 180 "$dependency" "$root/mkdt/target_rep/barlow_twins/FashionMNIST_target_rep_train.json"
dependency=${LAST_JOB:-$prepared}
expert_time=$(jq '.MNIST.experts' "$bench/digits_mkdt/timing.json")
shards=()
for start in $(seq 0 5 95); do
    printf -v key '%03d' "$start"
    submit "experts_$key" experts "$expert_time" "$dependency" "$root/mkdt/work/FashionMNIST/shard_$key.json" --start "$start"
    [[ -z $LAST_JOB ]] || shards+=("$LAST_JOB")
done
if (( ${#shards[@]} )); then dependency=$(IFS=:; echo "${shards[*]}"); fi
submit init init 240 "$dependency" "$root/mkdt/work/FashionMNIST/init_complete.json"
echo "Receipt: $receipt"
