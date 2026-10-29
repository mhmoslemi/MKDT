#!/bin/bash
# First stage only: 3 full-data seeds/dataset, each followed by the two main probes.
set -euo pipefail

if [[ ${1:-} == --worker ]]; then
    shift
    [[ $# -ge 4 ]] || exit 2
    bench=$1 venv=$2 data=$3 stage=$4
    shift 4
    : "${SLURM_JOB_ID:?Submit through Slurm}"
    module --force purge
    module load StdEnv/2023 python/3.11.5
    source "$venv/bin/activate"
    export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
    export XDG_CACHE_HOME="$bench/digits_simclr/cache" HF_HOME="$bench/digits_simclr/cache/hf"
    export TORCH_HOME="$bench/digits_simclr/cache/torch" TMPDIR="${SLURM_TMPDIR:?}"
    cd "$bench/digits_simclr"
    exec python "$bench/digits_simclr/worker.py" "$stage" --data-root "$data" "$@"
fi

bench=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
launcher="$bench/submit_digits_full_simclr_vulcan.sh"
root="$bench/digits_simclr"
stage=${1:---prepare}
venv=${2:-$HOME/ENV}
data=${3:-$bench/../../data}
[[ $stage == --prepare || $stage == --submit-prepared ]] || exit 2
[[ -r $venv/bin/activate && -d $data ]] || { echo 'Missing venv or data root.' >&2; exit 2; }
venv=$(cd -- "$venv" && pwd -P)
data=$(cd -- "$data" && pwd -P)
[[ $(scontrol show config | awk '$1=="ClusterName"{print $3}') == vulcan ]] || exit 2
module spider python/3.11.5 >/dev/null 2>&1
sinfo -h -o '%G' --Node | sort -u | grep -q 'gpu:l40s:'
accounts=$(sacctmgr -nP show assoc "user=$(id -un)" format=Account)
for account in aip-boyuwang aip-yiweilu; do
    cut -d'|' -f1 <<< "$accounts" | grep -Fxq "$account" || exit 2
done
for split in train test; do [[ -s $data/SVHN/${split}_32x32.mat ]] || exit 2; done
mkdir -p "$root/logs"
exec 9>"$root/.submit.lock"
flock -n 9 || { echo 'Another digit submitter is active.' >&2; exit 2; }
queue=$(squeue -u "$(id -un)" -h -o '%i|%j')

if [[ $stage == --prepare ]]; then
    if [[ -s $root/timing.json && -s $root/prepared.json ]]; then
        exec 9>&-
        exec bash "$launcher" --submit-prepared "$venv" "$data"
    fi
    if grep -q '|digits_prepare$' <<< "$queue"; then
        echo 'Digit preparation is already queued/running.'
        exit 0
    fi
    job=$(sbatch --parsable --job-name=digits_prepare --account=aip-boyuwang \
        --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=6G --gres=gpu:l40s:1 --time=00:15:00 \
        --chdir="$bench" --output="$root/logs/%x-%j.out" --error="$root/logs/%x-%j.err" \
        "$launcher" --worker "$bench" "$venv" "$data" prepare --launcher "$launcher" --venv "$venv")
    printf '%s\n' "${job%%;*}" > "$root/preparation_job.txt"
    printf 'Submitted preparation %s. It stages MNIST, measures runtimes, then submits six full-data jobs.\n' "$job"
    exit 0
fi

[[ -s $root/timing.json && -s $root/prepared.json ]] || exit 2
# Run here only after preparation, in its allocated GPU job; no login-node Python.
[[ -n ${SLURM_JOB_ID:-} ]] || {
    job=$(sbatch --parsable --job-name=digits_prepare --account=aip-boyuwang \
        --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=6G --gres=gpu:l40s:1 --time=00:05:00 \
        --chdir="$bench" --output="$root/logs/%x-%j.out" --error="$root/logs/%x-%j.err" \
        "$launcher" --worker "$bench" "$venv" "$data" prepare --launcher "$launcher" --venv "$venv")
    echo "Submitted preparation/resume dispatcher $job"
    exit 0
}
receipt="$root/submitted_$(date -u +%Y%m%dT%H%M%SZ)_${SLURM_JOB_ID}.tsv"
printf 'job_id\tdataset\tseed\taccount\twalltime\n' > "$receipt"
for source in SVHN MNIST; do
    seconds=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]]["requested_seconds"])' "$root/timing.json" "$source")
    [[ $seconds =~ ^[0-9]+$ && $seconds -le 604800 ]] || exit 2
    printf -v wall '%d:%02d:%02d' "$((seconds/3600))" "$((seconds/60%60))" "$((seconds%60))"
    for seed in 0 1 2; do
        name="digits_${source}_full_s${seed}"
        if grep -q "|$name$" <<< "$queue"; then echo "SKIP active $name"; continue; fi
        printf -v seed_dir 'seed_%03d' "$seed"
        output="$root/checkpoints/$source/ResNet18/full/pt100/simclr/$seed_dir"
        if [[ -s $output/complete.json ]]; then echo "SKIP completed $name"; continue; fi
        # Balanced split: three training seeds on each verified account.
        account=aip-boyuwang
        if [[ $source == SVHN && $seed == 2 || $source == MNIST && $seed != 0 ]]; then account=aip-yiweilu; fi
        job=$(sbatch --parsable --job-name="$name" --account="$account" \
            --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=6G --gres=gpu:l40s:1 --time="$wall" \
            --chdir="$bench" --output="$root/logs/%x-%j.out" --error="$root/logs/%x-%j.err" \
            "$launcher" --worker "$bench" "$venv" "$data" full --source "$source" --seed "$seed")
        printf '%s\t%s\t%s\t%s\t%s\n' "${job%%;*}" "$source" "$seed" "$account" "$wall" >> "$receipt"
        printf 'SUBMITTED %s: %s, %s, %s\n' "$job" "$name" "$account" "$wall"
    done
done
echo "Receipt: $receipt"
