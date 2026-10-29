#!/bin/bash
# New MKDT prerequisites only; existing SimCLR jobs/data/code remain untouched.
set -euo pipefail

if [[ ${1:-} == --worker ]]; then
    shift
    [[ $# -ge 4 ]] || exit 2
    bench=$1 venv=$2 data=$3 stage=$4
    shift 4
    : "${SLURM_JOB_ID:?Use the submitter}"
    module --force purge
    module load StdEnv/2023 python/3.11.5
    source "$venv/bin/activate"
    export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 WANDB_MODE=offline
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
    export XDG_CACHE_HOME="$bench/digits_mkdt/cache" HF_HOME="$bench/digits_mkdt/cache/hf"
    export TORCH_HOME="$bench/digits_mkdt/cache/torch" TMPDIR="${SLURM_TMPDIR:?}"
    cd "$bench/digits_mkdt"
    exec python "$bench/digits_mkdt/worker.py" "$stage" --data-root "$data" "$@"
fi

bench=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
launcher="$bench/submit_digits_mkdt_vulcan.sh"
root="$bench/digits_mkdt"
stage=${1:---prepare}
venv=${2:-$HOME/ENV}
data=${3:-$bench/../../data}
[[ $stage == --prepare || $stage == --submit-prepared ]] || exit 2
[[ -r $venv/bin/activate && -d $data && -s $bench/digits_simclr/prepared.json ]] || {
    echo 'Missing environment, digit data, or prepared digit statistics.' >&2; exit 2;
}
venv=$(cd -- "$venv" && pwd -P)
data=$(cd -- "$data" && pwd -P)
[[ $(scontrol show config | awk '$1=="ClusterName"{print $3}') == vulcan ]] || exit 2
module spider python/3.11.5 >/dev/null 2>&1
inventory=$(sinfo -h -o '%G' --Node)
[[ $inventory == *gpu:l40s:* ]] || exit 2
accounts=$(sacctmgr -nP show assoc "user=$(id -un)" format=Account)
for account in aip-boyuwang aip-yiweilu; do
    cut -d'|' -f1 <<< "$accounts" | grep -Fxq "$account" || exit 2
done
mkdir -p "$root/logs"
exec 9>"$root/.submit.lock"
flock -n 9 || { echo 'Another MKDT digit submitter is active.' >&2; exit 2; }
queue=$(squeue -h -u "$(id -un)" -o '%i|%j')

if [[ $stage == --prepare || -z ${SLURM_JOB_ID:-} ]]; then
    if grep -q '|dmk_prepare$' <<< "$queue"; then
        echo 'MKDT preparation is already queued/running.'; exit 0
    fi
    job=$(sbatch --parsable --job-name=dmk_prepare --account=aip-boyuwang \
        --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=6G --gres=gpu:l40s:1 --time=00:10:00 \
        --chdir="$bench" --output="$root/logs/%x-%j.out" --error="$root/logs/%x-%j.err" \
        "$launcher" --worker "$bench" "$venv" "$data" prepare --launcher "$launcher" --venv "$venv")
    printf '%s\n' "${job%%;*}" > "$root/preparation_job.txt"
    echo "Submitted MKDT preparation $job; it retains epoch 1 and dispatches 46 prerequisite jobs."
    exit 0
fi

[[ -s $root/timing.json && -s $root/recipe.json ]] || exit 2
receipt="$root/submitted_$(date -u +%Y%m%dT%H%M%SZ)_${SLURM_JOB_ID}.tsv"
printf 'job_id\tdataset\tstage\tstart\taccount\twalltime\tdependencies\n' > "$receipt"
LAST_JOB=''
submit() {
    local source=$1 part=$2 start=$3 seconds=$4 dependency=$5 name account wall job
    name="dmk_${source}_${part}"
    [[ $part != experts ]] || name+="_${start}"
    job=$(awk -F'|' -v name="$name" '$2==name {print $1}' <<< "$queue")
    if [[ -n $job ]]; then LAST_JOB=$job; echo "SKIP active $name ($job)"; return; fi
    account=aip-boyuwang
    [[ $source != MNIST ]] || account=aip-yiweilu
    printf -v wall '%d:%02d:%02d' "$((seconds/3600))" "$((seconds/60%60))" "$((seconds%60))"
    local command=(sbatch --parsable --job-name="$name" --account="$account"
        --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=6G --time="$wall" --chdir="$bench"
        --output="$root/logs/%x-%j.out" --error="$root/logs/%x-%j.err")
    [[ $part == init ]] || command+=(--gres=gpu:l40s:1)
    [[ -z $dependency ]] || command+=(--dependency="afterok:$dependency" --kill-on-invalid-dep=yes)
    job=$("${command[@]}" "$launcher" --worker "$bench" "$venv" "$data" "$part" --source "$source" --start "$start")
    job=${job%%;*}
    [[ $job =~ ^[0-9]+$ ]] || exit 2
    LAST_JOB=$job
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$job" "$source" "$part" "$start" "$account" "$wall" "$dependency" >> "$receipt"
    printf 'SUBMITTED %s: %s, %s, %s, after=%s\n' "$job" "$name" "$account" "$wall" "${dependency:-none}"
}

for source in SVHN MNIST; do
    read -r teacher_time feature_time expert_time init_time < <(python -c \
        'import json,sys; t=json.load(open(sys.argv[1]))[sys.argv[2]]; print(*(t[k] for k in ("teacher","features","experts","init")))' \
        "$root/timing.json" "$source")
    for seconds in "$teacher_time" "$feature_time" "$expert_time" "$init_time"; do
        [[ $seconds =~ ^[0-9]+$ && $seconds -ge 60 && $seconds -le 604800 ]] || exit 2
    done
    dependency=''
    if [[ ! -s $root/teachers/$source/complete.json ]]; then
        submit "$source" teacher 0 "$teacher_time" ''
        dependency=$LAST_JOB
    fi
    if [[ ! -s $root/target_rep/barlow_twins/${source}_target_rep_train.json || -n $dependency ]]; then
        submit "$source" features 0 "$feature_time" "$dependency"
        dependency=$LAST_JOB
    fi
    shard_jobs=()
    for start in $(seq 0 5 95); do
        printf -v key '%03d' "$start"
        [[ ! -s $root/work/$source/shard_${key}.json ]] || continue
        submit "$source" experts "$start" "$expert_time" "$dependency"
        shard_jobs+=("$LAST_JOB")
    done
    if [[ ${#shard_jobs[@]} -gt 0 ]]; then dependency=$(IFS=:; echo "${shard_jobs[*]}"); fi
    if [[ ! -s $root/work/$source/init_complete.json || -n $dependency ]]; then
        submit "$source" init 0 "$init_time" "$dependency"
    fi
done
echo "MKDT prerequisite receipt: $receipt"
