#!/bin/bash
# MKDT generation only: 100 KD experts/dataset, high-loss initialization, trajectory matching.
set -euo pipefail

if [[ ${1:-} == --worker ]]; then
    shift
    [[ $# == 8 ]] || { echo 'Invalid worker arguments.' >&2; exit 2; }
    code=$1 stage=$2 dataset=$3 stop=$4 work=$5 synthetic=$6 data=$7 venv=$8
    : "${SLURM_JOB_ID:?Submit through this launcher}"
    module --force purge
    module load StdEnv/2023 python/3.11.5
    source "$venv/bin/activate"
    export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
    export HF_HOME="$work/cache/hf" XDG_CACHE_HOME="$work/cache/xdg"
    export WANDB_MODE=offline TMPDIR="${SLURM_TMPDIR:?}"
    cd "$code"
    export PYTHONPATH="$code:$(cd -- "$work/../../.." && pwd -P)${PYTHONPATH:+:$PYTHONPATH}"
    if [[ ${SLURM_CLUSTER_NAME:-${CC_CLUSTER:-}} == vulcan ]]; then
        exec python -u vulcan_worker.py "$stage" "$dataset" "$stop" "$work" "$synthetic" "$data"
    fi
    exec python -u worker.py "$stage" "$dataset" "$stop" "$work" "$synthetic" "$data"
fi

dry_run=0
if [[ ${1:-} == --dry-run ]]; then dry_run=1; shift; fi
[[ $# == 0 && -z ${SLURM_JOB_ID:-} ]] || { echo 'Run this submitter on the destination login node.' >&2; exit 2; }
bench=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
repo=$(cd -- "$bench/.." && pwd -P)
work="$bench/mkdt_02_jobs/runs"
synthetic=${MKDT_SYNTH_ROOT:-$repo/benchmarks/mkdt_multidataset/results/batch_20261010T160046Z_Y6dqZS/synthetic}
data=${BENCH_DATA_ROOT:-$repo/../data}
venv=${BENCH_VENV:-$HOME/ENV}
[[ -r $venv/bin/activate && -d $data && -d $synthetic ]] || { echo 'Missing environment, data or MKDT synthetic root.' >&2; exit 2; }
synthetic=$(cd -- "$synthetic" && pwd -P)
data=$(cd -- "$data" && pwd -P)
venv=$(cd -- "$venv" && pwd -P)
cluster=$(scontrol show config | awk '$1=="ClusterName" {print $3}')
expected_cluster=${MKDT_SUBMIT_CLUSTER:-killarney}
[[ $expected_cluster == killarney || $expected_cluster == vulcan ]] || exit 2
[[ $cluster == "$expected_cluster" ]] || { echo "Submit on $expected_cluster only." >&2; exit 2; }
inventory=$(sinfo -h -o '%G' --Node)
[[ $inventory == *gpu:l40s:* ]] || { echo 'L40S GPU type not found.' >&2; exit 2; }
accounts=$(sacctmgr -nP show assoc "user=$(id -un)" format=Account)
awk -F '|' '$1=="aip-boyuwang" {found=1} END {exit !found}' <<< "$accounts" || { echo 'Wang account unavailable.' >&2; exit 2; }
if [[ $cluster == vulcan ]]; then
    awk -F '|' '$1=="aip-yiweilu" {found=1} END {exit !found}' <<< "$accounts" || { echo 'Lu account unavailable.' >&2; exit 2; }
fi
module spider python/3.11.5 >/dev/null 2>&1
for dataset in CIFAR10 CIFAR100 Tiny; do
    [[ -s $synthetic/work/target_rep/barlow_twins/${dataset}_target_rep_train.pt ]] || {
        echo "Missing saved teacher representations for $dataset; no SSL teacher retraining is submitted." >&2; exit 2;
    }
done
for marker in cifar-10-batches-py/data_batch_1 cifar-100-python/train tinyimagenet.pt; do
    [[ -r $data/$marker ]] || { echo "Missing data: $data/$marker" >&2; exit 2; }
done
mkdir -p "$work/batches" "$work/logs"
exec 9>"$work/.submit.lock"
flock -n 9 || { echo 'Another copy of this submitter is running.' >&2; exit 2; }
cd "$repo"
sources=(buffer.py sort_dset.py distill.py utils.py networks.py reparam_module.py
    Evaluate_benchamrk/mkdt_02_jobs/worker.py Evaluate_benchamrk/mkdt_02_jobs/check_resume.py)
recipe=$(sha256sum "${sources[@]}" | sha256sum | awk '{print $1}')
if [[ -f $work/recipe.sha256 ]]; then
    [[ $(< "$work/recipe.sha256") == "$recipe" ]] || { echo 'Generation code changed since the first submission; existing runs were preserved. Do not mix recipes.' >&2; exit 2; }
else
    printf '%s\n' "$recipe" > "$work/recipe.sha256"
fi
batch=$(mktemp -d "$work/batches/batch_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")
mkdir "$batch/code"
cp "${sources[@]}" "$batch/code/"
[[ $cluster != vulcan ]] || cp "$bench/mkdt_02_jobs/vulcan_worker.py" "$batch/code/"
cp "$bench/submit_mkdt_02_wang_killarney.sh" "$batch/submit.sh"
declare -A active
while IFS='|' read -r name id; do
    name=${name//[[:space:]]/}; id=${id//[[:space:]]/}
    [[ $name == mk02_* && $id =~ ^[0-9]+$ ]] && active[$name]=$id
done < <(squeue -h -u "$(id -un)" -t PD,R,CF,CG -o '%150j|%i')
printf 'name\tjob\tstage\tdataset\tstop\twalltime\tdependency\taccount\n' > "$batch/submitted.tsv"
submitted=0
LAST_JOB=''
submit() {
    local name=$1 stage=$2 dataset=$3 stop=$4 seconds=$5 dependency=$6 walltime job account=aip-boyuwang
    if [[ $cluster == vulcan && $dataset == CIFAR100 ]]; then account=aip-yiweilu; fi
    if [[ -n ${active[$name]:-} ]]; then LAST_JOB=${active[$name]}; echo "ALREADY QUEUED: $name ($LAST_JOB)"; return; fi
    printf -v walltime '%02d:%02d:%02d' "$((seconds/3600))" "$((seconds/60%60))" "$((seconds%60))"
    local command=(sbatch --parsable --job-name="$name" --account="$account"
        --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=6G --time="$walltime"
        --chdir="$repo" --output="$work/logs/%x-%j.out" --error="$work/logs/%x-%j.err")
    if [[ $stage != init ]]; then
        command+=(--gres=gpu:l40s:1)
        [[ $cluster != killarney ]] || command+=(--exclude=kn101)
    fi
    [[ -z $dependency ]] || command+=(--dependency="afterok:$dependency" --kill-on-invalid-dep=yes)
    submitted=$((submitted + 1))
    if (( dry_run )); then
        job=$((900000000 + submitted))
    else
        job=$("${command[@]}" "$batch/submit.sh" --worker "$batch/code" "$stage" "$dataset" "$stop" "$work" "$synthetic" "$data" "$venv")
        job=${job%%;*}
        [[ $job =~ ^[0-9]+$ ]] || { echo "Unexpected sbatch output: $job" >&2; exit 2; }
    fi
    LAST_JOB=$job
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$name" "$job" "$stage" "$dataset" "$stop" "$walltime" "$dependency" "$account" >> "$batch/submitted.tsv"
    printf '%s: %s, %s, dependency=%s\n' "$name" "$walltime" "$account" "${dependency:-none}"
}

for dataset in CIFAR10 CIFAR100 Tiny; do
    folder="$work/$dataset"
    source=$dataset; total=5000; expert_seconds=750; distill_seconds=1650
    if [[ $dataset == Tiny ]]; then source=TinyImageNet; total=10000; expert_seconds=1950; distill_seconds=2250; fi
    output="$synthetic/generated/ConvNet/$source/${dataset}_0.2pct"
    if [[ -s $output/complete_0.2.json && -s $output/images_${total}.pt ]]; then
        echo "SKIP completed 0.2% generation: $dataset"; continue
    fi
    dependencies=()
    for start in $(seq 0 5 95); do
        [[ -s $folder/shard_$(printf '%03d' "$start").json ]] && continue
        submit "mk02_${dataset}_exp_${start}" experts "$dataset" "$start" "$expert_seconds" ''
        dependencies+=("$LAST_JOB")
    done
    dependency=$(IFS=:; echo "${dependencies[*]}")
    if [[ ! -s $folder/init_complete.json || -n $dependency ]]; then
        submit "mk02_${dataset}_init" init "$dataset" 0 210 "$dependency"
        dependency=$LAST_JOB
    fi
    for stop in $(seq 1000 1000 "$total"); do
        [[ -s $folder/distill_$(printf '%05d' "$stop").json ]] && continue
        submit "mk02_${dataset}_dist_${stop}" distill "$dataset" "$stop" "$distill_seconds" "$dependency"
        dependency=$LAST_JOB
    done
done
printf 'DONE: %s jobs %s. Receipt: %s\n' "$submitted" "$([[ $dry_run == 1 ]] && echo planned || echo submitted)" "$batch/submitted.tsv"
echo 'Re-run this same command after a timeout: completed stages are skipped and distillation restores full training.pt state.'
