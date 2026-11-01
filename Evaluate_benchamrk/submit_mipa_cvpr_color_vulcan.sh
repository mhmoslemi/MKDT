#!/bin/bash
set -euo pipefail
bench=/scratch/mmoslem3/MKDT/Evaluate_benchamrk
run="$bench/mipa_cvpr_color"
if [[ ${1:-} == --worker ]]; then
    : "${SLURM_JOB_ID:?Submit through Slurm}"
    module --force purge
    module load StdEnv/2023 python/3.11.5
    source /home/mmoslem3/ENV/bin/activate
    export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
    export XDG_CACHE_HOME="$run/cache" TORCH_HOME="$run/cache/torch" HF_HOME="$run/cache/hf" MPLCONFIGDIR="$run/cache/matplotlib"
    export TMPDIR="${SLURM_TMPDIR:?}"
    cd "$bench"
    exec python -u "$run/snapshot/worker.py" "$2" "$3"
fi
[[ $# == 0 || ( $# == 1 && $1 == --dry-run ) ]] || exit 2
dry=0
[[ ${1:-} != --dry-run ]] || dry=1
cluster=${MIPA_CVPR_CLUSTER:-vulcan}
[[ $cluster == vulcan || $cluster == killarney ]] || exit 2
[[ $(scontrol show config | awk '$1=="ClusterName" {print $3}') == "$cluster" ]] || { echo "Run this launcher on $cluster." >&2; exit 2; }
[[ $(sinfo -h -o '%G' --Node | sort -u) == *gpu:l40s:* ]] || exit 2
accounts=$(sacctmgr -nP show assoc user=mmoslem3 format=Account)
requested_account=aip-boyuwang
[[ $cluster != killarney ]] || requested_account=aip-yiweilu
awk -F '|' -v account="$requested_account" '$1==account {found=1} END {exit !found}' <<< "$accounts"
module spider python/3.11.5 >/dev/null 2>&1
if ! command -v node >/dev/null; then
    module spider nodejs/20.16.0 >/dev/null 2>&1
    module load StdEnv/2023 nodejs/20.16.0
fi
logs="$run/logs"
receipt="$run/submitted.tsv"
if [[ $cluster == killarney ]]; then
    for file in "$run/manifest.json" "$run/snapshot/method1_cvpr.py" "$run/snapshot/worker.py"; do
        [[ -s $file ]] || { echo "Sync mipa_cvpr_color/manifest.json and mipa_cvpr_color/snapshot/ from Vulcan first: missing $file" >&2; exit 2; }
    done
    logs="$run/logs/killarney"
    receipt="$run/submitted_killarney.tsv"
    for split in train test; do
        target="/scratch/mmoslem3/data/SVHN/${split}_32x32.mat"
        original="/scratch/mmoslem3/data/${split}_32x32.mat"
        if [[ ! -e $target && ! -L $target && -s $original ]]; then
            mkdir -p /scratch/mmoslem3/data/SVHN
            ln -s "$original" "$target"
        fi
    done
fi
for file in /scratch/mmoslem3/data/cifar-10-batches-py/data_batch_1 /scratch/mmoslem3/data/cifar-100-python/train /scratch/mmoslem3/data/tinyimagenet.pt /scratch/mmoslem3/data/SVHN/train_32x32.mat /scratch/mmoslem3/data/SVHN/test_32x32.mat /home/mmoslem3/ENV/bin/activate; do
    [[ -s $file ]] || { echo "Missing: $file" >&2; exit 2; }
done
mkdir -p "$logs"
exec 9>"$run/.submit.lock"
flock -n 9 || exit 2
plan=$(node "$run/plan.js")
declare -A active=() active_account=()
while IFS='|' read -r name id account; do active["$name"]=$id; active_account["$name"]=$account; done < <(squeue -h -u mmoslem3 -o '%j|%i|%a')
submit() {
    local name=$1 account=$2 memory=$3 seconds=$4 manifest=$5 stage=$6 parent=${7:-} wall job
    if [[ -n ${active[$name]:-} ]]; then
        [[ ${active_account[$name]} == "$account" ]] || { echo "$name (${active[$name]}) is already queued under ${active_account[$name]}, not $account; existing job left untouched." >&2; exit 2; }
        REPLY=${active[$name]}; echo "ACTIVE $name: $REPLY"; return
    fi
    printf -v wall '%02d:%02d:%02d' "$((seconds/3600))" "$((seconds/60%60))" "$((seconds%60))"
    local dependency=()
    [[ -z $parent ]] || dependency=("--dependency=afterok:$parent" --kill-on-invalid-dep=yes)
    job=DRY_RUN
    if (( ! dry )); then
        job=$(sbatch --parsable --job-name="$name" --account="$account" --nodes=1 --ntasks=1 --cpus-per-task=1 --mem="$memory" --gres=gpu:l40s:1 --time="$wall" --chdir="$bench" --output="$logs/%x-%j.out" --error="$logs/%x-%j.err" "${dependency[@]}" "$bench/submit_mipa_cvpr_color_vulcan.sh" --worker "$manifest" "$stage")
        job=${job%%;*}
        [[ $job =~ ^[0-9]+$ ]] || exit 2
        printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$name" "$job" "$account" "$wall" "$stage" "${parent:--}" >> "$receipt"
    fi
    active[$name]=$job
    active_account[$name]=$account
    REPLY=$job
    echo "$name: $job, $account, $wall"
}
while IFS=$'\t' read -r unit account memory generation_seconds evaluation_seconds manifest; do
    [[ -n $unit ]] || continue
    parent=
    if (( generation_seconds > 0 )); then
        submit "cvprg_$unit" "$account" "$memory" "$generation_seconds" "$manifest" generation
        parent=$REPLY
    fi
    submit "cvpre_$unit" "$account" "$memory" "$evaluation_seconds" "$manifest" evaluation "$parent"
done <<< "$plan"
echo "Receipt: $receipt"
