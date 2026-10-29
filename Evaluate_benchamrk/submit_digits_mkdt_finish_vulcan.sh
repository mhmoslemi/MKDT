#!/bin/bash
# Resume MNIST prerequisites; original MKDT -> 2k SimCLR -> five probes per label budget.
set -euo pipefail
if [[ ${1:-} == --worker ]]; then
    shift
    [[ $# == 4 ]] || exit 2
    bench=$1 manifest=$2 venv=$3 data=$4
    : "${SLURM_JOB_ID:?Submit through Slurm}"
    stage=$(jq -r .stage "$manifest")
    if [[ $stage == prerequisite ]]; then
        phase=$(jq -r .phase "$manifest")
        start=$(jq -r .start "$manifest")
        exec bash "$bench/submit_digits_mkdt_vulcan.sh" --worker "$bench" "$venv" "$data" \
            "$phase" --source MNIST --start "$start"
    fi
    module --force purge
    module load StdEnv/2023 python/3.11.5
    source "$venv/bin/activate"
    export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 WANDB_MODE=offline
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
    export XDG_CACHE_HOME="$bench/digits_mkdt_finish/cache/xdg" HF_HOME="$bench/digits_mkdt_finish/cache/hf"
    export TORCH_HOME="$bench/digits_mkdt_finish/cache/torch" TMPDIR="${SLURM_TMPDIR:?}"
    cd "$bench"
    python -c 'import ast,pathlib,sys; ast.parse(pathlib.Path(sys.argv[1]).read_text())' "$bench/digits_mkdt_finish/worker.py"
    exec python -u "$bench/digits_mkdt_finish/worker.py" "$manifest" "$data"
fi
dry=0
if [[ ${1:-} == --dry-run ]]; then dry=1; shift; fi
[[ $# == 0 && -z ${SLURM_JOB_ID:-} ]] || exit 2
bench=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
run="$bench/digits_mkdt_finish"
venv=${BENCH_VENV:-$HOME/ENV}
data=${BENCH_DATA_ROOT:-$bench/../../data}
[[ -r $venv/bin/activate && -d $data ]] || exit 2
venv=$(cd -- "$venv" && pwd -P)
data=$(cd -- "$data" && pwd -P)
[[ $(scontrol show config | awk '$1=="ClusterName" {print $3}') == vulcan ]] || exit 2
[[ $(sinfo -h -o '%G' --Node | sort -u) == *gpu:l40s:* ]] || exit 2
available=$(sacctmgr -nP show assoc "user=$(id -un)" format=Account)
for account in aip-boyuwang aip-yiweilu; do
    awk -F'|' -v account="$account" '$1==account {found=1} END {exit !found}' <<< "$available" || exit 2
done
module spider python/3.11.5 >/dev/null 2>&1
mkdir -p "$run/batches" "$run/logs"
exec 9>"$run/.submit.lock"
flock -n 9 || exit 2
batch=$(mktemp -d "$run/batches/batch_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")
squeue -h -u "$(id -un)" -o '%200j|%i' > "$batch/queue.tsv"
node "$run/plan.js" "$batch"
declare -A ids=()
while IFS='|' read -r name id; do name=${name// /}; [[ -z $name ]] || ids[$name]=${id// /}; done < "$batch/queue.tsv"
printf 'name\tstage\tjob\taccount\twalltime\tdependency\n' > "$batch/submitted.tsv"
while IFS=$'\t' read -r name stage account seconds parents manifest; do
    dependency=()
    if [[ $parents != - ]]; then
        parent_ids=(); IFS=',' read -ra parent_names <<< "$parents"
        for parent in "${parent_names[@]}"; do [[ -n ${ids[$parent]:-} ]] || exit 2; parent_ids+=("${ids[$parent]}"); done
        joined=$(IFS=:; echo "${parent_ids[*]}")
        dependency=("--dependency=afterok:$joined" --kill-on-invalid-dep=yes)
    fi
    printf -v wall '%02d:%02d:%02d' "$((seconds/3600))" "$((seconds/60%60))" "$((seconds%60))"
    gpu=(); if [[ $(jq -r .gpu "$manifest") == true ]]; then gpu=(--gres=gpu:l40s:1); fi
    job=DRY_RUN
    if (( ! dry )); then
        job=$(sbatch --parsable --job-name="$name" --account="$account" --nodes=1 --ntasks=1 \
            --cpus-per-task=1 --mem=6G "${gpu[@]}" --time="$wall" --chdir="$bench" \
            --output="$run/logs/%x-%j.out" --error="$run/logs/%x-%j.err" "${dependency[@]}" \
            "$bench/submit_digits_mkdt_finish_vulcan.sh" --worker "$bench" "$manifest" "$venv" "$data")
        job=${job%%;*}; [[ $job =~ ^[0-9]+$ ]] || exit 2
    fi
    ids[$name]=$job
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$name" "$stage" "$job" "$account" "$wall" "${dependency[*]:--}" >> "$batch/submitted.tsv"
    printf '%s: %s, %s, %s\n' "$name" "$job" "$account" "$wall"
done < "$batch/jobs.tsv"
printf 'Receipt: %s\n' "$batch/submitted.tsv"
