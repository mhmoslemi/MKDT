#!/bin/bash
# Reuse full teachers; one A2MIM encoder per existing subset, five probes per table cell.
set -euo pipefail
if [[ ${1:-} == --worker ]]; then
    shift
    [[ $# == 4 ]] || exit 2
    bench=$1 manifest=$2 venv=$3 data=$4
    : "${SLURM_JOB_ID:?Submit through Slurm}"
    module --force purge
    module load StdEnv/2023 gcc/12.3 cuda/12.6 python/3.11.5 nodejs/20.16.0
    source "$venv/bin/activate"
    export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
    export HF_HOME="$bench/a2mim_eval_2k_1ssl5probes/cache/hf" XDG_CACHE_HOME="$bench/a2mim_eval_2k_1ssl5probes/cache/xdg"
    export TORCH_HOME="$bench/a2mim_eval_2k_1ssl5probes/cache/torch" TMPDIR="${SLURM_TMPDIR:?}"
    cd "$bench"
    exec python -u a2mim_eval_2k_1ssl5probes/worker.py "$manifest" "$data"
fi
dry=0
if [[ ${1:-} == --dry-run ]]; then dry=1; shift; fi
[[ $# == 0 && -z ${SLURM_JOB_ID:-} ]] || exit 2
bench=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
run="$bench/a2mim_eval_2k_1ssl5probes"
venv=${BENCH_VENV:-$HOME/ENV}
data=${BENCH_DATA_ROOT:-$bench/../../data}
[[ -r $venv/bin/activate && -d $data ]] || exit 2
venv=$(cd -- "$venv" && pwd -P)
data=$(cd -- "$data" && pwd -P)
[[ $(scontrol show config | awk '$1=="ClusterName" {print $3}') == vulcan ]] || exit 2
inventory=$(sinfo -h -o '%G' --Node | sort -u)
[[ $inventory == *gpu:l40s:* ]] || exit 2
available=$(sacctmgr -nP show assoc "user=$(id -un)" format=Account)
for account in aip-boyuwang aip-yiweilu; do
    awk -F '|' -v account="$account" '$1==account {found=1} END {exit !found}' <<< "$available" || exit 2
done
module spider python/3.11.5 >/dev/null 2>&1
module spider nodejs/20.16.0 >/dev/null 2>&1
module spider gcc/12.3 >/dev/null 2>&1
module spider cuda/12.6 >/dev/null 2>&1
if ! command -v node >/dev/null; then module load StdEnv/2023 nodejs/20.16.0; fi
for marker in cifar-10-batches-py/data_batch_1 cifar-10-batches-py/test_batch cifar-100-python/train cifar-100-python/test tinyimagenet.pt; do
    [[ -r $data/$marker ]] || { echo "Missing dataset: $data/$marker" >&2; exit 2; }
done
mkdir -p "$run/batches" "$run/logs"
exec 9>"$run/.submit.lock"
flock -n 9 || exit 2
batch=$(mktemp -d "$run/batches/batch_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")
squeue -h -u "$(id -un)" -o '%200j|%i' > "$batch/queue.tsv"
node "$run/jobs.js" "$batch"
declare -A ids=()
while IFS='|' read -r name id; do name=${name// /}; [[ -z $name ]] || ids[$name]=${id// /}; done < "$batch/queue.tsv"
printf 'name\tstage\tjob\taccount\twalltime\tdependency\n' > "$batch/submitted.tsv"
new_ids=()
while IFS=$'\t' read -r name stage account seconds parents manifest; do
    dependency=()
    if [[ $parents != - ]]; then
        IFS=, read -ra parent_names <<< "$parents"
        parent_ids=()
        for parent in "${parent_names[@]}"; do [[ -n ${ids[$parent]:-} ]] || exit 2; parent_ids+=("${ids[$parent]}"); done
        joined=$(IFS=:; echo "${parent_ids[*]}")
        dependency=("--dependency=afterok:$joined" --kill-on-invalid-dep=yes)
    fi
    printf -v wall '%02d:%02d:%02d' "$((seconds/3600))" "$((seconds/60%60))" "$((seconds%60))"
    warning=()
    [[ $stage != ssl ]] || warning=(--signal=USR1@30)
    job=DRY_RUN
    if (( ! dry )); then
        job=$(sbatch --parsable --hold --job-name="$name" --account="$account" --nodes=1 --ntasks=1 \
            --cpus-per-task=1 --mem=6G --gres=gpu:l40s:1 --time="$wall" "${warning[@]}" --chdir="$bench" \
            --output="$run/logs/%x-%j.out" --error="$run/logs/%x-%j.err" "${dependency[@]}" \
            "$bench/submit_a2mim_table_2k_1ssl5probes_vulcan.sh" --worker "$bench" "$manifest" "$venv" "$data")
        job=${job%%;*}; [[ $job =~ ^[0-9]+$ ]] || exit 2
        new_ids+=("$job")
    fi
    ids[$name]=$job
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$name" "$stage" "$job" "$account" "$wall" "${dependency[*]:--}" >> "$batch/submitted.tsv"
    printf '%s: %s, %s, %s\n' "$name" "$job" "$account" "$wall"
done < "$batch/jobs.tsv"
if (( ! dry && ${#new_ids[@]} )); then scontrol release "$(IFS=,; echo "${new_ids[*]}")"; fi
printf 'Receipt: %s\n' "$batch/submitted.tsv"
