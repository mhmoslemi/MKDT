#!/bin/bash
set -euo pipefail
if [[ ${1:-} == --worker ]]; then
    shift
    [[ $# == 4 ]] || exit 2
    bench=$1 manifest=$2 venv=$3 data=$4
    : "${SLURM_JOB_ID:?Submit through Slurm}"
    module --force purge
    module load StdEnv/2023 python/3.11.5 nodejs/20.16.0
    source "$venv/bin/activate"
    export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
    export HF_HOME="$bench/cifar_small_simclr2k_1ssl5probes/cache/hf" XDG_CACHE_HOME="$bench/cifar_small_simclr2k_1ssl5probes/cache/xdg"
    export TORCH_HOME="$bench/cifar_small_simclr2k_1ssl5probes/cache/torch" TMPDIR="${SLURM_TMPDIR:?}"
    cd "$bench"
    exec node cifar_small_simclr2k_1ssl5probes/bundles.js --run "$manifest" "$data"
fi
dry=0 hold=0
while (( $# )); do
    case "$1" in --dry-run) dry=1 ;; --hold) hold=1 ;; *) exit 2 ;; esac
    shift
done
[[ -z ${SLURM_JOB_ID:-} ]] || exit 2
bench=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
venv=${BENCH_VENV:-$HOME/ENV}
data=${BENCH_DATA_ROOT:-$bench/../../data}
[[ -r $venv/bin/activate && -d $data ]] || exit 2
venv=$(cd -- "$venv" && pwd -P)
data=$(cd -- "$data" && pwd -P)
[[ $(scontrol show config | awk '$1=="ClusterName" {print $3}') == vulcan ]] || exit 2
[[ $(sinfo -h -o '%G' --Node) == *gpu:l40s:* ]] || exit 2
module spider python/3.11.5 >/dev/null 2>&1
module spider nodejs/20.16.0 >/dev/null 2>&1
if ! command -v node >/dev/null; then module load StdEnv/2023 nodejs/20.16.0; fi
run="$bench/cifar_small_simclr2k_1ssl5probes"
exec 9>"$run/.submit.lock"
flock -n 9 || exit 2
batch=$(mktemp -d "$run/batches/bundled_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")
squeue -h -u "$(id -un)" -o '%200j|%i|%T' > "$batch/queue.tsv"
node "$run/bundles.js" "$batch"
declare -A ids=()
while IFS='|' read -r name id state; do name=${name// /}; [[ -z $name ]] || ids[$name]=$id; done < "$batch/queue.tsv"
printf 'name\tstage\tjob\taccount\twalltime\tdependency\n' > "$batch/submitted.tsv"
new_ids=()
while IFS=$'\t' read -r name stage seconds parents manifest; do
    dependency=()
    if [[ $parents != - ]]; then
        IFS=, read -ra names <<< "$parents"
        parents_ids=()
        for parent in "${names[@]}"; do [[ -n ${ids[$parent]:-} ]] || exit 2; parents_ids+=("${ids[$parent]}"); done
        joined=$(IFS=:; echo "${parents_ids[*]}")
        dependency=("--dependency=afterok:$joined" --kill-on-invalid-dep=yes)
    fi
    printf -v wall '%02d:%02d:%02d' "$((seconds/3600))" "$((seconds/60%60))" "$((seconds%60))"
    script="$bench/submit_cifar_small_simclr2k_bundled_wang_vulcan.sh"
    [[ $stage != ssl ]] || script="$bench/submit_cifar_small_simclr2k_1ssl5probes_wang_vulcan.sh"
    job=DRY_RUN
    if (( ! dry )); then
        job=$(sbatch --parsable --hold --job-name="$name" --account=aip-boyuwang --nodes=1 --ntasks=1 \
            --cpus-per-task=1 --mem=6G --gres=gpu:l40s:1 --time="$wall" --chdir="$bench" \
            --output="$run/logs/%x-%j.out" --error="$run/logs/%x-%j.err" "${dependency[@]}" \
            "$script" --worker "$bench" "$manifest" "$venv" "$data")
        job=${job%%;*}; [[ $job =~ ^[0-9]+$ ]] || exit 2
        new_ids+=("$job")
    fi
    ids[$name]=$job
    printf '%s\t%s\t%s\taip-boyuwang\t%s\t%s\n' "$name" "$stage" "$job" "$wall" "${dependency[*]:--}" >> "$batch/submitted.tsv"
    printf '%s: %s, Wang, %s\n' "$name" "$job" "$wall"
done < "$batch/bundled_jobs.tsv"
if (( ! dry && ! hold && ${#new_ids[@]} )); then scontrol release "$(IFS=,; echo "${new_ids[*]}")"; fi
printf 'Receipt: %s\n' "$batch/submitted.tsv"
