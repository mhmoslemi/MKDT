#!/bin/bash
set -euo pipefail
if [[ ${1:-} == --worker ]]; then
    shift
    [[ $# == 4 ]] || exit 2
    bench=$1 manifest=$2 venv=$3 data=$4
    : "${SLURM_JOB_ID:?Submit through this launcher}"
    module --force purge
    module load StdEnv/2023 python/3.11.5 nodejs/20.16.0
    source "$venv/bin/activate"
    export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
    export HF_HOME="$bench/cifar_small_simclr2k_1ssl5probes/cache/hf" XDG_CACHE_HOME="$bench/cifar_small_simclr2k_1ssl5probes/cache/xdg"
    export TORCH_HOME="$bench/cifar_small_simclr2k_1ssl5probes/cache/torch" TMPDIR="${SLURM_TMPDIR:?}"
    cd "$bench"
    exec python -u cifar_small_simclr2k_1ssl5probes/worker.py "$manifest" "$data"
fi
dry_run=0
if [[ ${1:-} == --dry-run ]]; then dry_run=1; shift; fi
[[ $# == 0 && -z ${SLURM_JOB_ID:-} ]] || { echo 'Run on the Vulcan login node.' >&2; exit 2; }
bench=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
venv=${BENCH_VENV:-$HOME/ENV}
data=${BENCH_DATA_ROOT:-$bench/../../data}
[[ -r $venv/bin/activate && -d $data ]] || { echo 'Missing environment or data.' >&2; exit 2; }
venv=$(cd -- "$venv" && pwd -P)
data=$(cd -- "$data" && pwd -P)
[[ $(scontrol show config | awk '$1=="ClusterName" {print $3}') == vulcan ]] || exit 2
inventory=$(sinfo -h -o '%G' --Node)
[[ $inventory == *gpu:l40s:* ]] || { echo 'L40S unavailable.' >&2; exit 2; }
module spider python/3.11.5 >/dev/null 2>&1
module spider nodejs/20.16.0 >/dev/null 2>&1
if ! command -v node >/dev/null; then module load StdEnv/2023 nodejs/20.16.0; fi
for marker in cifar-10-batches-py/data_batch_1 cifar-100-python/train; do
    [[ -r $data/$marker ]] || { echo "Missing dataset: $data/$marker" >&2; exit 2; }
done
run="$bench/cifar_small_simclr2k_1ssl5probes"
mkdir -p "$run/batches" "$run/logs"
exec 9>"$run/.submit.lock"
flock -n 9 || { echo 'Another copy is submitting.' >&2; exit 2; }
batch=$(mktemp -d "$run/batches/batch_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")
squeue -h -u "$(id -un)" -o '%200j|%i' > "$batch/queue.tsv"
node "$run/plan.js" "$batch" "$batch/queue.tsv"
declare -A ids=()
while IFS='|' read -r name id; do name=${name// /}; [[ -z $name ]] || ids[$name]=$id; done < "$batch/queue.tsv"
printf 'name\tstage\tjob\taccount\twalltime\tdependency\n' > "$batch/submitted.tsv"
while IFS=$'\t' read -r name stage seconds dependencies manifest; do
    dependency=()
    if [[ $dependencies != - ]]; then
        [[ -n ${ids[$dependencies]:-} ]] || { echo "Missing encoder: $dependencies" >&2; exit 2; }
        dependency=("--dependency=afterok:${ids[$dependencies]}" --kill-on-invalid-dep=yes)
    fi
    printf -v walltime '%02d:%02d:%02d' "$((seconds/3600))" "$((seconds/60%60))" "$((seconds%60))"
    printf '%s: %s, Wang, 1 L40S / 1 CPU / 6G\n' "$name" "$walltime"
    job=DRY_RUN
    if (( ! dry_run )); then
        job=$(sbatch --parsable --job-name="$name" --account=aip-boyuwang \
            --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=6G --gres=gpu:l40s:1 --time="$walltime" \
            --chdir="$bench" --output="$run/logs/%x-%j.out" --error="$run/logs/%x-%j.err" \
            "${dependency[@]}" "$bench/submit_cifar_small_simclr2k_1ssl5probes_wang_vulcan.sh" --worker "$bench" "$manifest" "$venv" "$data")
        job=${job%%;*}
        [[ $job =~ ^[0-9]+$ ]] || { echo "Unexpected job ID: $job" >&2; exit 2; }
    fi
    ids[$name]=$job
    printf '%s\t%s\t%s\taip-boyuwang\t%s\t%s\n' "$name" "$stage" "$job" "$walltime" "${dependency[*]:--}" >> "$batch/submitted.tsv"
done < "$batch/jobs.tsv"
printf 'Receipt: %s\n' "$batch/submitted.tsv"
