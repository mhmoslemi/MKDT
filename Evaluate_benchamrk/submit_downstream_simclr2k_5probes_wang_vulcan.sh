#!/bin/bash
# Random/K-means: saved 2k encoder, five downstream probes, unified table only.
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
    export HF_HOME="$bench/downstream_simclr2k_1ssl5probes/cache/hf" XDG_CACHE_HOME="$bench/downstream_simclr2k_1ssl5probes/cache/xdg"
    export TORCH_HOME="$bench/downstream_simclr2k_1ssl5probes/cache/torch" TMPDIR="${SLURM_TMPDIR:?}"
    cd "$bench"
    exec python -u downstream_simclr2k_1ssl5probes/worker.py "$manifest" "$data"
fi
dry=0
if [[ ${1:-} == --dry-run ]]; then dry=1; shift; fi
[[ $# == 0 && -z ${SLURM_JOB_ID:-} ]] || exit 2
bench=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
run="$bench/downstream_simclr2k_1ssl5probes"
venv=${BENCH_VENV:-$HOME/ENV}
data=${BENCH_DATA_ROOT:-$bench/../../data}
[[ -r $venv/bin/activate && -d $data ]] || { echo 'Missing environment or data.' >&2; exit 2; }
venv=$(cd -- "$venv" && pwd -P)
data=$(cd -- "$data" && pwd -P)
[[ $(scontrol show config | awk '$1=="ClusterName" {print $3}') == vulcan ]] || exit 2
inventory=$(sinfo -h -o '%G' --Node | sort -u)
[[ $inventory == *gpu:l40s:* ]] || exit 2
accounts=$(sacctmgr -nP show assoc "user=$(id -un)" format=Account)
awk -F '|' '$1=="aip-boyuwang" {found=1} END {exit !found}' <<< "$accounts" || exit 2
module spider python/3.11.5 >/dev/null 2>&1
module spider nodejs/20.16.0 >/dev/null 2>&1
if ! command -v node >/dev/null; then module load StdEnv/2023 nodejs/20.16.0; fi
for marker in cifar-10-batches-py/data_batch_1 cifar-10-batches-py/test_batch \
    cifar-100-python/train cifar-100-python/test tinyimagenet.pt \
    fgvc-aircraft-2013b/data/variants.txt CUB_200_2011/images.txt StanfordDogs/train_list.mat StanfordDogs/test_list.mat; do
    [[ -r $data/$marker ]] || { echo "Missing dataset: $data/$marker" >&2; exit 2; }
done
mkdir -p "$run/batches" "$run/logs"
exec 9>"$run/.submit.lock"
flock -n 9 || exit 2
batch=$(mktemp -d "$run/batches/batch_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")
squeue -h -u "$(id -un)" -o '%200j' > "$batch/queue.txt"
node "$run/jobs.js" "$batch"
printf 'name\tjob\taccount\tprobes\twalltime\n' > "$batch/submitted.tsv"
while IFS=$'\t' read -r name seconds probes manifest; do
    printf -v wall '%02d:%02d:%02d' "$((seconds/3600))" "$((seconds/60%60))" "$((seconds%60))"
    job=DRY_RUN
    if (( ! dry )); then
        job=$(sbatch --parsable --job-name="$name" --account=aip-boyuwang \
            --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=6G --gres=gpu:l40s:1 --time="$wall" \
            --chdir="$bench" --output="$run/logs/%x-%j.out" --error="$run/logs/%x-%j.err" \
            "$bench/submit_downstream_simclr2k_5probes_wang_vulcan.sh" --worker "$bench" "$manifest" "$venv" "$data")
        job=${job%%;*}; [[ $job =~ ^[0-9]+$ ]] || exit 2
    fi
    printf '%s\t%s\taip-boyuwang\t%s\t%s\n' "$name" "$job" "$probes" "$wall" >> "$batch/submitted.tsv"
    printf '%s: %s, %s probes, Wang, %s\n' "$name" "$job" "$probes" "$wall"
done < "$batch/jobs.tsv"
printf 'Receipt: %s\n' "$batch/submitted.tsv"
