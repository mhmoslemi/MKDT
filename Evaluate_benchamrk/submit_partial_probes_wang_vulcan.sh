#!/bin/bash
# Complete the selected partial cells using frozen encoders only; never train SSL.
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
    export HF_HOME="$bench/partial_probe_jobs/cache/hf" XDG_CACHE_HOME="$bench/partial_probe_jobs/cache/xdg"
    export TORCH_HOME="$bench/partial_probe_jobs/cache/torch" TMPDIR="${SLURM_TMPDIR:?}"
    cd "$bench"
    exec python partial_probe_jobs/worker.py "$manifest" "$data"
fi
dry_run=0
if [[ ${1:-} == --dry-run ]]; then dry_run=1; shift; fi
[[ $# == 0 && -z ${SLURM_JOB_ID:-} ]] || { echo 'Run with bash on the Vulcan login node.' >&2; exit 2; }
bench=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
venv=${BENCH_VENV:-$HOME/ENV}
data=${BENCH_DATA_ROOT:-$bench/../../data}
[[ -r $venv/bin/activate && -d $data ]] || { echo 'Missing environment or data.' >&2; exit 2; }
venv=$(cd -- "$venv" && pwd -P)
data=$(cd -- "$data" && pwd -P)
cluster=$(scontrol show config | awk '$1=="ClusterName" {print $3}')
[[ $cluster == vulcan ]] || { echo 'Vulcan only.' >&2; exit 2; }
inventory=$(sinfo -h -o '%G' --Node)
[[ $inventory == *gpu:l40s:* ]] || { echo 'L40S GPU type not found.' >&2; exit 2; }
accounts=$(sacctmgr -nP show assoc "user=$(id -un)" format=Account)
awk -F '|' '$1=="aip-boyuwang" {found=1} END {exit !found}' <<< "$accounts" || { echo 'Wang account unavailable.' >&2; exit 2; }
module spider python/3.11.5 >/dev/null 2>&1
module spider nodejs/20.16.0 >/dev/null 2>&1
if ! command -v node >/dev/null; then module load StdEnv/2023 nodejs/20.16.0; fi
for marker in cifar-10-batches-py/data_batch_1 cifar-100-python/train tinyimagenet.pt \
    fgvc-aircraft-2013b/data/variants.txt CUB_200_2011/images.txt StanfordDogs/train_list.mat Oxford102Flowers/setid.mat; do
    [[ -r $data/$marker ]] || { echo "Missing dataset: $data/$marker" >&2; exit 2; }
done
run="$bench/partial_probe_jobs"
mkdir -p "$run/batches" "$run/logs"
exec 9>"$run/.submit.lock"
flock -n 9 || { echo 'Another copy of this submitter is running.' >&2; exit 2; }
batch=$(mktemp -d "$run/batches/batch_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")
queue=$(squeue -h -u "$(id -un)" -o '%200j' | awk '{$1=$1;print}')
node "$run/plan.js" "$batch" "$queue"
printf 'name\tjob\taccount\tmissing_probes\twalltime\n' > "$batch/submitted.tsv"
submitted=0
while IFS=$'\t' read -r name seconds missing manifest; do
    printf -v walltime '%02d:%02d:%02d' "$((seconds/3600))" "$((seconds/60%60))" "$((seconds%60))"
    printf '%s: %s new probes, %s, Wang/L40S/1 CPU/6G\n' "$name" "$missing" "$walltime"
    job=DRY_RUN
    if (( ! dry_run )); then
        job=$(sbatch --parsable --job-name="$name" --account=aip-boyuwang \
            --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=6G --gres=gpu:l40s:1 \
            --time="$walltime" --chdir="$bench" --output="$run/logs/%x-%j.out" --error="$run/logs/%x-%j.err" \
            "$bench/submit_partial_probes_wang_vulcan.sh" --worker "$bench" "$manifest" "$venv" "$data")
        job=${job%%;*}
        [[ $job =~ ^[0-9]+$ ]] || { echo "Unexpected sbatch response: $job" >&2; exit 2; }
    fi
    printf '%s\t%s\taip-boyuwang\t%s\t%s\n' "$name" "$job" "$missing" "$walltime" >> "$batch/submitted.tsv"
    submitted=$((submitted+1))
done < "$batch/jobs.tsv"
printf 'DONE: %s jobs %s. Receipt: %s\n' "$submitted" "$([[ $dry_run == 1 ]] && echo planned || echo submitted)" "$batch/submitted.tsv"
