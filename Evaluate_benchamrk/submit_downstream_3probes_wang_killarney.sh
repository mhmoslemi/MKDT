#!/bin/bash
# Random + Ours downstream; pack five short three-probe bundles per GPU job.
set -euo pipefail

if [[ ${1:-} == --worker ]]; then
    shift
    [[ $# == 4 ]] || { echo 'Invalid worker arguments.' >&2; exit 2; }
    bench_dir=$1 manifest=$2 venv=$3 data_root=$4
    : "${SLURM_JOB_ID:?Submit through the launcher}"
    module --force purge
    module load StdEnv/2023 python/3.11.5
    source "$venv/bin/activate"
    export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
    export HF_HOME="$bench_dir/downstream_probe_jobs/cache/hf"
    export XDG_CACHE_HOME="$bench_dir/downstream_probe_jobs/cache/xdg"
    export TORCH_HOME="$bench_dir/downstream_probe_jobs/cache/torch" TMPDIR="${SLURM_TMPDIR:?}"
    cd "$bench_dir"
    exec python downstream_probe_jobs/worker.py "$manifest" "$data_root"
fi

dry_run=0
if [[ ${1:-} == --dry-run ]]; then dry_run=1; shift; fi
[[ $# == 0 && -z ${SLURM_JOB_ID:-} ]] || { echo 'Run on the Killarney login node.' >&2; exit 2; }
bench_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
run_dir="$bench_dir/downstream_probe_jobs"
venv=${BENCH_VENV:-$HOME/ENV}
data_root=${BENCH_DATA_ROOT:-$bench_dir/../../data}
[[ -r $venv/bin/activate && -d $data_root ]] || { echo 'Missing environment or data directory.' >&2; exit 2; }
venv=$(cd -- "$venv" && pwd -P)
data_root=$(cd -- "$data_root" && pwd -P)
cluster=$(scontrol show config | awk '$1=="ClusterName" {print $3}')
[[ $cluster == killarney ]] || { echo 'Submit on Killarney only.' >&2; exit 2; }
inventory=$(sinfo -h -o '%G' --Node)
[[ $inventory == *gpu:l40s:* ]] || { echo 'L40S GPU type not found.' >&2; exit 2; }
accounts=$(sacctmgr -nP show assoc "user=$(id -un)" format=Account)
awk -F '|' '$1=="aip-boyuwang" {found=1} END {exit !found}' <<< "$accounts" || { echo 'Wang account unavailable.' >&2; exit 2; }
module spider python/3.11.5 >/dev/null 2>&1
if ! command -v node >/dev/null; then
    module spider nodejs/20.16.0 >/dev/null 2>&1
    module load StdEnv/2023 nodejs/20.16.0
fi
for marker in cifar-10-batches-py/data_batch_1 cifar-10-batches-py/test_batch \
    cifar-100-python/train cifar-100-python/test tinyimagenet.pt \
    fgvc-aircraft-2013b/data/variants.txt CUB_200_2011/images.txt \
    StanfordDogs/train_list.mat Oxford102Flowers/setid.mat; do
    [[ -r $data_root/$marker ]] || { echo "Missing dataset: $data_root/$marker" >&2; exit 2; }
done
mkdir -p "$run_dir/batches" "$run_dir/logs"
exec 9>"$run_dir/.submit.lock"
flock -n 9 || { echo 'Another copy of this launcher is running.' >&2; exit 2; }
batch=$(mktemp -d "$run_dir/batches/batch_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")
queue=$(squeue -h -u "$(id -un)" -o '%200j' | awk '{$1=$1;print}')
node "$run_dir/plan.js" "$batch" "$queue"
receipt="$batch/submitted.tsv"
printf 'name\tjob\taccount\tmissing_probes\twalltime\n' > "$receipt"
submitted=0 skipped=0
while IFS=$'\t' read -r name seconds missing manifest; do
    if awk -v n="$name" '$0==n {found=1} END {exit !found}' <<< "$queue"; then
        echo "SKIP queued: $name"; skipped=$((skipped + 1)); continue
    fi
    printf -v walltime '%02d:%02d:%02d' "$((seconds/3600))" "$((seconds/60%60))" "$((seconds%60))"
    printf '%s: %s missing probes, %s, Wang/L40S/1 CPU/6G\n' "$name" "$missing" "$walltime"
    job=DRY_RUN
    if (( ! dry_run )); then
        job=$(sbatch --parsable --job-name="$name" --account=aip-boyuwang \
            --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=6G --gres=gpu:l40s:1 \
            --exclude=kn101 --time="$walltime" --chdir="$bench_dir" \
            --output="$run_dir/logs/%x-%j.out" --error="$run_dir/logs/%x-%j.err" \
            "$bench_dir/submit_downstream_3probes_wang_killarney.sh" \
            --worker "$bench_dir" "$manifest" "$venv" "$data_root")
        job=${job%%;*}
        [[ $job =~ ^[0-9]+$ ]] || { echo "Unexpected sbatch response: $job" >&2; exit 2; }
    fi
    printf '%s\t%s\taip-boyuwang\t%s\t%s\n' "$name" "$job" "$missing" "$walltime" >> "$receipt"
    submitted=$((submitted + 1))
done < "$batch/jobs.tsv"
printf 'DONE: %s jobs %s; %s queued skipped. Receipt: %s\n' \
    "$submitted" "$([[ $dry_run == 1 ]] && echo planned || echo submitted)" "$skipped" "$receipt"
