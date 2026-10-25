#!/bin/bash
# Finish main cells: under 1h uses Wang, 1h or longer uses Lu; one SSL seed per job.
set -euo pipefail

if [[ ${1:-} == --worker ]]; then
    shift
    [[ $# == 5 ]] || exit 2
    bench_dir=$1 manifest=$2 venv=$3 data_root=$4 synthetic_root=$5
    : "${SLURM_JOB_ID:?Submit through this launcher}"
    module --force purge
    module load StdEnv/2023 python/3.11.5
    source "$venv/bin/activate"
    export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
    export HF_HOME="$bench_dir/main_finish_jobs/cache/hf"
    export XDG_CACHE_HOME="$bench_dir/main_finish_jobs/cache/xdg"
    export TORCH_HOME="$bench_dir/main_finish_jobs/cache/torch" TMPDIR="${SLURM_TMPDIR:?}"
    cd "$bench_dir"
    exec python main_finish_jobs/worker.py "$manifest" "$venv" "$data_root" "$synthetic_root"
fi

dry_run=0
if [[ ${1:-} == --dry-run ]]; then dry_run=1; shift; fi
[[ $# == 0 && -z ${SLURM_JOB_ID:-} ]] || { echo 'Run with bash on the Killarney login node.' >&2; exit 2; }
bench_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
run_dir="$bench_dir/main_finish_jobs"
venv=${BENCH_VENV:-$HOME/ENV}
data_root=${BENCH_DATA_ROOT:-$bench_dir/../../data}
synthetic_root=${MKDT_GENERATED_ROOT:-$bench_dir/../benchmarks/mkdt_multidataset/results/batch_20261010T160046Z_Y6dqZS/synthetic/generated}
[[ -r $venv/bin/activate && -d $data_root && -d $synthetic_root ]] || {
    echo 'Missing Python environment, data directory, or MKDT synthetic root.' >&2; exit 2;
}
venv=$(cd -- "$venv" && pwd -P)
data_root=$(cd -- "$data_root" && pwd -P)
synthetic_root=$(cd -- "$synthetic_root" && pwd -P)
cluster=$(scontrol show config | awk '$1=="ClusterName" {print $3}')
[[ $cluster == killarney ]] || { echo 'Submit on Killarney only.' >&2; exit 2; }
inventory=$(sinfo -h -o '%G' --Node)
[[ $inventory == *gpu:l40s:* ]] || { echo 'L40S GPU type not found.' >&2; exit 2; }
accounts=$(sacctmgr -nP show assoc "user=$(id -un)" format=Account)
for account in aip-boyuwang aip-yiweilu; do
    awk -F '|' -v account="$account" '$1==account {found=1} END {exit !found}' <<< "$accounts" || {
        echo "Account unavailable: $account" >&2; exit 2;
    }
done
module spider python/3.11.5 >/dev/null 2>&1
if ! command -v node >/dev/null; then
    module spider nodejs/20.16.0 >/dev/null 2>&1
    module load StdEnv/2023 nodejs/20.16.0
fi
[[ -r $data_root/tinyimagenet.pt ]] || { echo 'Missing tinyimagenet.pt' >&2; exit 2; }
mkdir -p "$run_dir/batches" "$run_dir/logs"
exec 9>"$run_dir/.submit.lock"
flock -n 9 || { echo 'Another copy of this launcher is running.' >&2; exit 2; }
batch=$(mktemp -d "$run_dir/batches/batch_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")
queue=$(squeue -h -u "$(id -un)" -o '%200j' | awk '{$1=$1;print}')
node "$run_dir/plan.js" "$batch" "$synthetic_root" "$queue"
receipt="$batch/submitted.tsv"
printf 'name\tjob\taccount\tresume_epoch_estimate\tmissing_probes\twalltime\n' > "$receipt"
submitted=0
while IFS=$'\t' read -r name seconds resume missing manifest; do
    account=aip-yiweilu
    if (( seconds < 3600 )); then
        account=aip-boyuwang
    fi
    printf -v walltime '%02d:%02d:%02d' "$((seconds/3600))" "$((seconds/60%60))" "$((seconds%60))"
    printf '%s: resume ~%s, %s probes, %s, %s/L40S/1 CPU/6G\n' "$name" "$resume" "$missing" "$walltime" "$account"
    job=DRY_RUN
    if (( ! dry_run )); then
        job=$(sbatch --parsable --job-name="$name" --account="$account" \
            --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=6G --gres=gpu:l40s:1 \
            --exclude=kn101 --time="$walltime" --chdir="$bench_dir" \
            --output="$run_dir/logs/%x-%j.out" --error="$run_dir/logs/%x-%j.err" \
            "$bench_dir/submit_main_table_lu_killarney.sh" --worker \
            "$bench_dir" "$manifest" "$venv" "$data_root" "$synthetic_root")
        job=${job%%;*}
        [[ $job =~ ^[0-9]+$ ]] || { echo "Unexpected sbatch response: $job" >&2; exit 2; }
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$name" "$job" "$account" "$resume" "$missing" "$walltime" >> "$receipt"
    submitted=$((submitted + 1))
done < "$batch/jobs.tsv"
printf 'DONE: %s jobs %s. Receipt: %s\n' "$submitted" \
    "$([[ $dry_run == 1 ]] && echo planned || echo submitted)" "$receipt"
