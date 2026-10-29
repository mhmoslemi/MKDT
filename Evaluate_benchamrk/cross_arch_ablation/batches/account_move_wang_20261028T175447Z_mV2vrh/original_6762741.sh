#!/bin/bash
set -euo pipefail
if [[ ${1:-} == --worker ]]; then
    shift
    [[ $# == 4 ]] || exit 2
    bench=$1 manifest=$2 venv=$3 data=$4
    : "${SLURM_JOB_ID:?Submit through this launcher}"
    module --force purge
    module load StdEnv/2023 python/3.11.5
    source "$venv/bin/activate"
    export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
    export HF_HOME="$bench/cross_arch_ablation/cache/hf" XDG_CACHE_HOME="$bench/cross_arch_ablation/cache/xdg"
    export TORCH_HOME="$bench/cross_arch_ablation/cache/torch" TMPDIR="${SLURM_TMPDIR:?}"
    cd "$bench"
    exec python -u cross_arch_ablation/worker.py "$manifest" "$data"
fi
dispatch=0
if [[ ${1:-} == --dispatch ]]; then
    [[ $# == 4 && -n ${SLURM_JOB_ID:-} ]] || exit 2
    dispatch=1
    bench=$2 venv=$3 data=$4
    shift 4
    module --force purge
    module load StdEnv/2023 nodejs/20.16.0
fi
dry_run=0
if [[ ${1:-} == --dry-run ]]; then dry_run=1; shift; fi
[[ $# == 0 && ( -z ${SLURM_JOB_ID:-} || $dispatch == 1 ) ]] || { echo 'Run on the Vulcan login node.' >&2; exit 2; }
if (( ! dispatch )); then bench=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P); fi
venv=${BENCH_VENV:-${venv:-$HOME/ENV}}
data=${BENCH_DATA_ROOT:-${data:-$bench/../../data}}
[[ -r $venv/bin/activate && -d $data ]] || { echo 'Missing environment or datasets.' >&2; exit 2; }
venv=$(cd -- "$venv" && pwd -P)
data=$(cd -- "$data" && pwd -P)
[[ $(scontrol show config | awk '$1=="ClusterName" {print $3}') == vulcan ]] || exit 2
inventory=$(sinfo -h -o '%G' --Node)
[[ $inventory == *gpu:l40s:* ]] || { echo 'L40S unavailable.' >&2; exit 2; }
accounts=$(sacctmgr -nP show assoc "user=$(id -un)" format=Account)
awk -F '|' '$1=="aip-boyuwang" {found=1} END {exit !found}' <<< "$accounts" || { echo 'Wang account unavailable.' >&2; exit 2; }
module spider python/3.11.5 >/dev/null 2>&1
module spider nodejs/20.16.0 >/dev/null 2>&1
if ! command -v node >/dev/null; then module load StdEnv/2023 nodejs/20.16.0; fi
for marker in cifar-10-batches-py/data_batch_1 cifar-100-python/train tinyimagenet.pt; do
    [[ -r $data/$marker ]] || { echo "Missing dataset: $data/$marker" >&2; exit 2; }
done
run="$bench/cross_arch_ablation"
mkdir -p "$run/batches" "$run/logs"
exec 9>"$run/.submit.lock"
flock -n 9 || { echo 'Another copy is submitting.' >&2; exit 2; }
if [[ ! -r $run/calibration.json ]]; then
    (( ! dry_run && ! dispatch )) || { echo 'BN/runtime preflight has not completed.' >&2; exit 2; }
    queued=$(squeue -h -u "$(id -un)" -o '%200j|%i')
    preflight=$(awk -F '|' '$1~/^ab_xarch_bn_preflight *$/ {print $2}' <<< "$queued")
    if [[ -z $preflight ]]; then
        preflight=$(sbatch --parsable --job-name=ab_xarch_bn_preflight --account=aip-boyuwang \
            --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=6G --gres=gpu:l40s:1 --time=00:06:00 \
            --chdir="$bench" --output="$run/logs/%x-%j.out" --error="$run/logs/%x-%j.err" \
            "$run/calibrate.sh" "$bench" "$venv" "$data")
        preflight=${preflight%%;*}
    fi
    [[ $preflight =~ ^[0-9]+$ ]] || { echo 'Ambiguous preflight jobs.' >&2; exit 2; }
    submitter=$(awk -F '|' '$1~/^ab_xarch_submit_all_wang *$/ {print $2}' <<< "$queued")
    if [[ -z $submitter ]]; then
        submitter=$(sbatch --parsable --job-name=ab_xarch_submit_all_wang --account=aip-boyuwang \
            --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=1G --time=00:05:00 \
            --dependency="afterok:$preflight" --kill-on-invalid-dep=yes --chdir="$bench" \
            --output="$run/logs/%x-%j.out" --error="$run/logs/%x-%j.err" \
            "$bench/submit_cross_arch_ablation_wang_vulcan.sh" --dispatch "$bench" "$venv" "$data")
        submitter=${submitter%%;*}
    fi
    [[ $submitter =~ ^[0-9]+$ ]] || exit 2
    printf 'preflight\tsubmitter\taccount\n%s\t%s\taip-boyuwang\n' "$preflight" "$submitter" > "$run/deferred_submission.tsv"
    printf 'Queued Wang preflight %s and automatic submitter %s. It submits all 108 experiment jobs after validation/runtime measurement.\n' "$preflight" "$submitter"
    exit 0
fi
batch=$(mktemp -d "$run/batches/batch_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")
squeue -h -u "$(id -un)" -o '%200j|%i' > "$batch/queue.tsv"
node "$run/plan.js" "$batch" "$batch/queue.tsv"
declare -A ids=()
while IFS='|' read -r name id; do name=${name// /}; [[ -z $name ]] || ids[$name]=$id; done < "$batch/queue.tsv"
printf 'name\tjob\taccount\twalltime\tdependency\n' > "$batch/submitted.tsv"
while IFS=$'\t' read -r name seconds dependencies manifest; do
    dependency=()
    if [[ $dependencies != - ]]; then
        IFS=, read -ra parents <<< "$dependencies"
        parents_ids=()
        for parent in "${parents[@]}"; do
            [[ -n ${ids[$parent]:-} ]] || { echo "Missing dependency: $parent" >&2; exit 2; }
            parents_ids+=("${ids[$parent]}")
        done
        joined=$(IFS=:; echo "${parents_ids[*]}")
        dependency=("--dependency=afterok:$joined" --kill-on-invalid-dep=yes)
    fi
    printf -v walltime '%02d:%02d:%02d' "$((seconds/3600))" "$((seconds/60%60))" "$((seconds%60))"
    printf '%s: Wang, %s, 1 L40S / 1 CPU / 6G\n' "$name" "$walltime"
    job=DRY_RUN
    if (( ! dry_run )); then
        job=$(sbatch --parsable --job-name="$name" --account=aip-boyuwang \
            --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=6G --gres=gpu:l40s:1 \
            --time="$walltime" --chdir="$bench" --output="$run/logs/%x-%j.out" --error="$run/logs/%x-%j.err" \
            "${dependency[@]}" "$bench/submit_cross_arch_ablation_wang_vulcan.sh" --worker "$bench" "$manifest" "$venv" "$data")
        job=${job%%;*}
        [[ $job =~ ^[0-9]+$ ]] || { echo "Unexpected job ID: $job" >&2; exit 2; }
    fi
    ids[$name]=$job
    printf '%s\t%s\taip-boyuwang\t%s\t%s\n' "$name" "$job" "$walltime" "${dependency[*]:--}" >> "$batch/submitted.tsv"
done < "$batch/jobs.tsv"
printf 'Receipt: %s\n' "$batch/submitted.tsv"
