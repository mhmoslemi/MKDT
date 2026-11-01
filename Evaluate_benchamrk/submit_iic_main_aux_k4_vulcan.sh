#!/bin/bash
set -euo pipefail
if [[ ${1:-} == --worker ]]; then
    shift
    [[ $# == 3 ]] || exit 2
    bench=$1 manifest=$2 stage=$3
    : "${SLURM_JOB_ID:?Submit through Slurm}"
    module --force purge
    module load StdEnv/2023 python/3.11.5
    source /home/mmoslem3/ENV/bin/activate
    export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
    export XDG_CACHE_HOME="${SLURM_TMPDIR:?}/cache"
    export TORCH_HOME="$XDG_CACHE_HOME/torch" HF_HOME="$XDG_CACHE_HOME/hf" MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
    export TMPDIR="$SLURM_TMPDIR"
    cd "$bench"
    exec python -u iic_init/distill_main_aux.py --job "$manifest" --stage "$stage"
fi
dry=0
if [[ ${1:-} == --dry-run ]]; then dry=1; shift; fi
[[ $# == 0 && -z ${SLURM_JOB_ID:-} ]] || exit 2
bench=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
run="$bench/iic_init/main_aux_k4_10k"
data=${BENCH_DATA_ROOT:-/scratch/mmoslem3/data}
[[ -r /home/mmoslem3/ENV/bin/activate ]] || exit 2
[[ $(scontrol show config | awk '$1=="ClusterName" {print $3}') == vulcan ]] || exit 2
inventory=$(sinfo -h -o '%G' --Node | sort -u)
[[ $inventory == *gpu:l40s:* ]] || exit 2
available=$(sacctmgr -nP show assoc "user=$(id -un)" format=Account)
for account in aip-boyuwang aip-yiweilu; do
    awk -F '|' -v account="$account" '$1==account {found=1} END {exit !found}' <<< "$available"
done
module spider python/3.11.5 >/dev/null 2>&1
if ! command -v node >/dev/null; then
    module spider nodejs/20.16.0 >/dev/null 2>&1
    module load StdEnv/2023 nodejs/20.16.0
fi
mkdir -p "$run/jobs" "$run/logs"
exec 9>"$run/.submit.lock"
flock -n 9 || { echo 'Another main/aux submitter is running.' >&2; exit 2; }
node "$bench/iic_init/main_aux_plan.js" "$run" "$data" > "$run/plan.tsv"
declare -A ids=()
while IFS='|' read -r name job; do
    name=${name// /}; [[ -z $name ]] || ids[$name]=${job// /}
done < <(squeue -h -u "$(id -un)" -o '%200j|%i')
receipt="$run/submitted_$(date -u +%Y%m%dT%H%M%SZ).tsv"
printf 'dataset\tsize\tstage\tjob\taccount\twalltime\tparent\n' > "$receipt"
while IFS=$'\t' read -r source tag account generation_seconds evaluation_seconds manifest generation_done evaluation_done; do
    parent=''
    for stage in generate evaluate; do
        name="iic4_${source}_p${tag}_${stage}"
        if [[ $stage == generate && $generation_done == 1 || $stage == evaluate && $evaluation_done == 1 ]]; then
            printf 'SKIP complete: %s\n' "$name"
            continue
        fi
        if [[ -n ${ids[$name]:-} ]]; then
            [[ $stage != generate ]] || parent=${ids[$name]}
            printf 'Already queued: %s (%s)\n' "$name" "${ids[$name]}"
            continue
        fi
        seconds=$generation_seconds
        [[ $stage != evaluate ]] || seconds=$evaluation_seconds
        printf -v wall '%02d:%02d:%02d' "$((seconds/3600))" "$((seconds/60%60))" "$((seconds%60))"
        dependency=()
        if [[ $stage == evaluate && -n $parent && $parent != DRY_RUN ]]; then
            dependency=("--dependency=afterok:$parent" --kill-on-invalid-dep=yes)
        fi
        job=DRY_RUN
        if (( ! dry )); then
            job=$(sbatch --parsable --job-name="$name" --account="$account" --nodes=1 --ntasks=1 \
                --cpus-per-task=1 --mem=6G --gres=gpu:l40s:1 --time="$wall" --chdir="$bench" \
                --output="$run/logs/%x-%j.out" --error="$run/logs/%x-%j.err" "${dependency[@]}" \
                "$bench/submit_iic_main_aux_k4_vulcan.sh" --worker "$bench" "$manifest" "$stage")
            job=${job%%;*}; [[ $job =~ ^[0-9]+$ ]] || exit 2
        fi
        ids[$name]=$job
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$source" "$tag" "$stage" "$job" "$account" "$wall" "${parent:--}" >> "$receipt"
        printf '%s: %s, %s, %s\n' "$name" "$job" "$account" "$wall"
        [[ $stage != generate ]] || parent=$job
    done
done < "$run/plan.tsv"
printf 'Receipt: %s\nRerun this script to skip completed stages and resume interrupted ones.\n' "$receipt"
