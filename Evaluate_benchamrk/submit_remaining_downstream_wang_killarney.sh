#!/bin/bash
# Fill only empty cells of the three requested downstream tables; leave old partial cells alone.
set -euo pipefail

if [[ ${1:-} == --worker ]]; then
    shift
    [[ $# == 5 ]] || { echo 'Invalid worker arguments.' >&2; exit 2; }
    bench=$1 manifest=$2 venv=$3 data=$4 synthetic=$5
    : "${SLURM_JOB_ID:?Submit through this launcher}"
    module --force purge
    module load StdEnv/2023 python/3.11.5
    source "$venv/bin/activate"
    export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
    export HF_HOME="$bench/downstream_finish_jobs/cache/hf"
    export XDG_CACHE_HOME="$bench/downstream_finish_jobs/cache/xdg"
    export TORCH_HOME="$bench/downstream_finish_jobs/cache/torch" TMPDIR="${SLURM_TMPDIR:?}"
    cd "$bench"
    python -u downstream_finish_jobs/worker.py "$manifest" "$data" "$synthetic/generated" | awk -f eval_jobs/ssl_console.awk
    exit
fi

dry_run=0
skip02=()
for argument in "$@"; do
    case "$argument" in
        --dry-run) dry_run=1 ;;
        --skip-mkdt02) skip02=(--skip-mkdt02) ;;
        *) echo "Unknown argument: $argument" >&2; exit 2 ;;
    esac
done
[[ -z ${SLURM_JOB_ID:-} ]] || { echo 'Run the submitter on the Killarney login node.' >&2; exit 2; }
bench=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
repo=$(cd -- "$bench/.." && pwd -P)
run="$bench/downstream_finish_jobs"
venv=${BENCH_VENV:-$HOME/ENV}
data=${BENCH_DATA_ROOT:-$repo/../data}
synthetic=${MKDT_SYNTH_ROOT:-$repo/benchmarks/mkdt_multidataset/results/batch_20261010T160046Z_Y6dqZS/synthetic}
[[ -r $venv/bin/activate && -d $data && -d $synthetic ]] || { echo 'Missing environment, data or MKDT synthetic root.' >&2; exit 2; }
venv=$(cd -- "$venv" && pwd -P)
data=$(cd -- "$data" && pwd -P)
synthetic=$(cd -- "$synthetic" && pwd -P)
export MKDT_SYNTH_ROOT="$synthetic"
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
    [[ -r $data/$marker ]] || { echo "Missing dataset: $data/$marker" >&2; exit 2; }
done
mkdir -p "$run/batches" "$run/logs"
exec 9>"$run/.submit.lock"
flock -n 9 || { echo 'Another copy of this submitter is running.' >&2; exit 2; }
batch=$(mktemp -d "$run/batches/batch_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")
queue_file="$batch/queue.json"
refresh_queue() {
    squeue -h -u "$(id -un)" -t PD,R,CF,CG -o '%200j|%i' | node -e '
        let text="";process.stdin.on("data",x=>text+=x);process.stdin.on("end",()=>{
            const q={};for(const line of text.trim().split("\n")){const [n,i]=line.split("|").map(x=>x.trim());
                if(n&&/^\d+$/.test(i)&&(!q[n]||Number(i)>Number(q[n])))q[n]=i;}
            require("fs").writeFileSync(process.argv[1],JSON.stringify(q));});' "$queue_file"
}
refresh_queue
mkdir "$batch/preflight"
node "$run/plan.js" "$batch/preflight" "$queue_file" --dry-run "${skip02[@]}"
if (( ! dry_run )); then
    blocked=$(node -e 'console.log(require(process.argv[1]).summary.blocked_generation_jobs.length)' "$batch/preflight/coverage.json")
    if (( blocked )); then
        echo 'MKDT 0.2% is not generated yet; submitting/reusing its existing Wang generation pipeline first.'
        bash "$bench/submit_mkdt_02_wang_killarney.sh"
        refresh_queue
    fi
    node "$run/plan.js" "$batch" "$queue_file" "${skip02[@]}"
    manifests="$batch/jobs.tsv"
else
    manifests="$batch/preflight/jobs.tsv"
fi
declare -A ids
while IFS=$'\t' read -r name id; do ids[$name]=$id; done < <(
    node -e 'for(const [n,i] of Object.entries(require(process.argv[1])))console.log(n+"\t"+i)' "$queue_file")
printf 'name\tjob\tstage\taccount\twalltime\tdependency\n' > "$batch/submitted.tsv"
submitted=0
while IFS=$'\t' read -r name stage seconds dependencies manifest; do
    if [[ -n ${ids[$name]:-} ]]; then echo "SKIP queued: $name (${ids[$name]})"; continue; fi
    dependency_ids=()
    IFS=',' read -ra dependency_names <<< "$dependencies"
    for parent in "${dependency_names[@]}"; do
        [[ -n $parent && $parent != - ]] || continue
        if [[ -z ${ids[$parent]:-} ]]; then
            if (( dry_run )); then ids[$parent]=999999999;
            else echo "Missing dependency: $parent" >&2; exit 2; fi
        fi
        dependency_ids+=("${ids[$parent]}")
    done
    dependency=$(IFS=:; echo "${dependency_ids[*]}")
    printf -v walltime '%02d:%02d:%02d' "$((seconds/3600))" "$((seconds/60%60))" "$((seconds%60))"
    command=(sbatch --parsable --job-name="$name" --account=aip-boyuwang
        --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=6G --gres=gpu:l40s:1
        --exclude=kn101 --time="$walltime" --chdir="$bench"
        --output="$run/logs/%x-%j.out" --error="$run/logs/%x-%j.err")
    [[ -z $dependency ]] || command+=(--dependency="afterok:$dependency" --kill-on-invalid-dep=yes)
    submitted=$((submitted+1))
    if (( dry_run )); then job=$((900000000+submitted));
    else job=$("${command[@]}" "$bench/submit_remaining_downstream_wang_killarney.sh" \
        --worker "$bench" "$manifest" "$venv" "$data" "$synthetic"); job=${job%%;*}; fi
    [[ $job =~ ^[0-9]+$ ]] || { echo "Unexpected sbatch response: $job" >&2; exit 2; }
    ids[$name]=$job
    printf '%s\t%s\t%s\taip-boyuwang\t%s\t%s\n' "$name" "$job" "$stage" "$walltime" "$dependency" >> "$batch/submitted.tsv"
    printf '%s: %s, Wang/L40S/1 CPU/6G\n' "$name" "$walltime"
done < "$manifests"
printf 'DONE: %s downstream jobs %s. Receipt: %s\n' "$submitted" "$([[ $dry_run == 1 ]] && echo planned || echo submitted)" "$batch/submitted.tsv"
echo 'Original partial cells and all 2% Barlow cells are excluded. Re-run this command to resume only the newly approved work.'
