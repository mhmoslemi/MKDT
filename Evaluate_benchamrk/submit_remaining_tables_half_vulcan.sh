#!/bin/bash
# Remaining combined main + CIFAR10/100 downstream cells; half Wang, half Lu.
set -euo pipefail
if [[ ${1:-} == --worker ]]; then
    shift
    [[ $# == 6 ]] || exit 2
    bench=$1 manifest=$2 stage=$3 venv=$4 data=$5 synthetic=$6
    : "${SLURM_JOB_ID:?Submit through this launcher}"
    module --force purge
    module load StdEnv/2023 python/3.11.5 nodejs/20.16.0
    source "$venv/bin/activate"
    export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
    export HF_HOME="$bench/partial_probe_jobs/cache/hf" XDG_CACHE_HOME="$bench/partial_probe_jobs/cache/xdg"
    export TORCH_HOME="$bench/partial_probe_jobs/cache/torch" TMPDIR="${SLURM_TMPDIR:?}"
    cd "$bench"
    if [[ $stage == ssl ]]; then
        python -u partial_probe_jobs/train_missing.py "$manifest" "$data" "$synthetic" | awk -f eval_jobs/ssl_console.awk
    elif [[ $stage == probe ]]; then
        python -u partial_probe_jobs/worker.py "$manifest" "$data"
    else exit 2; fi
    exit
fi
dry_run=0
if [[ ${1:-} == --dry-run ]]; then dry_run=1; shift; fi
[[ $# == 0 && -z ${SLURM_JOB_ID:-} ]] || { echo 'Run with bash on the Vulcan login node.' >&2; exit 2; }
bench=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
venv=${BENCH_VENV:-$HOME/ENV}
data=${BENCH_DATA_ROOT:-$bench/../../data}
synthetic=${MKDT_GENERATED_ROOT:-$bench/../benchmarks/mkdt_multidataset/results/batch_20261010T160046Z_Y6dqZS/synthetic/generated}
[[ -r $venv/bin/activate && -d $data && -d $synthetic ]] || { echo 'Missing environment, data or MKDT synthetic root.' >&2; exit 2; }
venv=$(cd -- "$venv" && pwd -P)
data=$(cd -- "$data" && pwd -P)
synthetic=$(cd -- "$synthetic" && pwd -P)
cluster=$(scontrol show config | awk '$1=="ClusterName" {print $3}')
[[ $cluster == vulcan ]] || { echo 'Vulcan only.' >&2; exit 2; }
inventory=$(sinfo -h -o '%G' --Node)
[[ $inventory == *gpu:l40s:* ]] || { echo 'L40S GPU type not found.' >&2; exit 2; }
accounts=$(sacctmgr -nP show assoc "user=$(id -un)" format=Account)
for account in aip-boyuwang aip-yiweilu; do
    awk -F '|' -v account="$account" '$1==account {found=1} END {exit !found}' <<< "$accounts" || { echo "Account unavailable: $account" >&2; exit 2; }
done
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
flock -n 9 || { echo 'Another probe-table submitter is running.' >&2; exit 2; }
batch=$(mktemp -d "$run/batches/remaining_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")
squeue -h -u "$(id -un)" -t PD,R,CF,CG -o '%200j|%i' | node -e '
    let text="";process.stdin.on("data",x=>text+=x);process.stdin.on("end",()=>{
        const q={};for(const row of text.trim().split("\n")){const [n,i]=row.split("|").map(x=>x.trim());
            if(n&&/^\d+$/.test(i)&&(!q[n]||Number(i)>Number(q[n])))q[n]=i;}
        require("fs").writeFileSync(process.argv[1],JSON.stringify(q));});' "$batch/queue.json"
node "$run/remaining_plan.js" "$batch" "$synthetic" "$batch/queue.json"
declare -A ids
while IFS=$'\t' read -r name id; do ids[$name]=$id; done < <(
    node -e 'for(const [n,i] of Object.entries(require(process.argv[1])))console.log(n+"\t"+i)' "$batch/queue.json")
printf 'name\tjob\tstage\taccount\twalltime\tdependency\n' > "$batch/submitted.tsv"
submitted=0
while IFS=$'\t' read -r name stage account seconds dependencies manifest; do
    [[ -z ${ids[$name]:-} ]] || { echo "SKIP queued: $name"; continue; }
    dependency_ids=()
    IFS=',' read -ra dependency_names <<< "$dependencies"
    for parent in "${dependency_names[@]}"; do
        [[ -n $parent && $parent != - ]] || continue
        [[ -n ${ids[$parent]:-} ]] || { echo "Missing dependency: $parent" >&2; exit 2; }
        dependency_ids+=("${ids[$parent]}")
    done
    dependency=$(IFS=:; echo "${dependency_ids[*]}")
    printf -v walltime '%02d:%02d:%02d' "$((seconds/3600))" "$((seconds/60%60))" "$((seconds%60))"
    command=(sbatch --parsable --job-name="$name" --account="$account" --nodes=1 --ntasks=1
        --cpus-per-task=1 --mem=6G --gres=gpu:l40s:1 --time="$walltime" --chdir="$bench"
        --output="$run/logs/%x-%j.out" --error="$run/logs/%x-%j.err")
    [[ -z $dependency ]] || command+=(--dependency="afterok:$dependency" --kill-on-invalid-dep=yes)
    submitted=$((submitted+1))
    if (( dry_run )); then job=$((900000000+submitted));
    else job=$("${command[@]}" "$bench/submit_remaining_tables_half_vulcan.sh" --worker "$bench" "$manifest" "$stage" "$venv" "$data" "$synthetic");job=${job%%;*};fi
    [[ $job =~ ^[0-9]+$ ]] || { echo "Unexpected sbatch response: $job" >&2; exit 2; }
    ids[$name]=$job
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$name" "$job" "$stage" "$account" "$walltime" "$dependency" >> "$batch/submitted.tsv"
    printf '%s: %s, %s, %s/L40S/1 CPU/6G\n' "$name" "$stage" "$walltime" "$account"
done < "$batch/jobs.tsv"
printf 'DONE: %s jobs %s. Receipt: %s\n' "$submitted" "$([[ $dry_run == 1 ]] && echo planned || echo submitted)" "$batch/submitted.tsv"
echo 'Blocked cells, if any, are listed in coverage.json.'
