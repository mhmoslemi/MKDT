#!/bin/bash
# 54 ordinary full-data SSL jobs: one seed/job, 49 Wang / 5 Lu. No evaluation or arrays.
set -euo pipefail

if [[ ${1:-} == --worker ]]; then
    shift
    [[ $# == 5 ]] || { echo 'Expected --worker BENCH_DIR TASK VENV DATA_ROOT OUTPUT_ROOT.' >&2; exit 2; }
    : "${SLURM_JOB_ID:?The worker must run through Slurm}"
    [[ ${SLURM_CLUSTER_NAME:-${CC_CLUSTER:-}} == vulcan ]] || {
        echo 'This worker is for Vulcan only.' >&2; exit 2;
    }
    # Slurm executes a spool copy: never resolve repository paths from it.
    bench_dir=$1 task=$2 venv=$3 data_root=$4 output_root=$5
    [[ $bench_dir == /* && -d $bench_dir ]] || { echo 'Expected an absolute BENCH_DIR.' >&2; exit 2; }
    bench_dir=$(cd -- "$bench_dir" && pwd -P)
    for required in "$bench_dir/full_ssl_jobs/worker.py" "$bench_dir/full_ssl_jobs/tasks.tsv" \
        "$bench_dir/protocol.json" "$venv/bin/activate"; do
        [[ -r $required ]] || { echo "Missing worker input: $required" >&2; exit 2; }
    done
    module --force purge
    module load StdEnv/2023 python/3.11.5
    source "$venv/bin/activate"
    export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
    export HF_HOME="$output_root/cache/hf" XDG_CACHE_HOME="$output_root/cache/xdg"
    export TORCH_HOME="$output_root/cache/torch" TMPDIR="${SLURM_TMPDIR:?}"
    cd "$bench_dir"
    printf 'JOB=%s WORKER=%s\n' "$SLURM_JOB_ID" "$bench_dir/full_ssl_jobs/worker.py"
    nvidia-smi -L
    exec python "$bench_dir/full_ssl_jobs/worker.py" "$task" "$data_root" "$output_root"
fi

bench_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
launcher="$bench_dir/full_ssl_vulcan.sh"
manifest="$bench_dir/full_ssl_jobs/tasks.tsv"
case ${1:---submit} in
    --submit) dry_run=0 ;;
    --dry-run) dry_run=1 ;;
    --help|-h)
        printf '%s\n' \
            'bash Evaluate_benchamrk/full_ssl_vulcan.sh            # submit on Vulcan' \
            'bash Evaluate_benchamrk/full_ssl_vulcan.sh --dry-run  # print only, no writes' \
            '54 jobs / 54 encoders: 3 datasets x 3 BN backbones x 2 SSL methods x seeds 0,1,2.' \
            '100% TRAIN data, 1000 epochs, batch 256; 1 L40S, 1 CPU, 6G per job.' \
            '49 Wang / 5 Lu (~90%/10%). No arrays, probes, or data generation.' \
            'Replace training.pt every 50 epochs; retain it and final encoder.pt.' \
            'Rerun to submit unfinished jobs; workers resume from training.pt.' \
            'Walltimes: measured Vulcan subset step rates projected to full data + 15min.' \
            'Optional paths: BENCH_VENV, BENCH_DATA_ROOT, BENCH_OUTPUT_ROOT.'
        exit 0 ;;
    *) echo 'Use --submit, --dry-run, or --help.' >&2; exit 2 ;;
esac
[[ $# -le 1 && -z ${SLURM_JOB_ID:-} ]] || { echo 'Run with bash on the login node, not sbatch.' >&2; exit 2; }
venv=${BENCH_VENV:-$HOME/ENV}
data_root=${BENCH_DATA_ROOT:-$bench_dir/../../data}
output_root=${BENCH_OUTPUT_ROOT:-$bench_dir/full_ssl_checkpoints_1000ep_3seeds}
for required in "$manifest" "$bench_dir/full_ssl_jobs/worker.py" "$bench_dir/common.py" \
    "$bench_dir/data.py" "$bench_dir/train_ssl.py" "$bench_dir/run.py" "$bench_dir/protocol.json" \
    "$bench_dir/../networks.py" "$bench_dir/../benchmarks/downstream_data.py"; do
    [[ -r $required ]] || { echo "Missing code: $required" >&2; exit 2; }
done
# Exact coverage and times; no Python or training on the login node.
awk -F '\t' '
    NR==1 {next}
    {jobs++; if(task[$1]++ || ($2!="CIFAR10" && $2!="CIFAR100" && $2!="TinyImageNet") ||
        ($3!="ConvNet" && $3!="VGG11" && $3!="ResNet18") ||
        ($4!="simclr" && $4!="barlowtwins") || $5!~/^[012]$/ || $6!=1000 ||
        ($10!="aip-boyuwang" && $10!="aip-yiweilu") || seed[$2 FS $3 FS $4 FS $5]++)bad=1;
     if($10=="aip-boyuwang")wang++;else if($10=="aip-yiweilu")lu++;
     split($9,d,"-"); days=0;clock=$9;if(length(d)==2){days=d[1];clock=d[2]}
     split(clock,t,":"); seconds=days*86400+t[1]*3600+t[2]*60+t[3];
     if($8!=$7+900 || seconds!=$8 || seconds>604800)bad=1}
    END {if(bad || jobs!=54 || wang!=49 || lu!=5)exit 2}
' "$manifest" || { echo 'Invalid full-data task manifest.' >&2; exit 2; }
printf 'Full-data SSL only: 54 single-seed jobs, 49 Wang / 5 Lu; 1000 epochs; checkpoint every 50.\n'
printf 'Walltimes are full-data projections from Vulcan logs + 15min, not measured full-data runtimes.\n'
if (( dry_run )); then
    sed -n '1,55p' "$manifest"
    printf 'Checkpoint output: %s\nDry run: nothing submitted or written.\n' "$output_root"
    exit 0
fi

cluster=$(scontrol show config | awk '$1 == "ClusterName" {print $3}')
[[ $cluster == vulcan ]] || { echo 'Submit on Vulcan only.' >&2; exit 2; }
[[ -r $venv/bin/activate && -d $data_root ]] || { echo 'Missing venv or source data root.' >&2; exit 2; }
venv=$(cd -- "$venv" && pwd -P)
data_root=$(cd -- "$data_root" && pwd -P)
output_root=$(realpath -m -- "$output_root")
[[ $output_root == "$bench_dir/"* && $output_root != "$data_root" && $output_root != "$data_root/"* ]] || {
    echo 'Keep checkpoints inside Evaluate_benchamrk, outside the data directory.' >&2; exit 2;
}
for marker in cifar-10-batches-py/data_batch_{1..5} cifar-10-batches-py/batches.meta \
    cifar-100-python/train cifar-100-python/meta tinyimagenet.pt; do
    [[ -s $data_root/$marker ]] || { echo "Missing training data: $data_root/$marker" >&2; exit 2; }
done
module spider python/3.11.5 >/dev/null 2>&1
inventory=$(sinfo -h -o '%G' --Node)
printf '%s\n' "$inventory" | tr ',' '\n' | grep -Eq '^gpu:l40s:' || { echo 'L40S GRES unavailable.' >&2; exit 2; }
accounts=$(sacctmgr -nP show assoc "user=$(id -un)" format=Account)
for account in aip-boyuwang aip-yiweilu; do
    printf '%s\n' "$accounts" | cut -d'|' -f1 | grep -Fxq "$account" || {
        echo "Account unavailable: $account" >&2; exit 2;
    }
done
if command -v diskusage_report >/dev/null 2>&1; then diskusage_report; fi
mkdir -p "$output_root/logs"
exec 9>"$output_root/.submit.lock"
flock -n 9 || { echo 'Another full-data launcher is running.' >&2; exit 2; }
declare -A active
queue=$(squeue -u "$(id -un)" -h -o '%j')
while read -r name; do [[ -z $name ]] || active[$name]=1; done <<< "$queue"
receipt="$output_root/submitted_$(date -u +%Y%m%dT%H%M%SZ)_$$.tsv"
printf 'job_id\ttask\tsource\tmodel\tssl\tseed\tepochs\taccount\twalltime\n' > "$receipt"
submitted=0
while IFS=$'\t' read -r task source_name model ssl seed epochs estimated requested walltime account; do
    [[ $task != task ]] || continue
    name="full_ref_${source_name}_${model}_${ssl}_e${epochs}_s${seed}"
    if [[ -n ${active[$name]:-} ]]; then printf 'SKIP %s: matching job is active.\n' "$task"; continue; fi
    printf -v seed_dir 'seed_%03d' "$seed"
    folder="$output_root/$source_name/$model/full/pt100/$ssl/$seed_dir"
    if [[ -s $folder/config.json && -s $folder/encoder.pt && -s $folder/ssl_complete.json ]]; then
        printf 'SKIP %s: final encoder and marker exist.\n' "$task"; continue
    fi
    job=$(sbatch --parsable --job-name="$name" --account="$account" \
        --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=6G --gres=gpu:l40s:1 \
        --time="$walltime" --chdir="$bench_dir" \
        --output="$output_root/logs/%x-%j.out" --error="$output_root/logs/%x-%j.err" \
        "$launcher" --worker "$bench_dir" "$task" "$venv" "$data_root" "$output_root")
    job=${job%%;*}
    [[ $job =~ ^[0-9]+$ ]] || { echo "Unexpected sbatch response: $job" >&2; exit 2; }
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$job" "$task" "$source_name" "$model" "$ssl" "$seed" "$epochs" "$account" "$walltime" >> "$receipt"
    printf 'SUBMITTED %s: %s/%s/%s seed=%s, %s, %s\n' "$job" "$source_name" "$model" "$ssl" "$seed" "$account" "$walltime"
    submitted=$((submitted+1))
done < "$manifest"
printf 'Submitted %s full-data SSL jobs. Receipt: %s\nCheckpoints: %s\n' "$submitted" "$receipt" "$output_root"
