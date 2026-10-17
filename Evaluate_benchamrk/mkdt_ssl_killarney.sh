#!/bin/bash
# SSL only on existing MKDT images: no distillation, evaluation or submission arrays.
set -euo pipefail

if [[ ${1:-} == --worker ]]; then
    shift
    [[ $# == 6 ]] || { echo 'Expected --worker BENCH_DIR TASK VENV DATA_ROOT GENERATED_ROOT OUTPUT_ROOT.' >&2; exit 2; }
    : "${SLURM_JOB_ID:?This worker must run through Slurm}"
    [[ ${SLURM_CLUSTER_NAME:-${CC_CLUSTER:-}} == killarney ]] || {
        echo 'This worker is for Killarney only.' >&2; exit 2;
    }
    # Slurm runs a spool copy of this script. Its BASH_SOURCE directory is NOT
    # the repository; use the absolute path passed by the submission branch.
    bench_dir=$1 task=$2 venv=$3 data_root=$4 generated_root=$5 output_root=$6
    [[ $bench_dir == /* && -d $bench_dir ]] || {
        echo 'Worker BENCH_DIR must be an existing absolute repository directory.' >&2; exit 2;
    }
    bench_dir=$(cd -- "$bench_dir" && pwd -P)
    for required in "$bench_dir/mkdt_ssl_jobs/worker.py" "$bench_dir/mkdt_ssl_jobs/tasks.tsv" \
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
    printf 'JOB=%s GPU_ALLOCATION=%s CUDA_VISIBLE_DEVICES=%s\n' \
        "$SLURM_JOB_ID" "${SLURM_JOB_GPUS:-unset}" "${CUDA_VISIBLE_DEVICES:-unset}"
    printf 'MKDT worker: %s\n' "$bench_dir/mkdt_ssl_jobs/worker.py"
    nvidia-smi -L
    exec python "$bench_dir/mkdt_ssl_jobs/worker.py" \
        "$task" "$data_root" "$generated_root" "$output_root"
fi

# Only the submit-side script location is used to discover the repository.
bench_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
launcher="$bench_dir/mkdt_ssl_killarney.sh"
manifest="$bench_dir/mkdt_ssl_jobs/tasks.tsv"

case ${1:---submit} in
    --submit) dry_run=0 ;;
    --dry-run) dry_run=1 ;;
    --help|-h)
        printf '%s\n' \
            'bash Evaluate_benchamrk/mkdt_ssl_killarney.sh --dry-run  # preview, no writes' \
            'bash Evaluate_benchamrk/mkdt_ssl_killarney.sh            # submit ON KILLARNEY' \
            '144 ordinary jobs / 216 retained encoders; 72 two-seed + 72 single-seed jobs.' \
            '3 sources x 3 BN backbones x 1%/2% x SimCLR/Barlow Twins x 6 SSL seeds.' \
            'Existing MKDT synthetic images only; no synthesis, label files, synlr files, or probes.' \
            'Generator mapping: ConvNet -> ConvNet/ResNet18, VGG11 -> VGG11.' \
            'No 0.2% MKDT artifact was found in generation scripts; 5% table rows are disabled.' \
            'Epochs/seed: CIFAR10/CIFAR100 1%=8000, 2%=6000; Tiny 1%=5000, 2%=4000.' \
            '1 L40S, 1 CPU, 6G; 115 Wang / 29 Lu (~80/20, as for Killarney random).' \
            'Overwrite training.pt every 500 epochs; retain it and the final encoder.pt.' \
            'Estimated SSL work + 15min/job. Two-seed projections >5h split into single seeds.' \
            'Single seeds can still exceed 5h; no epoch reduction or staged training.' \
            'MKDT_GENERATED_ROOT overrides the directory containing ConvNet/ and VGG11/.' \
            'BENCH_VENV, BENCH_DATA_ROOT, BENCH_OUTPUT_ROOT override environment/data/output.' \
            'MKDT_SSL_TIME=HH:MM:SS optionally overrides all requested walltimes.' \
            'Reruns skip matching live/finished jobs; unfinished seeds resume from training.pt.'
        exit 0 ;;
    *) echo 'Use --submit, --dry-run, or --help.' >&2; exit 2 ;;
esac
[[ $# -le 1 && -z ${SLURM_JOB_ID:-} ]] || {
    echo 'Run with bash on the login node, not sbatch.' >&2; exit 2;
}
generated_root=${MKDT_GENERATED_ROOT:-$bench_dir/../benchmarks/mkdt_multidataset/results/batch_20261010T160046Z_Y6dqZS/synthetic/generated}
output_root=${BENCH_OUTPUT_ROOT:-$bench_dir/mkdt_ssl_checkpoints_cifar8k6k_tiny5k4k_6seeds}
data_root=${BENCH_DATA_ROOT:-$bench_dir/../../data}
venv=${BENCH_VENV:-$HOME/ENV}
for required in "$manifest" "$bench_dir/mkdt_ssl_jobs/worker.py" "$bench_dir/common.py" \
    "$bench_dir/data.py" "$bench_dir/train_ssl.py" "$bench_dir/run.py" "$bench_dir/evaluate.py" \
    "$bench_dir/protocol.json" "$bench_dir/../networks.py" "$bench_dir/../benchmarks/downstream_data.py"; do
    [[ -r $required ]] || { echo "Missing code: $required; sync the new files first." >&2; exit 2; }
done
# Lightweight exact coverage check. No Python or training on the login node.
awk -F '\t' '
    NR==1 {next}
    {jobs++; total+=$8; split($11,t,":");
     if(task[$1]++ || ($2!="CIFAR10" && $2!="CIFAR100" && $2!="TinyImageNet") ||
        ($3!="ConvNet" && $3!="VGG11" && $3!="ResNet18") || ($5!=1 && $5!=2) ||
        ($6!="simclr" && $6!="barlowtwins") || ($8!=1 && $8!=2) || $7<0 || $7+$8>6 ||
        ($8==2 && $7%2) || $4!=($3=="VGG11"?"VGG11":"ConvNet") ||
        $9!=($2=="TinyImageNet"?($5==1?5000:4000):($5==1?8000:6000)) ||
        t[1]*3600+t[2]*60+t[3]!=$10+900) bad=1;
     if($12=="aip-yiweilu")lu++; else if($12!="aip-boyuwang")bad=1;
     for(s=$7;s<$7+$8;s++)if(seen[$2 FS $3 FS $5 FS $6 FS s]++)bad=1}
    END {if(bad || jobs!=144 || total!=216 || lu!=29)exit 2}
' "$manifest" || { echo 'Invalid MKDT SSL manifest.' >&2; exit 2; }
printf 'MKDT SSL only: 144 jobs / 216 encoders; six seeds, 1%% and 2%%; Killarney.\n'
printf 'Generator: ConvNet for ConvNet/ResNet18; VGG11 for VGG11. No synthetic regeneration.\n'
printf 'Walltimes: matching current-trainer random-subset estimates + 15min; MKDT throughput unmeasured.\n'
printf 'task\tdataset\tmodel\tgenerator\tsize\tssl\tseeds\tepochs\taccount\twalltime\n'
if (( dry_run )); then
    while IFS=$'\t' read -r task source_name model generator subset ssl start count epochs seconds walltime account relative; do
        [[ $task != task ]] || continue
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s..%s\t%s\t%s\t%s\n' \
            "$task" "$source_name" "$model" "$generator" "$subset" "$ssl" "$start" "$((start+count-1))" \
            "$epochs" "$account" "${MKDT_SSL_TIME:-$walltime}"
    done < "$manifest"
    printf 'Synthetic input root: %s\nCheckpoint output: %s\nDry run: nothing submitted or written.\n' "$generated_root" "$output_root"
    exit 0
fi

cluster=$(scontrol show config | awk '$1 == "ClusterName" {print $3}')
[[ $cluster == killarney ]] || { echo 'Submit this launcher on Killarney only.' >&2; exit 2; }
[[ -r $venv/bin/activate && -d $data_root && -d $generated_root ]] || {
    echo 'Missing venv, source data, or MKDT generated directory. Check the path overrides in --help.' >&2; exit 2;
}
venv=$(cd -- "$venv" && pwd -P)
data_root=$(cd -- "$data_root" && pwd -P)
generated_root=$(cd -- "$generated_root" && pwd -P)
output_root=$(realpath -m -- "$output_root")
[[ $output_root == "$bench_dir/"* && $output_root != "$generated_root" && $output_root != "$generated_root/"* ]] || {
    echo 'Use a dedicated output directory inside Evaluate_benchamrk, outside the generated data.' >&2; exit 2;
}
missing=0
while read -r relative; do
    if [[ ! -s $generated_root/$relative ]]; then
        printf 'MISSING SYNTHETIC IMAGES: %s\n' "$generated_root/$relative" >&2
        missing=$((missing+1))
    fi
done < <(awk -F '\t' 'NR>1 {print $13}' "$manifest" | sort -u)
(( missing==0 )) || { echo "$missing missing MKDT image files; no jobs submitted." >&2; exit 2; }
for marker in cifar-10-batches-py/data_batch_{1..5} cifar-10-batches-py/batches.meta \
    cifar-100-python/train cifar-100-python/meta tinyimagenet.pt; do
    [[ -r $data_root/$marker ]] || { echo "Missing source data for provenance: $data_root/$marker" >&2; exit 2; }
done
module spider python/3.11.5 >/dev/null 2>&1
inventory=$(sinfo -h -o '%G' --Node)
printf '%s\n' "$inventory" | tr ',' '\n' | grep -Eq '^gpu:l40s:' || {
    echo 'L40S GRES is not available.' >&2; exit 2;
}
accounts=$(sacctmgr -nP show assoc "user=$(id -un)" format=Account)
for account in aip-boyuwang aip-yiweilu; do
    printf '%s\n' "$accounts" | cut -d'|' -f1 | grep -Fxq -- "$account" || {
        echo "Account unavailable: $account" >&2; exit 2;
    }
done
if command -v diskusage_report >/dev/null 2>&1; then diskusage_report; fi
mkdir -p "$output_root/logs"
exec 9>"$output_root/.submit.lock"
flock -n 9 || { echo 'Another MKDT SSL launcher is running.' >&2; exit 2; }
declare -A active
queue=$(squeue -u "$(id -un)" -h -o '%j')
while read -r name; do [[ -z $name ]] || active[$name]=1; done <<< "$queue"
receipt="$output_root/submitted_$(date -u +%Y%m%dT%H%M%SZ)_$$.tsv"
printf 'job_id\ttask\taccount\tdataset\tmodel\tgenerator\tsubset\tssl\tseed_start\tseed_count\tepochs\twalltime\timages\n' > "$receipt"
submitted=0
while IFS=$'\t' read -r task source_name model generator subset ssl start count epochs seconds walltime account relative; do
    [[ $task != task ]] || continue
    end=$((start+count-1))
    name="mkdt_ref_${source_name}_${model}_${subset}_${ssl}_e${epochs}_s${start}-${end}"
    if [[ -n ${active[$name]:-} ]]; then
        printf 'SKIP %s: matching job is active.\n' "$task"; continue
    fi
    complete=1
    for ((seed=start; seed<=end; seed++)); do
        printf -v seed_dir 'seed_%03d' "$seed"
        folder="$output_root/$source_name/$model/mkdt/pt$subset/$ssl/$seed_dir"
        [[ -s $folder/config.json && -s $folder/encoder.pt && -s $folder/ssl_complete.json ]] || complete=0
    done
    if (( complete )); then printf 'SKIP %s: all final encoders and markers exist.\n' "$task"; continue; fi
    walltime=${MKDT_SSL_TIME:-$walltime}
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s..%s\t%s\t%s\t%s\n' \
        "$task" "$source_name" "$model" "$generator" "$subset" "$ssl" "$start" "$end" "$epochs" "$account" "$walltime"
    job_id=$(sbatch --parsable --job-name="$name" --account="$account" \
        --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=6G --gres=gpu:l40s:1 \
        --time="$walltime" --chdir="$bench_dir" \
        --output="$output_root/logs/%x-%j.out" --error="$output_root/logs/%x-%j.err" \
        "$launcher" --worker "$bench_dir" "$task" "$venv" "$data_root" "$generated_root" "$output_root")
    job_id=${job_id%%;*}
    [[ $job_id =~ ^[0-9]+$ ]] || { echo "Unexpected sbatch response: $job_id" >&2; exit 2; }
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$job_id" "$task" "$account" "$source_name" "$model" "$generator" "$subset" "$ssl" \
        "$start" "$count" "$epochs" "$walltime" "$generated_root/$relative" >> "$receipt"
    submitted=$((submitted+1))
done < "$manifest"
printf 'Submitted %s MKDT SSL jobs. IDs: %s\nCheckpoints: %s\n' "$submitted" "$receipt" "$output_root"
