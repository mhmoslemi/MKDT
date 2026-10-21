#!/bin/bash
# Replace only the 14 pasted Lu jobs on Killarney. Cancel originals separately.
set -euo pipefail

dry_run=0
if [[ ${1:-} == --dry-run && $# == 1 ]]; then
    dry_run=1
elif (( $# )); then
    echo 'Usage: bash Evaluate_benchamrk/resubmit_selected_mkdt_wang_killarney.sh [--dry-run]' >&2
    exit 2
fi
[[ -z ${SLURM_JOB_ID:-} ]] || { echo 'Run with bash on the login node.' >&2; exit 2; }
bench_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
launcher="$bench_dir/mkdt_ssl_killarney.sh"
manifest="$bench_dir/mkdt_ssl_jobs/tasks.tsv"
account=aip-boyuwang
# Original IDs -> unchanged manifest tasks:
# 6368305->045 6368385->125 6368320->060 6368315->055 6368330->070
# 6368355->095 6368345->085 6368350->090 6368375->115 6368380->120
# 6368370->110 6368400->140 6368404->144 6368395->135
selected=$(awk -F '\t' '
    BEGIN {split("045 125 060 055 070 095 085 090 115 120 110 140 144 135", tasks, " ");
           for (i in tasks) wanted[tasks[i]]=1}
    NR>1 && ($1 in wanted) {if (seen[$1]++ || $12!="aip-yiweilu") bad=1; print; jobs++; seeds+=$8}
    END {if (bad || jobs!=14 || seeds!=18) exit 2}
' "$manifest") || { echo 'Selected MKDT tasks changed or are missing; no jobs submitted.' >&2; exit 2; }

if (( dry_run )); then
    awk -F '\t' -v account="$account" '{print $1, $2, $3, $5"%", $6,
        "seeds=" $7 ".." ($7+$8-1), "epochs=" $9, "time=" $11, account}' <<< "$selected"
    echo '14 selected jobs / 18 encoder seeds. No scheduler calls or writes.'
    exit 0
fi
cluster=$(scontrol show config | awk '$1=="ClusterName" {print $3}')
[[ $cluster == killarney ]] || { echo 'Submit on Killarney only.' >&2; exit 2; }
inventory=$(sinfo -h -o '%G' --Node)
printf '%s\n' "$inventory" | tr ',' '\n' | grep -Eq '^gpu:l40s:' || {
    echo 'L40S GPUs are unavailable.' >&2; exit 2;
}
accounts=$(sacctmgr -nP show assoc "user=$(id -un)" format=Account)
printf '%s\n' "$accounts" | cut -d '|' -f 1 | grep -Fxq "$account" || {
    echo "Account unavailable: $account" >&2; exit 2;
}
venv=${BENCH_VENV:-$HOME/ENV}
data_root=${BENCH_DATA_ROOT:-$bench_dir/../../data}
generated_root=${MKDT_GENERATED_ROOT:-$bench_dir/../benchmarks/mkdt_multidataset/results/batch_20261010T160046Z_Y6dqZS/synthetic/generated}
output_root=${BENCH_OUTPUT_ROOT:-$bench_dir/mkdt_ssl_checkpoints_cifar8k6k_tiny5k4k_6seeds}
[[ -r $launcher && -r $venv/bin/activate && -d $data_root && -d $generated_root ]] || {
    echo 'Missing launcher, environment, source data or synthetic directory.' >&2; exit 2;
}
venv=$(cd -- "$venv" && pwd -P)
data_root=$(cd -- "$data_root" && pwd -P)
generated_root=$(cd -- "$generated_root" && pwd -P)
output_root=$(realpath -m -- "$output_root")
[[ $output_root == "$bench_dir/"* && $output_root != "$generated_root" && $output_root != "$generated_root/"* ]] || {
    echo 'Keep checkpoint output inside Evaluate_benchamrk, outside the synthetic data.' >&2; exit 2;
}
while IFS=$'\t' read -r task source_name model generator subset ssl start count epochs seconds walltime old_account images; do
    [[ -s $generated_root/$images ]] || { echo "Missing synthetic images: $generated_root/$images" >&2; exit 2; }
done <<< "$selected"

mkdir -p "$output_root/logs"
exec 9>"$output_root/.submit.lock"
flock -n 9 || { echo 'Another MKDT launcher is running.' >&2; exit 2; }
queue=$(squeue -h -u "$(id -un)" -o '%j')
receipt="$output_root/resubmitted_wang_$(date -u +%Y%m%dT%H%M%SZ)_$$.tsv"
printf 'job_id\ttask\taccount\n' > "$receipt"
submitted=0
while IFS=$'\t' read -r task source_name model generator subset ssl start count epochs seconds walltime old_account images; do
    end=$((start+count-1))
    name="mkdt_ref_${source_name}_${model}_${subset}_${ssl}_e${epochs}_s${start}-${end}"
    if grep -Fxq -- "$name" <<< "$queue"; then
        printf 'SKIP task %s: matching job is still active.\n' "$task"
        continue
    fi
    complete=1
    for ((seed=start; seed<=end; seed++)); do
        printf -v seed_dir 'seed_%03d' "$seed"
        folder="$output_root/$source_name/$model/mkdt/pt$subset/$ssl/$seed_dir"
        [[ -s $folder/config.json && -s $folder/encoder.pt && -s $folder/ssl_complete.json ]] || complete=0
    done
    if (( complete )); then printf 'SKIP task %s: completed checkpoints exist.\n' "$task"; continue; fi
    job_id=$(sbatch --parsable --job-name="$name" --account="$account" \
        --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=6G --gres=gpu:l40s:1 \
        --time="$walltime" --chdir="$bench_dir" \
        --output="$output_root/logs/%x-%j.out" --error="$output_root/logs/%x-%j.err" \
        "$launcher" --worker "$bench_dir" "$task" "$venv" "$data_root" "$generated_root" "$output_root")
    job_id=${job_id%%;*}
    [[ $job_id =~ ^[0-9]+$ ]] || { echo "Unexpected sbatch response: $job_id" >&2; exit 2; }
    printf '%s\t%s\t%s\n' "$job_id" "$task" "$account" >> "$receipt"
    printf 'Submitted task %s as %s on %s (%s).\n' "$task" "$job_id" "$account" "$walltime"
    submitted=$((submitted+1))
done <<< "$selected"
printf 'Submitted %s jobs. Receipt: %s\n' "$submitted" "$receipt"
