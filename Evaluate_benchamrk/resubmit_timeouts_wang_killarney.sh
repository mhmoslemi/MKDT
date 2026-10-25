#!/bin/bash
# Retry only the original 40 expert shards and five SSL timeouts; no probes/distillation.
set -euo pipefail
dry_run=0
if [[ ${1:-} == --dry-run ]]; then dry_run=1; shift; fi
[[ $# == 0 && -z ${SLURM_JOB_ID:-} ]] || { echo 'Run with bash on the Killarney login node.' >&2; exit 2; }
bench=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
repo=$(cd -- "$bench/.." && pwd -P)
work="$bench/mkdt_02_jobs/runs"
downstream="$bench/downstream_finish_jobs"
run="$bench/timeout_retries"
venv=${BENCH_VENV:-$HOME/ENV}
data=${BENCH_DATA_ROOT:-$repo/../data}
synthetic=${MKDT_SYNTH_ROOT:-$repo/benchmarks/mkdt_multidataset/results/batch_20261010T160046Z_Y6dqZS/synthetic}
[[ -r $venv/bin/activate && -d $data && -d $synthetic ]] || { echo 'Missing environment, data or MKDT synthetic root.' >&2; exit 2; }
venv=$(cd -- "$venv" && pwd -P)
data=$(cd -- "$data" && pwd -P)
synthetic=$(cd -- "$synthetic" && pwd -P)
cluster=$(scontrol show config | awk '$1=="ClusterName" {print $3}')
[[ $cluster == killarney ]] || { echo 'Submit on Killarney only.' >&2; exit 2; }
inventory=$(sinfo -h -o '%G' --Node)
[[ $inventory == *gpu:l40s:* ]] || { echo 'L40S GPU type not found.' >&2; exit 2; }
accounts=$(sacctmgr -nP show assoc "user=$(id -un)" format=Account)
awk -F '|' '$1=="aip-boyuwang" {found=1} END {exit !found}' <<< "$accounts" || { echo 'Wang account unavailable.' >&2; exit 2; }
module spider python/3.11.5 >/dev/null 2>&1
command -v jq >/dev/null || { echo 'jq is required.' >&2; exit 2; }
queue=$(squeue -h -u "$(id -un)" -t PD,R,CF,CG -o '%200j' | awk '{$1=$1;print}')
queued() { awk -v name="$1" '$0==name {found=1} END {exit !found}' <<< "$queue"; }

cd "$repo"
sources=(buffer.py sort_dset.py distill.py utils.py networks.py reparam_module.py
    Evaluate_benchamrk/mkdt_02_jobs/worker.py Evaluate_benchamrk/mkdt_02_jobs/check_resume.py)
recipe=$(sha256sum "${sources[@]}" | sha256sum | awk '{print $1}')
[[ -s $work/recipe.sha256 && $(< "$work/recipe.sha256") == "$recipe" ]] || {
    echo 'Generation recipe changed or is missing; existing expert buffers were preserved.' >&2; exit 2;
}
for marker in cifar-10-batches-py/data_batch_1 cifar-100-python/train tinyimagenet.pt; do
    [[ -r $data/$marker ]] || { echo "Missing dataset: $data/$marker" >&2; exit 2; }
done

declare -a expert_datasets=() expert_starts=() ssl_names=() ssl_manifests=() ssl_minutes=()
skipped=0
for dataset in CIFAR10 CIFAR100; do
    folder="$work/$dataset"
    [[ -s $synthetic/work/target_rep/barlow_twins/${dataset}_target_rep_train.pt ]] || {
        echo "Missing saved teacher representations: $dataset" >&2; exit 2;
    }
    for start in $(seq 0 5 95); do
        name="mk02_${dataset}_exp_${start}"
        marker="$folder/shard_$(printf '%03d' "$start").json"
        if queued "$name" || { [[ -s $marker ]] && jq -e '.complete == true' "$marker" >/dev/null; }; then
            echo "SKIP completed/queued: $name"; skipped=$((skipped+1)); continue
        fi
        saved=0
        for ((index=start;index<start+5;index++)); do
            if [[ -s $folder/buffers/$dataset/ConvNet/replay_buffer_${index}.pt ]]; then saved=$((saved+1)); fi
        done
        (( saved >= 4 )) || { echo "$name: expected at least four saved experts; found $saved. No restart from zero." >&2; exit 2; }
        expert_datasets+=("$dataset")
        expert_starts+=("$start")
    done
done

while read -r name minutes; do
    if queued "$name" || queued "mainlu_${name#dfssl_}"; then
        echo "SKIP queued training: $name"; skipped=$((skipped+1)); continue
    fi
    manifest=
    for candidate in "$downstream"/batches/batch_*/"$name.json"; do
        if [[ -s $candidate ]]; then manifest=$candidate; break; fi
    done
    [[ -n $manifest ]] || { echo "Missing original SSL manifest: $name" >&2; exit 2; }
    jq -e --arg name "$name" '.name == $name and .stage == "ssl" and
        .unit.source == "TinyImageNet" and .unit.model == "ResNet18" and
        .unit.ssl == "simclr" and .unit.epochs == 2000' "$manifest" >/dev/null
    relative=$(jq -er '.unit.checkpoint_dir' "$manifest")
    [[ $relative != /* && $relative != *..* ]] || { echo 'Invalid checkpoint path.' >&2; exit 2; }
    folder="$bench/$relative"
    if [[ -s $folder/encoder.pt && -s $folder/ssl_complete.json ]] &&
        jq -e '.complete == true and .epochs == 2000' "$folder/ssl_complete.json" >/dev/null; then
        echo "SKIP completed SSL: $name"; skipped=$((skipped+1)); continue
    fi
    [[ -s $folder/training.pt && -s $folder/config.json && -s $folder/ssl.jsonl ]] || {
        echo "Missing saved SSL state: $folder/training.pt. No restart from zero." >&2; exit 2;
    }
    logged=$(tail -n 10 "$folder/ssl.jsonl" | jq -Rr 'fromjson? | .epoch // empty' | tail -n 1)
    [[ $logged =~ ^[0-9]+$ ]] && (( logged >= 1500 && logged <= 2000 )) || {
        echo "Unexpected SSL progress for $name: $logged; retry time assumes checkpoint 1500 or later." >&2; exit 2;
    }
    ssl_names+=("$name")
    ssl_manifests+=("$manifest")
    ssl_minutes+=("$minutes")
done <<'SSL_RETRIES'
dfssl_mkdt_TinyImageNet_p1_simclr_s0 26
dfssl_mkdt_TinyImageNet_p1_simclr_s1 26
dfssl_ours_TinyImageNet_p1_simclr_s1 25
dfssl_mkdt_TinyImageNet_p2_simclr_s0 59
dfssl_mkdt_TinyImageNet_p2_simclr_s1 59
SSL_RETRIES

mkdir -p "$run"
exec 9>"$run/.submit.lock"
flock -n 9 || { echo 'Another copy of this retry submitter is running.' >&2; exit 2; }
batch=$(mktemp -d "$run/batch_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")
mkdir "$batch/code"
cp "${sources[@]}" "$batch/code/"
cp "$bench/submit_mkdt_02_wang_killarney.sh" "$batch/generation.sh"
receipt="$batch/submitted.tsv"
printf 'name\tjob\taccount\twalltime\n' > "$receipt"
submitted=0
submit() {
    local name=$1 minutes=$2 logs=$3 walltime job
    shift 3
    printf -v walltime '00:%02d:00' "$minutes"
    echo "$name: $walltime, Wang/L40S/1 CPU/6G"
    job=DRY_RUN
    if (( ! dry_run )); then
        job=$(sbatch --parsable --job-name="$name" --account=aip-boyuwang \
            --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=6G --gres=gpu:l40s:1 \
            --exclude=kn101 --time="$walltime" --chdir="$repo" \
            --output="$logs/%x-%j.out" --error="$logs/%x-%j.err" "$@")
        job=${job%%;*}
        [[ $job =~ ^[0-9]+$ ]] || { echo "Unexpected sbatch response: $job" >&2; exit 2; }
    fi
    printf '%s\t%s\taip-boyuwang\t%s\n' "$name" "$job" "$walltime" >> "$receipt"
    submitted=$((submitted+1))
}
mkdir -p "$work/logs" "$downstream/logs"
for i in "${!expert_datasets[@]}"; do
    dataset=${expert_datasets[$i]} start=${expert_starts[$i]}
    submit "mk02_${dataset}_exp_${start}" 7 "$work/logs" "$batch/generation.sh" \
        --worker "$batch/code" experts "$dataset" "$start" "$work" "$synthetic" "$data" "$venv"
done
for i in "${!ssl_names[@]}"; do
    submit "${ssl_names[$i]}" "${ssl_minutes[$i]}" "$downstream/logs" \
        "$bench/submit_remaining_downstream_wang_killarney.sh" \
        --worker "$bench" "${ssl_manifests[$i]}" "$venv" "$data" "$synthetic"
done
printf 'DONE: %s retries %s; %s complete/queued skipped. Receipt: %s\n' \
    "$submitted" "$([[ $dry_run == 1 ]] && echo planned || echo submitted)" "$skipped" "$receipt"
