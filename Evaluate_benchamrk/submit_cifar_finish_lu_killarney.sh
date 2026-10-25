#!/bin/bash
# Run with bash on Killarney. One SSL seed and its main-table probes per job.
set -euo pipefail

if [[ ${1:-} == --worker ]]; then
    shift
    [[ $# == 6 || $# == 7 ]] || { echo 'Invalid worker arguments.' >&2; exit 2; }
    bench_dir=$1 manifest=$2 task=$3 venv=$4 data_root=$5 synthetic_root=$6
    worker_script=${7:-$bench_dir/cifar_finish_jobs/worker.py}
    : "${SLURM_JOB_ID:?Run the worker through the submit script}"
    module --force purge
    module load StdEnv/2023 python/3.11.5
    source "$venv/bin/activate"
    export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
    export HF_HOME="$bench_dir/cifar_finish_jobs/cache/hf"
    export XDG_CACHE_HOME="$bench_dir/cifar_finish_jobs/cache/xdg"
    export TORCH_HOME="$bench_dir/cifar_finish_jobs/cache/torch"
    export TMPDIR="${SLURM_TMPDIR:?}"
    cd "$bench_dir"
    exec python "$worker_script" "$task" --manifest "$manifest" \
        --venv "$venv" --data-root "$data_root" --synthetic-root "$synthetic_root"
fi

dry_run=0
if [[ ${1:-} == --dry-run ]]; then dry_run=1; shift; fi
[[ $# == 0 && -z ${SLURM_JOB_ID:-} ]] || {
    echo 'Run on the Killarney login node: bash Evaluate_benchamrk/submit_cifar_finish_lu_killarney.sh [--dry-run]' >&2
    exit 2
}
bench_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
launcher="$bench_dir/submit_cifar_finish_lu_killarney.sh"
manifest=${CIFAR_FINISH_MANIFEST:-$bench_dir/cifar_finish_jobs/tasks.tsv}
worker_script=${CIFAR_FINISH_WORKER:-$bench_dir/cifar_finish_jobs/worker.py}
max_seconds=${CIFAR_FINISH_MAX_SECONDS:-0}
[[ $max_seconds =~ ^(0|[1-9][0-9]*)$ ]] || { echo 'Invalid job time limit.' >&2; exit 2; }
model_filter=${CIFAR_FINISH_MODEL:-}
[[ -z $model_filter || $model_filter == ResNet18 ]] || { echo 'Unsupported model filter.' >&2; exit 2; }
venv=${BENCH_VENV:-$HOME/ENV}
data_root=${BENCH_DATA_ROOT:-$bench_dir/../../data}
synthetic_root=${MKDT_GENERATED_ROOT:-$bench_dir/../benchmarks/mkdt_multidataset/results/batch_20261010T160046Z_Y6dqZS/synthetic/generated}
account=aip-yiweilu
[[ -r $manifest && -r $worker_script && -r $venv/bin/activate && -d $data_root && -d $synthetic_root ]] || {
    echo 'Missing manifest, Python environment, source data, or MKDT synthetic root.' >&2; exit 2;
}
venv=$(cd -- "$venv" && pwd -P)
data_root=$(cd -- "$data_root" && pwd -P)
synthetic_root=$(cd -- "$synthetic_root" && pwd -P)
cluster=$(scontrol show config | awk '$1=="ClusterName" {print $3}')
[[ $cluster == killarney ]] || { echo 'Submit on Killarney only.' >&2; exit 2; }
inventory=$(sinfo -h -o '%G' --Node)
[[ $inventory == *gpu:l40s:* ]] || { echo 'L40S GPU type not found.' >&2; exit 2; }
accounts=$(sacctmgr -nP show assoc "user=$(id -un)" format=Account)
awk -F '|' -v a="$account" '$1==a {found=1} END {exit !found}' <<< "$accounts" || {
    echo "Account unavailable: $account" >&2; exit 2;
}
module spider python/3.11.5 >/dev/null 2>&1 || { echo 'python/3.11.5 module unavailable.' >&2; exit 2; }
command -v jq >/dev/null || { echo 'jq is required for reading completion JSONs.' >&2; exit 2; }
for marker in cifar-10-batches-py/data_batch_{1..5} cifar-10-batches-py/test_batch \
              cifar-100-python/train cifar-100-python/test; do
    [[ -r $data_root/$marker ]] || { echo "Missing source data: $data_root/$marker" >&2; exit 2; }
done

# Check all required provenance, not just the existence of a result filename.
probe_done() {
    local path=$1 expected_probe=$seed labels
    labels=${path##*_lbl}; labels=${labels%%[_.]*}
    if [[ $path =~ _probe([012])\.json$ ]]; then expected_probe=${BASH_REMATCH[1]}; fi
    [[ -s $bench_dir/$path && -s $bench_dir/$output_dir/config.json \
       && -s $bench_dir/$checkpoint_dir/ssl_complete.json ]] || return 1
    jq -e -s --arg source "$source_name" --arg kind "$kind" --arg model "$model" \
        --arg ssl "$ssl" --argjson subset "$subset" --argjson seed "$seed" \
        --argjson probe "$expected_probe" --argjson labels "$labels" --argjson epochs "$epochs" '
        .[0] as $c | .[1] as $m | .[2] as $r |
        $c.signature.source == $source and $c.signature.model == $model and
        $c.signature.subset == $subset and $c.signature.ssl == $ssl and
        $c.signature.selection_method == $kind and $c.identity.seed == $seed and
        $c.signature.protocol.ssl.subset_epochs == $epochs and
        $m.complete == true and $m.epochs == $epochs and $m.identity == $c.identity and
        $r.complete == true and $r.encoder_unchanged == true and
        $r.encoder_sha256 == $m.encoder_sha256 and $r.expected.identity == $c.identity and
        $r.expected.source == $source and $r.expected.target == $source and
        $r.expected.seed == $probe and $r.expected.label_percentage == $labels and
        $r.expected.evaluation == $c.signature.protocol.evaluation and
        $r.expected.evaluation_code_hashes == $c.signature.code_hashes and
        ($r.accuracy_percent | type == "number" and . >= 0 and . <= 100)
        ' "$bench_dir/$output_dir/config.json" "$bench_dir/$checkpoint_dir/ssl_complete.json" \
          "$bench_dir/$path" >/dev/null 2>&1
}

saved_epoch() {
    local folder=$1 horizon=$2 interval=500 logged=0 marked
    [[ $kind != random ]] || interval=25
    if [[ -s $folder/ssl_complete.json ]]; then
        marked=$(jq -er '.epochs | select(type=="number")' "$folder/ssl_complete.json")
        [[ $marked =~ ^[0-9]+$ && $marked -le $horizon ]] || return 2
        echo "$marked"; return
    fi
    if [[ -s $folder/ssl.jsonl ]]; then
        logged=$(tail -n 10 "$folder/ssl.jsonl" | jq -Rr 'fromjson? | .epoch // empty' | tail -n 1)
        logged=${logged:-0}
    fi
    [[ $logged =~ ^[0-9]+$ && $logged -le $horizon ]] || return 2
    # The JSONL can run ahead of the rolling checkpoint; round down, never up.
    echo "$((logged / interval * interval))"
}

queue=$(squeue -h -u "$(id -un)" -o '%j')
run_dir="$bench_dir/cifar_finish_jobs"
mkdir -p "$run_dir/logs"
exec 9>"$run_dir/.submit.lock"
flock -n 9 || { echo 'Another copy of this submit script is running.' >&2; exit 2; }
receipt="$run_dir/submitted_$(date -u +%Y%m%dT%H%M%SZ)_$$.tsv"
printf 'task\tjob\taccount\tresume_epoch_estimate\tmissing_probes\twalltime\n' > "$receipt"
submitted=0 skipped=0 blocked=0
while IFS=$'\t' read -r task kind source_name model subset ssl seed epochs checkpoint_dir \
    parent_dir output_dir evaluations probe_paths rate probe1 probe5 scope; do
    [[ $task != task ]] || continue
    [[ $scope == main && $task =~ ^[a-zA-Z0-9_]+$ ]] || { echo 'Invalid manifest row.' >&2; exit 2; }
    [[ -z $model_filter || $model == $model_filter ]] || continue
    [[ $output_dir != - ]] || output_dir=$checkpoint_dir
    name="cifar_finish_lu_$task"
    if awk -v name="$name" '$0==name {found=1} END {exit !found}' <<< "$queue"; then
        echo "SKIP queued: $task"; skipped=$((skipped + 1)); continue
    fi
    missing=0 probe_seconds=0
    IFS=';' read -ra paths <<< "$probe_paths"
    for path in "${paths[@]}"; do
        path="$output_dir/evaluations/$path"
        if ! probe_done "$path"; then
            missing=$((missing + 1))
            if [[ $path == *_lbl1.json || $path == *_lbl1_probe*.json ]]; then
                probe_seconds=$(awk -v a="$probe_seconds" -v b="$probe1" 'BEGIN {print a+b}')
            else
                probe_seconds=$(awk -v a="$probe_seconds" -v b="$probe5" 'BEGIN {print a+b}')
            fi
        fi
    done
    if (( missing == 0 )); then echo "SKIP complete: $task"; skipped=$((skipped + 1)); continue; fi

    folder="$bench_dir/$checkpoint_dir"
    resume=0 missing_state=
    if [[ -s $folder/encoder.pt ]]; then
        resume=$epochs
    elif [[ -s $folder/training.pt ]]; then
        resume=$(saved_epoch "$folder" "$epochs")
        if (( resume == 0 )) && [[ -s $folder/continuation.json ]]; then
            resume=$(jq -er '.resumed_epoch' "$folder/continuation.json")
        fi
    elif [[ $parent_dir != - ]]; then
        parent="$bench_dir/$parent_dir"
        if [[ -s $parent/training.pt ]]; then
            horizon=$(jq -er '.signature.protocol.ssl.subset_epochs' "$parent/config.json")
            resume=$(saved_epoch "$parent" "$horizon")
        elif [[ -s $parent/encoder.pt || -s $parent/ssl_complete.json ]]; then
            missing_state="$parent/training.pt"
            horizon=$(jq -er '.signature.protocol.ssl.subset_epochs' "$parent/config.json")
            resume=$(saved_epoch "$parent" "$horizon")
        fi
    fi
    if [[ $kind != random ]]; then
        if [[ $kind == ours ]]; then
            synthetic="$bench_dir/../OT-SSL-DD/ours_syntehtic/$source_name/${model}BN/pt$subset/res_IIC_${source_name}_${model}BN_${subset}percent_final.pt"
        else
            generator=ConvNet
            [[ $model != VGG11 ]] || generator=VGG11
            synthetic="$synthetic_root/$generator/$source_name/${source_name}_${subset}pct/images_5000.pt"
        fi
        if [[ ! -s $synthetic ]]; then
            echo "BLOCKED $task: missing synthetic data $synthetic" >&2
            blocked=$((blocked + 1)); continue
        fi
    fi
    seconds=$(awk -v e="$epochs" -v r="$resume" -v s="$rate" -v p="$probe_seconds" \
        'BEGIN {x=(e-r)*s+p+900; print int(x)+(x>int(x))}')
    printf -v walltime '%02d:%02d:%02d' "$((seconds / 3600))" "$((seconds / 60 % 60))" "$((seconds % 60))"
    if (( max_seconds && seconds > max_seconds )); then
        echo "SKIP long: $task needs ~$walltime including 15-minute margin."
        skipped=$((skipped + 1)); continue
    fi
    if [[ -n $missing_state ]]; then
        echo "BLOCKED $task: sync full continuation state $missing_state" >&2
        blocked=$((blocked + 1)); continue
    fi
    printf '%s: resume ~%s/%s, %s probes, time=%s, %s, L40S/1 CPU/6G\n' \
        "$task" "$resume" "$epochs" "$missing" "$walltime" "$account"
    job=DRY_RUN
    if (( ! dry_run )); then
        job=$(sbatch --parsable --job-name="$name" --account="$account" \
            --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=6G --gres=gpu:l40s:1 \
            --exclude=kn101 \
            --time="$walltime" --chdir="$bench_dir" \
            --output="$run_dir/logs/%x-%j.out" --error="$run_dir/logs/%x-%j.err" \
            "$launcher" --worker "$bench_dir" "$manifest" "$task" "$venv" "$data_root" "$synthetic_root" "$worker_script")
        job=${job%%;*}
        [[ $job =~ ^[0-9]+$ ]] || { echo "Unexpected sbatch response: $job" >&2; exit 2; }
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$task" "$job" "$account" "$resume" "$missing" "$walltime" >> "$receipt"
    submitted=$((submitted + 1))
done < "$manifest"
printf 'DONE: %s %s, %s skipped, %s blocked. Receipt: %s\n' \
    "$submitted" "$([[ $dry_run == 1 ]] && echo planned || echo submitted)" "$skipped" "$blocked" "$receipt"
(( blocked == 0 ))
