#!/bin/bash
# Missing Random downstream cells: CIFAR100 2%; TinyImageNet 0.2%, 1%, 2%.
# One ResNet18BN encoder seed and ONE table-cell probe per ordinary job.
set -euo pipefail

if [[ ${1:-} == --worker ]]; then
    shift
    [[ $# == 6 ]] || { echo 'Invalid worker arguments.' >&2; exit 2; }
    bench_dir=$1 route=$2 target=$3 labels=$4 venv=$5 data_root=$6
    : "${SLURM_JOB_ID:?Submit the worker through this launcher}"
    module --force purge
    module load StdEnv/2023 python/3.11.5
    source "$venv/bin/activate"
    export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
    export HF_HOME="$bench_dir/random_downstream_jobs/cache/hf"
    export XDG_CACHE_HOME="$bench_dir/random_downstream_jobs/cache/xdg"
    export TORCH_HOME="$bench_dir/random_downstream_jobs/cache/torch" TMPDIR="${SLURM_TMPDIR:?}"
    cd "$bench_dir"
    exec python "$bench_dir/random_downstream_jobs/worker.py" "$route" "$target" "$labels" "$data_root"
fi

dry_run=0
if [[ ${1:-} == --dry-run ]]; then dry_run=1; shift; fi
[[ $# == 0 && -z ${SLURM_JOB_ID:-} ]] || { echo 'Run with bash on the Killarney login node.' >&2; exit 2; }
bench_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
launcher="$bench_dir/submit_random_downstream_lu_killarney.sh"
run_dir="$bench_dir/random_downstream_jobs"
venv=${BENCH_VENV:-$HOME/ENV}
data_root=${BENCH_DATA_ROOT:-$bench_dir/../../data}
account=aip-yiweilu
[[ -r $venv/bin/activate && -r $run_dir/cells.tsv && -r $run_dir/worker.py && -d $data_root ]] || {
    echo 'Missing environment, cell manifest, or dataset directory.' >&2; exit 2;
}
venv=$(cd -- "$venv" && pwd -P)
data_root=$(cd -- "$data_root" && pwd -P)
cluster=$(scontrol show config | awk '$1=="ClusterName" {print $3}')
[[ $cluster == killarney ]] || { echo 'Submit on Killarney only.' >&2; exit 2; }
inventory=$(sinfo -h -o '%G' --Node)
[[ $inventory == *gpu:l40s:* ]] || { echo 'L40S GPU type not found.' >&2; exit 2; }
accounts=$(sacctmgr -nP show assoc "user=$(id -un)" format=Account)
awk -F '|' -v a="$account" '$1==a {found=1} END {exit !found}' <<< "$accounts" || {
    echo "Account unavailable: $account" >&2; exit 2;
}
module spider python/3.11.5 >/dev/null 2>&1
command -v jq >/dev/null || { echo 'jq is required.' >&2; exit 2; }
for marker in cifar-10-batches-py/data_batch_1 cifar-10-batches-py/test_batch \
    cifar-100-python/train cifar-100-python/test tinyimagenet.pt \
    fgvc-aircraft-2013b/data/variants.txt CUB_200_2011/images.txt \
    StanfordDogs/train_list.mat Oxford102Flowers/setid.mat; do
    [[ -r $data_root/$marker ]] || { echo "Missing dataset: $data_root/$marker" >&2; exit 2; }
done

probe_done() {
    local target=$1 labels=$2
    local config="$bench_dir/$checkpoint_dir/config.json"
    local marker="$bench_dir/$checkpoint_dir/ssl_complete.json"
    local result="$bench_dir/$output_dir/evaluations/${target}_lbl${labels}.json"
    [[ -s $config && -s $marker && -s $result ]] || return 1
    jq -e -s --arg source "$source_name" --arg ssl "$ssl" --arg target "$target" \
        --argjson subset "$subset" --argjson seed "$seed" --argjson labels "$labels" \
        --argjson epochs "$epochs" '
        .[0] as $c | .[1] as $m | .[2] as $r |
        $c.signature.source == $source and $c.signature.model == "ResNet18" and
        $c.signature.subset == $subset and $c.signature.ssl == $ssl and
        $c.signature.selection_method == "random" and $c.identity.seed == $seed and
        $c.signature.protocol.ssl.subset_epochs == $epochs and
        $m.complete == true and $m.identity == $c.identity and $m.epochs == $epochs and
        $r.complete == true and $r.encoder_unchanged == true and
        $r.encoder_sha256 == $m.encoder_sha256 and $r.expected.identity == $c.identity and
        $r.expected.source == $source and $r.expected.target == $target and
        $r.expected.seed == $seed and $r.expected.label_percentage == $labels and
        $r.expected.evaluation == $c.signature.protocol.evaluation and
        $r.expected.evaluation_code_hashes == $c.signature.code_hashes and
        ($r.accuracy_percent | type == "number" and . >= 0 and . <= 100)
        ' "$config" "$marker" "$result" >/dev/null 2>&1
}

mkdir -p "$run_dir/logs" "$run_dir/routes"
exec 9>"$run_dir/.submit.lock"
flock -n 9 || { echo 'Another copy of this launcher is running.' >&2; exit 2; }
queue=$(squeue -h -u "$(id -un)" -o '%200j|%i' | awk -F '|' '{gsub(/ /,"",$1); print $1 "|" $2}')
receipt="$run_dir/submitted_$(date -u +%Y%m%dT%H%M%SZ)_$$.tsv"
printf 'cell_seed\tjob\taccount\tcheckpoint\tssl_epochs\twalltime\tdependency\n' > "$receipt"
submitted=0 skipped=0 training=0
for source_name in CIFAR100 TinyImageNet; do
    sizes=(2)
    [[ $source_name != TinyImageNet ]] || sizes=(0.2 1 2)
    for subset in "${sizes[@]}"; do
        for ssl in simclr barlowtwins; do
            for seed in {0..9}; do
                printf -v seed_dir 'seed_%03d' "$seed"
                relative="$source_name/ResNet18/random/pt$subset/$ssl/$seed_dir"
                key="${source_name}_p${subset/./p}_${ssl}_s${seed}"
                route="$run_dir/routes/$key.json"
                if [[ ! -s $route ]]; then
                    # Freeze one encoder choice per seed so all twelve cells use the same weights.
                    checkpoint_dir="random_ssl_checkpoints_cifar10k8k6k_tiny6k5k4k/$relative"
                    output_dir="random_eval_results_interactive_10seeds/$relative"
                    mode=existing
                    if [[ -s $bench_dir/$checkpoint_dir/encoder.pt ]]; then
                        epochs=$(jq -er '.signature.protocol.ssl.subset_epochs' "$bench_dir/$checkpoint_dir/config.json")
                    else
                        mode=fallback
                        checkpoint_dir="random_downstream_jobs/fallback_checkpoints/$relative"
                        output_dir="random_downstream_jobs/fallback_results/$relative"
                        epochs=2000
                        [[ $ssl != barlowtwins ]] || epochs=10000
                    fi
                    temporary="$run_dir/routes/.$key.$$.tmp"
                    jq -n --arg source "$source_name" --argjson subset "$subset" --arg ssl "$ssl" \
                        --argjson seed "$seed" --argjson epochs "$epochs" --arg mode "$mode" \
                        --arg checkpoint_dir "$checkpoint_dir" --arg output_dir "$output_dir" \
                        '{source:$source,subset:$subset,ssl:$ssl,seed:$seed,epochs:$epochs,
                          mode:$mode,checkpoint_dir:$checkpoint_dir,output_dir:$output_dir}' > "$temporary"
                    mv -- "$temporary" "$route"
                fi
                IFS=$'\t' read -r checkpoint_dir output_dir epochs mode < <(
                    jq -r '[.checkpoint_dir,.output_dir,.epochs,.mode] | @tsv' "$route")
                jq -e --arg source "$source_name" --arg ssl "$ssl" --argjson subset "$subset" --argjson seed "$seed" \
                    '.source==$source and .ssl==$ssl and .subset==$subset and .seed==$seed and
                     (.epochs|type=="number" and .>=1)' "$route" >/dev/null || {
                    echo "Encoder route belongs to another seed: $route" >&2; exit 2;
                }
                [[ $checkpoint_dir != /* && $checkpoint_dir != *..* && $output_dir != /* && $output_dir != *..* ]] || {
                    echo "Unsafe route: $route" >&2; exit 2;
                }
                needs_ssl=0 resume=0 bootstrap=
                if [[ ! -s $bench_dir/$checkpoint_dir/encoder.pt ]]; then
                    [[ $mode == fallback ]] || { echo "Saved encoder disappeared: $checkpoint_dir/encoder.pt" >&2; exit 2; }
                    needs_ssl=1
                    if [[ -s $bench_dir/$checkpoint_dir/training.pt && -s $bench_dir/$checkpoint_dir/ssl.jsonl ]]; then
                        logged=$(tail -n 5 "$bench_dir/$checkpoint_dir/ssl.jsonl" | jq -Rr 'fromjson? | .epoch // empty' | tail -n 1)
                        resume=$(( ${logged:-0} / 500 * 500 ))
                    fi
                fi
                while IFS=$'\t' read -r cell_source cell_subset cell_ssl target labels probe_seconds rate; do
                    [[ $cell_source == "$source_name" && $cell_subset == "$subset" && $cell_ssl == "$ssl" ]] || continue
                    if probe_done "$target" "$labels"; then
                        skipped=$((skipped + 1)); continue
                    fi
                    name="rnd_down_${key}_${target}_l${labels}"
                    queued=$(awk -F '|' -v name="$name" '$1==name {print $2; exit}' <<< "$queue")
                    if [[ -n $queued ]]; then
                        if (( needs_ssl )) && [[ -z $bootstrap ]]; then bootstrap=$queued; fi
                        skipped=$((skipped + 1)); continue
                    fi
                    remaining=0
                    dependencies=()
                    if (( needs_ssl )); then
                        if [[ -z $bootstrap ]]; then
                            remaining=$((epochs - resume))
                            (( remaining >= 0 )) || { echo 'Fallback resume exceeds requested epochs.' >&2; exit 2; }
                            training=$((training + 1))
                        else
                            dependencies=(--dependency="afterok:$bootstrap" --kill-on-invalid-dep=yes)
                        fi
                    fi
                    seconds=$(awk -v e="$remaining" -v r="$rate" -v p="$probe_seconds" 'BEGIN {x=e*r+p+900; print int(x)+(x>int(x))}')
                    printf -v walltime '%02d:%02d:%02d' "$((seconds/3600))" "$((seconds/60%60))" "$((seconds%60))"
                    printf '%s: time=%s, saved/fallback SSL=%s epochs, Lu/L40S/1 CPU/6G\n' "$name" "$walltime" "$epochs"
                    job=DRY_RUN
                    if (( ! dry_run )); then
                        job=$(sbatch --parsable --job-name="$name" --account="$account" \
                            --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=6G --gres=gpu:l40s:1 \
                            --exclude=kn101 --time="$walltime" --chdir="$bench_dir" \
                            --output="$run_dir/logs/%x-%j.out" --error="$run_dir/logs/%x-%j.err" \
                            "${dependencies[@]}" "$launcher" --worker "$bench_dir" "$route" "$target" "$labels" "$venv" "$data_root")
                        job=${job%%;*}
                        [[ $job =~ ^[0-9]+$ ]] || { echo "Unexpected sbatch response: $job" >&2; exit 2; }
                    fi
                    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$name" "$job" "$account" "$checkpoint_dir" "$epochs" "$walltime" "${bootstrap:--}" >> "$receipt"
                    if (( needs_ssl )) && [[ -z $bootstrap ]]; then bootstrap=$job; fi
                    submitted=$((submitted + 1))
                done < "$run_dir/cells.tsv"
            done
        done
    done
done
printf 'DONE: %s jobs %s; %s complete/queued skipped; %s include missing-encoder training.\nReceipt: %s\n' \
    "$submitted" "$([[ $dry_run == 1 ]] && echo planned || echo submitted)" "$skipped" "$training" "$receipt"
