#!/bin/bash
# Ours, 2%, ResNet18BN, SimCLR 1k: two SSL seeds, three main probes per label budget.
set -euo pipefail

if [[ ${1:-} == --worker ]]; then
    shift
    [[ $# == 5 ]] || { echo 'Invalid worker arguments.' >&2; exit 2; }
    bench_dir=$1 source_name=$2 seed=$3 venv=$4 data_root=$5
    : "${SLURM_JOB_ID:?Submit this worker through the launcher}"
    module --force purge
    module load StdEnv/2023 python/3.11.5
    source "$venv/bin/activate"
    export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
    export HF_HOME="$bench_dir/ours_eval_results_simclr1k_2ssl3probes_main/cache/hf"
    export XDG_CACHE_HOME="$bench_dir/ours_eval_results_simclr1k_2ssl3probes_main/cache/xdg"
    export TORCH_HOME="$bench_dir/ours_eval_results_simclr1k_2ssl3probes_main/cache/torch"
    export TMPDIR="${SLURM_TMPDIR:?}"
    cd "$bench_dir"
    exec python - "$bench_dir" "$source_name" "$seed" "$data_root" <<'PY'
import argparse
import functools
import importlib.util
import os
from pathlib import Path
import sys

bench, source, seed, data_root = sys.argv[1:]
bench, seed = Path(bench).resolve(), int(seed)
if source not in ('CIFAR10', 'CIFAR100', 'TinyImageNet') or seed not in (0, 1):
    raise ValueError('Expected one of the three datasets and SSL seed 0 or 1')
if (os.environ.get('SLURM_CLUSTER_NAME', os.environ.get('CC_CLUSTER')) != 'killarney'
        or os.environ.get('SLURM_JOB_ACCOUNT') != 'aip-boyuwang'
        or int(os.environ.get('WORLD_SIZE', '1')) != 1):
    raise RuntimeError('Use one Wang-account GPU job on Killarney')

# Reuse the same trainer, saved-image loader, frozen probes and six-probe summaries.
spec = importlib.util.spec_from_file_location('ours_1k_base', bench / 'cifar_finish_jobs/worker.py')
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
if not base.torch.cuda.is_available():
    raise RuntimeError('CUDA unavailable; no CPU fallback')
base.torch.set_num_threads(1)
base.torch.set_num_interop_threads(1)
base.torch.backends.cuda.matmul.allow_tf32 = False
base.torch.backends.cudnn.allow_tf32 = False

folder = (bench / 'ours_eval_results_simclr1k_2ssl3probes_main' / source
          / 'ResNet18BN/pt2/simclr' / f'ssl_seed_{seed:03d}')
task = {'kind': 'ours', 'source': source, 'model': 'ResNet18', 'subset': 2,
        'ssl': 'simclr', 'seed': seed, 'epochs': 1000, 'parent_dir': None,
        'checkpoint_dir': folder, 'output_dir': folder}
args = argparse.Namespace(data_root=Path(data_root).resolve())

@functools.lru_cache(maxsize=1)
def datasets(name):
    return (base.RawDataset(args.data_root, name, True),
            base.RawDataset(args.data_root, name, False))

print(f'OURS / MAIN ONLY: {source}, ResNet18BN, 2%, SimCLR 1000 epochs, SSL seed {seed}', flush=True)
print('Six 100-epoch frozen probes in this job: three at 1% labels and three at 5%.', flush=True)
base.run_synthetic(task, args, datasets)
print('COMPLETE:', folder, flush=True)
PY
fi

dry_run=0
if [[ ${1:-} == --dry-run ]]; then dry_run=1; shift; fi
[[ $# == 0 && -z ${SLURM_JOB_ID:-} ]] || { echo 'Run with bash on the Killarney login node.' >&2; exit 2; }
bench_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
launcher="$bench_dir/submit_ours_2pct_simclr1k_wang_killarney.sh"
output="$bench_dir/ours_eval_results_simclr1k_2ssl3probes_main"
venv=${BENCH_VENV:-$HOME/ENV}
data_root=${BENCH_DATA_ROOT:-$bench_dir/../../data}
account=aip-boyuwang
[[ -r $venv/bin/activate && -r $bench_dir/cifar_finish_jobs/worker.py && -d $data_root ]] || {
    echo 'Missing environment, benchmark worker, or dataset directory.' >&2; exit 2;
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
              cifar-100-python/train cifar-100-python/test tinyimagenet.pt; do
    [[ -r $data_root/$marker ]] || { echo "Missing data: $data_root/$marker" >&2; exit 2; }
done
for source_name in CIFAR10 CIFAR100 TinyImageNet; do
    synthetic="$bench_dir/../OT-SSL-DD/ours_syntehtic/$source_name/ResNet18BN/pt2/res_IIC_${source_name}_ResNet18BN_2percent_final.pt"
    [[ -s $synthetic ]] || { echo "Missing synthetic dataset: $synthetic" >&2; exit 2; }
done

probe_done() {
    local labels=$1 probe=$2
    local result="$folder/evaluations/${source_name}_lbl${labels}_probe${probe}.json"
    [[ -s $folder/config.json && -s $folder/ssl_complete.json && -s $result ]] || return 1
    jq -e -s --arg source "$source_name" --argjson seed "$seed" \
        --argjson labels "$labels" --argjson probe "$probe" '
        .[0] as $c | .[1] as $m | .[2] as $r |
        $c.signature.source==$source and $c.signature.model=="ResNet18" and
        $c.signature.selection_method=="ours" and $c.signature.subset==2 and
        $c.signature.ssl=="simclr" and $c.signature.protocol.ssl.subset_epochs==1000 and
        $c.identity.seed==$seed and $m.complete==true and $m.epochs==1000 and
        $m.identity==$c.identity and $r.complete==true and $r.encoder_unchanged==true and
        $r.encoder_sha256==$m.encoder_sha256 and $r.expected.identity==$c.identity and
        $r.expected.source==$source and $r.expected.target==$source and
        $r.expected.seed==$probe and $r.expected.label_percentage==$labels and
        $r.expected.evaluation==$c.signature.protocol.evaluation and
        $r.expected.evaluation_code_hashes==$c.signature.code_hashes and
        ($r.accuracy_percent | type=="number" and .>=0 and .<=100)
        ' "$folder/config.json" "$folder/ssl_complete.json" "$result" >/dev/null 2>&1
}

mkdir -p "$output/logs"
exec 9>"$output/.submit.lock"
flock -n 9 || { echo 'Another copy of this launcher is running.' >&2; exit 2; }
queue=$(squeue -h -u "$(id -un)" -o '%200j' | awk '{$1=$1;print}')
receipt="$output/submitted_$(date -u +%Y%m%dT%H%M%SZ)_$$.tsv"
printf 'source\tssl_seed\tjob\taccount\tresume_epoch_estimate\tmissing_probes\twalltime\n' > "$receipt"
submitted=0 skipped=0
for source_name in CIFAR10 CIFAR100 TinyImageNet; do
    # Measured 90th-percentile seconds; Tiny uses matching Random SSL/k-means main-probe logs.
    case "$source_name" in
        CIFAR10) rate=1.414253351; probe1=20.004329045; probe5=91.979042199 ;;
        CIFAR100) rate=1.404461195; probe1=19.843281462; probe5=90.647869924 ;;
        TinyImageNet) rate=6.331872196; probe1=50.820172051; probe5=240.666351137 ;;
    esac
    for seed in 0 1; do
        printf -v seed_dir 'ssl_seed_%03d' "$seed"
        folder="$output/$source_name/ResNet18BN/pt2/simclr/$seed_dir"
        name="ours1k_wang_${source_name}_s${seed}"
        if awk -v n="$name" '$0==n {found=1} END {exit !found}' <<< "$queue"; then
            echo "SKIP queued: $name"; skipped=$((skipped + 1)); continue
        fi
        missing=0 probe_seconds=0
        for labels in 1 5; do
            for probe in 0 1 2; do
                if ! probe_done "$labels" "$probe"; then
                    missing=$((missing + 1))
                    cost=$probe1
                    [[ $labels == 1 ]] || cost=$probe5
                    probe_seconds=$(awk -v a="$probe_seconds" -v b="$cost" 'BEGIN {print a+b}')
                fi
            done
        done
        if (( missing == 0 )); then echo "SKIP complete: $name"; skipped=$((skipped + 1)); continue; fi
        resume=0
        if [[ -s $folder/encoder.pt ]]; then
            resume=1000
        elif [[ -s $folder/training.pt && -s $folder/ssl.jsonl ]]; then
            logged=$(tail -n 5 "$folder/ssl.jsonl" | jq -Rr 'fromjson? | .epoch // empty' | tail -n 1)
            resume=$(( ${logged:-0} / 500 * 500 ))
        fi
        (( resume <= 1000 )) || { echo 'Checkpoint exceeds the 1000-epoch horizon.' >&2; exit 2; }
        seconds=$(awk -v e="$((1000-resume))" -v r="$rate" -v p="$probe_seconds" \
            'BEGIN {x=e*r+p+900;print int(x)+(x>int(x))}')
        printf -v walltime '%02d:%02d:%02d' "$((seconds/3600))" "$((seconds/60%60))" "$((seconds%60))"
        printf '%s: resume ~%s/1000, %s probes, time=%s, Wang/L40S/1 CPU/6G\n' "$name" "$resume" "$missing" "$walltime"
        job=DRY_RUN
        if (( ! dry_run )); then
            job=$(sbatch --parsable --job-name="$name" --account="$account" \
                --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=6G --gres=gpu:l40s:1 \
                --exclude=kn101 --time="$walltime" --chdir="$bench_dir" \
                --output="$output/logs/%x-%j.out" --error="$output/logs/%x-%j.err" \
                "$launcher" --worker "$bench_dir" "$source_name" "$seed" "$venv" "$data_root")
            job=${job%%;*}
            [[ $job =~ ^[0-9]+$ ]] || { echo "Unexpected sbatch response: $job" >&2; exit 2; }
        fi
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$source_name" "$seed" "$job" "$account" "$resume" "$missing" "$walltime" >> "$receipt"
        submitted=$((submitted + 1))
    done
done
printf 'DONE: %s jobs %s; %s complete/queued skipped. Receipt: %s\n' \
    "$submitted" "$([[ $dry_run == 1 ]] && echo planned || echo submitted)" "$skipped" "$receipt"
