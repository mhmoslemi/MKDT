#!/bin/bash
# Run with bash, not sbatch: this launcher submits exactly 54 ordinary jobs.
# Each job = one source / network / subset / SSL method, all ten seeds.
# No evaluation, arrays, dependencies, staging jobs, or final encoder deletion.
set -euo pipefail

if [[ ${1:-} == --worker ]]; then
    shift
    [[ $# == 8 ]] || { echo 'Invalid worker arguments' >&2; exit 2; }
    : "${SLURM_JOB_ID:?The worker must run inside a Slurm GPU job}"
    bench_dir=$1 source_name=$2 model=$3 subset=$4 ssl=$5 venv=$6 data_root=$7 output_root=$8
    module --force purge
    module load StdEnv/2023 python/3.11.5
    source "$venv/bin/activate"
    export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
    export HF_HOME="$output_root/cache/hf" XDG_CACHE_HOME="$output_root/cache/xdg"
    export TORCH_HOME="$output_root/cache/torch" TMPDIR="${SLURM_TMPDIR:?}"
    cd "$bench_dir"
    python - "$source_name" "$model" "$subset" "$ssl" "$data_root" "$output_root" "$0" <<'PY'
import gc
import json
import os
from pathlib import Path
import sys
import time

import torch
import torchvision

from common import (HERE, atomic_json, code_hashes, digest_json, exclusive_run,
                    file_hash, load_encoder)
from data import RawDataset, SOURCE_COUNTS, select_subset
from run import validate_protocol
from train_ssl import train

source, model, percentage, method, data_root, output_root, launcher = sys.argv[1:]
percentage = float(percentage)
output_root = Path(output_root).resolve()
if not output_root.is_relative_to(HERE):
    raise ValueError('Keep checkpoints inside Evaluate_benchamrk')
if not torch.cuda.is_available():
    raise RuntimeError('CUDA unavailable: no CPU fallback')
if int(os.environ.get('WORLD_SIZE', '1')) != 1:
    raise RuntimeError('Use one process and one GPU per job')
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
protocol = json.loads((HERE / 'protocol.json').read_text())
validate_protocol(protocol)
if (source not in SOURCE_COUNTS or model not in ('ConvNet', 'VGG11', 'ResNet18')
        or percentage not in (0.2, 1, 2) or method not in ('simclr', 'barlowtwins')
        or protocol['subset_seeds'] != list(range(10))):
    raise ValueError('Expected the fixed random-selection benchmark, seeds 0..9')

# Only the source TRAIN split is needed. No downstream data or probes are loaded.
base = RawDataset(data_root, source, True)
if len(base) != SOURCE_COUNTS[source]:
    raise ValueError('Unexpected source training split size')
signature = {'protocol': protocol, 'code_hashes': code_hashes(),
             'launcher_sha256': file_hash(launcher), 'source': source,
             'source_fingerprint': base.fingerprint, 'model': model,
             'selection_method': 'random', 'ssl': method, 'subset': percentage}
signature_hash = digest_json(signature)
group = output_root / source / model / 'random' / f'pt{percentage:g}' / method
print(f'Random SSL only: {source}/{model}/{percentage:g}%/{method}, seeds 0..9', flush=True)
for seed in range(10):
    folder = group / f'seed_{seed:03d}'
    with exclusive_run(folder):
        selected, selection = select_subset(base, 'random', percentage, seed, protocol,
                                            output_root / 'selections', 0, 'cuda')
        identity = {'signature': signature_hash, 'seed': seed,
                    'selection': selection, 'selection_method': 'random'}
        config = {'identity': identity, 'signature': signature}
        config_path = folder / 'config.json'
        if config_path.exists() and json.loads(config_path.read_text()) != config:
            raise ValueError(f'Existing checkpoint configuration differs: {folder}')
        atomic_json(config_path, config)
        if (folder / 'encoder.pt').exists():
            checked, _, saved = load_encoder(folder / 'encoder.pt', identity, device='cpu')
            del checked, saved
            print(f'SKIP verified encoder: {folder / "encoder.pt"}', flush=True)
        else:
            atomic_json(folder / 'environment.json', {
                'python': sys.version, 'torch': str(torch.__version__),
                'torchvision': str(torchvision.__version__),
                'gpu': torch.cuda.get_device_name(), 'workers': 0, 'threads': 1,
                'job_id': os.environ['SLURM_JOB_ID'],
                'cluster': os.environ.get('SLURM_CLUSTER_NAME')})
            started = time.monotonic()
            checkpoint, metadata = train(selected, protocol, source, model, method,
                                          seed, folder, identity, 0, 'cuda', full=False)
            atomic_json(folder / 'ssl_complete.json', {
                **metadata, 'checkpoint_file_sha256': file_hash(checkpoint),
                'training_wall_seconds': time.monotonic() - started})
            print(f'SAVED {checkpoint}', flush=True)
        # The resumable optimizer file is redundant after strict encoder validation.
        # encoder.pt and config.json are permanently retained for later probes.
        (folder / 'training.pt').unlink(missing_ok=True)
    del selected
    gc.collect()
    torch.cuda.empty_cache()
print(f'COMPLETE: ten retained encoders in {group}', flush=True)
PY
    exit 0
fi

case ${1:---submit} in
    --submit) dry_run=0 ;;
    --dry-run) dry_run=1 ;;
    --help|-h)
        printf '%s\n' \
            'bash Evaluate_benchamrk/random_ssl.sh --dry-run  # print all 54 jobs' \
            'bash Evaluate_benchamrk/random_ssl.sh            # submit on this cluster' \
            '54 jobs / 540 encoders: 3 sources x 3 BN networks x 3 sizes x 2 SSL methods.' \
            'Sizes: 0.2%, 1%, 2%. Each job trains seeds 0..9 sequentially; no evaluation.' \
            'Resources: 1 GPU, 1 CPU, 6G; walltime = approximate training estimate + 15 min.' \
            'Accounts: 49 jobs Wang, 5 Lu (approximately 90/10), verified before submission.' \
            'Uses the existing MKDT repository and Python environment on the destination server.' \
            'Defaults: BENCH_VENV=$HOME/ENV; BENCH_DATA_ROOT=<MKDT parent>/data.' \
            'GPU_TYPE may select a type from the destination cluster inventory; otherwise prefer L40S.' \
            'Results: Evaluate_benchamrk/random_ssl_checkpoints (logs and submission IDs there too).' \
            'Timings are old-code extrapolations, NOT measured new-trainer runtimes.' \
            'No automatic resubmission. Re-running resumes interrupted seeds and verifies finished encoders.'
        exit 0 ;;
    *) echo 'Use --submit (default), --dry-run, or --help' >&2; exit 2 ;;
esac
[[ $# -le 1 ]] || { echo 'Unexpected extra arguments' >&2; exit 2; }
[[ -z ${SLURM_JOB_ID:-} ]] || { echo 'Run this launcher with bash on the login node, not sbatch.' >&2; exit 2; }
bench_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
launcher="$bench_dir/random_ssl.sh"
output_root="$bench_dir/random_ssl_checkpoints"
for required in common.py data.py train_ssl.py run.py evaluate.py protocol.json \
                ../networks.py ../benchmarks/downstream_data.py; do
    [[ -r $bench_dir/$required ]] || {
        echo "Missing $bench_dir/$required; transfer the updated benchmark code to this server first." >&2
        exit 2
    }
done
data_root=${BENCH_DATA_ROOT:-$bench_dir/../../data}
venv=${BENCH_VENV:-$HOME/ENV}
wang=aip-boyuwang
lu=aip-yiweilu
epochs=$(awk '/"subset_epochs":/ {gsub(/[^0-9]/, ""); print; exit}' "$bench_dir/protocol.json")
batch=$(awk '/"batch_size":/ {gsub(/[^0-9]/, ""); print; exit}' "$bench_dir/protocol.json")
[[ $epochs =~ ^[1-9][0-9]*$ && $batch == 256 ]] || {
    echo 'Timing estimates require the existing batch-256 protocol.' >&2; exit 2;
}

estimate_seconds() {
    local dataset=$1 network=$2 method=$3 size=$4 reference=$1 rates rate image_scale=1 count
    local rate02 rate1 rate2
    # Measured mean seconds/step across ten historical RANDOM seeds, sizes .2/1/2.
    # Evidence: benchmarks/vulcan_subset_refresh/results/batch_20261016T163046Z_utW9z3/logs/
    # The historical code used cached 32px images, small projectors, GPU DiffAugment,
    # and different optimizer/backbones. These are timing references ONLY, never recipes.
    # Tiny has no matching random logs: extrapolate CIFAR100 by image area (64/32)^2,
    # and double .2% batch work (200 vs 100). No unmeasured CPU/projector multiplier.
    # New CPU augmentation / LARS / larger projectors may make these UNDERestimates;
    # Killarney GPU/CPU throughput can also differ. The user approved approximate times.
    [[ $dataset != TinyImageNet ]] || { reference=CIFAR100; image_scale=4; }
    case "$reference/$network/$method" in
        CIFAR10/ConvNet/simclr) rates='0.0056965 0.0164365 0.0144972' ;;
        CIFAR10/ConvNet/barlowtwins) rates='0.0058240 0.0167125 0.0165820' ;;
        CIFAR10/VGG11/simclr) rates='0.0083685 0.0174005 0.0174033' ;;
        CIFAR10/VGG11/barlowtwins) rates='0.0085570 0.0175860 0.0174180' ;;
        CIFAR10/ResNet18/simclr) rates='0.0224440 0.0597050 0.0597215' ;;
        CIFAR10/ResNet18/barlowtwins) rates='0.0224330 0.0600230 0.0598687' ;;
        CIFAR100/ConvNet/simclr) rates='0.0056755 0.0145650 0.0164160' ;;
        CIFAR100/ConvNet/barlowtwins) rates='0.0058715 0.0147705 0.0146455' ;;
        CIFAR100/VGG11/simclr) rates='0.0083995 0.0174895 0.0173542' ;;
        CIFAR100/VGG11/barlowtwins) rates='0.0085085 0.0175815 0.0174380' ;;
        CIFAR100/ResNet18/simclr) rates='0.0222980 0.0596675 0.0598595' ;;
        CIFAR100/ResNet18/barlowtwins) rates='0.0224670 0.0598445 0.0600357' ;;
        *) echo 'Missing timing reference' >&2; return 2 ;;
    esac
    read -r rate02 rate1 rate2 <<< "$rates"
    case $size in 0.2) rate=$rate02 ;; 1) rate=$rate1 ;; 2) rate=$rate2 ;; esac
    count=50000
    if [[ $dataset == TinyImageNet ]]; then
        count=100000
        [[ $size != 0.2 ]] || image_scale=8
    fi
    awk -v rate="$rate" -v scale="$image_scale" -v size="$size" -v count="$count" \
        -v epochs="$epochs" -v batch="$batch" 'BEGIN {
            n=int(count*size/100); b=(n<batch?n:batch);
            seconds=10*epochs*int(n/b)*rate*scale;
            rounded=int(seconds); if(seconds>rounded) rounded++;
            print rounded;
        }'
}

if (( ! dry_run )); then
    [[ -r $venv/bin/activate && -d $data_root ]] || {
        echo 'Set BENCH_VENV and BENCH_DATA_ROOT to the existing destination paths.' >&2; exit 2;
    }
    venv=$(cd -- "$venv" && pwd -P)
    data_root=$(cd -- "$data_root" && pwd -P)
    # Discover ON THE DESTINATION, before submitting anything. No partition guessing.
    module spider python/3.11.5 >/dev/null 2>&1
    sinfo -s
    inventory=$(sinfo -h -o '%G' --Node)
    gpu_types=$(printf '%s\n' "$inventory" | tr ',' '\n' | sed -nE 's/^gpu:([^:]+):.*/\1/p' | sort -u)
    gpu_type=${GPU_TYPE:-}
    if [[ -z $gpu_type ]]; then
        if grep -qx l40s <<< "$gpu_types"; then gpu_type=l40s
        elif grep -qx h100 <<< "$gpu_types"; then gpu_type=h100
        else echo "Set GPU_TYPE to a discovered type: $gpu_types" >&2; exit 2; fi
    fi
    grep -Fxq -- "$gpu_type" <<< "$gpu_types" || { echo "GPU type not available: $gpu_type" >&2; exit 2; }
    accounts=$(sacctmgr -nP show assoc "user=$(id -un)" format=Account)
    for account in "$wang" "$lu"; do
        printf '%s\n' "$accounts" | cut -d'|' -f1 | grep -Fxq -- "$account" || {
            echo "This account is not available here: $account" >&2; exit 2;
        }
    done
    mkdir -p "$output_root/logs"
    submission_log="$output_root/submitted_$(date -u +%Y%m%dT%H%M%SZ)_$$.tsv"
    printf 'job_id\taccount\tdataset\tmodel\tsubset_percent\tssl\tseeds\testimated_training_seconds\twalltime\n' > "$submission_log"
else
    gpu_type=${GPU_TYPE:-'<discovered-at-submit>'}
fi

printf 'Random SSL only; 54 jobs, ten seeds each, 540 retained encoders. Epochs/seed=%s\n' "$epochs"
printf 'Approximate OLD-LOG estimates + exactly 00:15:00; not measured new-trainer times.\n'
printf 'dataset\tmodel\tsubset_percent\tssl\tseeds\taccount\twalltime\n'
job_index=0
for source_name in CIFAR10 CIFAR100 TinyImageNet; do
    for model in ConvNet VGG11 ResNet18; do
        for subset in 0.2 1 2; do
            for ssl in simclr barlowtwins; do
                account=$wang
                if (( job_index % 10 == 9 )); then account=$lu; fi
                training_seconds=$(estimate_seconds "$source_name" "$model" "$ssl" "$subset")
                seconds=$((training_seconds + 900))
                printf -v walltime '%02d:%02d:%02d' "$((seconds / 3600))" "$((seconds / 60 % 60))" "$((seconds % 60))"
                printf '%s\t%s\t%s\t%s\t0..9\t%s\t%s\n' "$source_name" "$model" "$subset" "$ssl" "$account" "$walltime"
                if (( ! dry_run )); then
                    name="random_${source_name}_${model}_${subset/./p}_${ssl}"
                    job_id=$(sbatch --parsable --job-name="$name" --account="$account" \
                        --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=6G \
                        --gres="gpu:$gpu_type:1" --time="$walltime" --chdir="$bench_dir" \
                        --output="$output_root/logs/%x-%j.out" --error="$output_root/logs/%x-%j.err" \
                        "$launcher" --worker "$bench_dir" "$source_name" "$model" "$subset" "$ssl" \
                        "$venv" "$data_root" "$output_root")
                    job_id=${job_id%%;*}
                    printf '%s\t%s\t%s\t%s\t%s\t%s\t0..9\t%s\t%s\n' "$job_id" "$account" \
                        "$source_name" "$model" "$subset" "$ssl" "$training_seconds" "$walltime" >> "$submission_log"
                    printf 'Submitted job %s\n' "$job_id"
                fi
                job_index=$((job_index + 1))
            done
        done
    done
done
[[ $job_index == 54 ]]
if (( dry_run )); then
    printf 'Dry run only: no jobs submitted and no files written.\n'
else
    printf 'Submitted 54 jobs (49 Wang / 5 Lu). IDs: %s\nCheckpoints: %s\n' "$submission_log" "$output_root"
fi
