#!/bin/bash
# Run with bash on Vulcan, not sbatch: exactly 198 ordinary jobs.
# This is the tables' K-means Subset baseline, not a kNN evaluation classifier.
# Two-seed jobs whose projected walltime exceeds 5h are split into single seeds.
# No evaluation, arrays, dependencies, staging jobs, or final encoder deletion.
set -euo pipefail

epochs_for_subset() {
    case "$1/$2" in
        TinyImageNet/0.2) printf '6000\n' ;;
        TinyImageNet/1) printf '5000\n' ;;
        TinyImageNet/2) printf '4000\n' ;;
        CIFAR10/0.2|CIFAR100/0.2) printf '10000\n' ;;
        CIFAR10/1|CIFAR100/1) printf '8000\n' ;;
        CIFAR10/2|CIFAR100/2) printf '6000\n' ;;
        *) echo 'Unsupported source dataset or subset percentage.' >&2; return 2 ;;
    esac
}

if [[ ${1:-} == --worker ]]; then
    shift
    [[ $# == 11 ]] || { echo 'Invalid worker arguments' >&2; exit 2; }
    : "${SLURM_JOB_ID:?The worker must run inside a Slurm GPU job}"
    bench_dir=$1 source_name=$2 model=$3 subset=$4 ssl=$5 venv=$6 data_root=$7 output_root=$8
    seed_start=$9 epochs=${10} seed_count=${11}
    [[ ${SLURM_CLUSTER_NAME:-${CC_CLUSTER:-}} == vulcan ]] || {
        echo 'This worker is prepared for Vulcan only.' >&2; exit 2;
    }
    [[ $seed_start =~ ^[0-5]$ && $seed_count =~ ^[12]$ && $epochs == "$(epochs_for_subset "$source_name" "$subset")" ]] || {
        echo 'Expected one or two seeds and the configured source/subset epoch budget.' >&2; exit 2;
    }
    if (( seed_start + seed_count > 6 || (seed_count == 2 && seed_start % 2 != 0) )); then
        echo 'Invalid seed range: use a single seed or an aligned pair within 0..5.' >&2; exit 2
    fi
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
    nvidia-smi -L
    python - "$source_name" "$model" "$subset" "$ssl" "$data_root" "$output_root" "$0" "$seed_start" "$epochs" "$seed_count" <<'PY'
import fcntl
import gc
import hashlib
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

source, model, percentage, method, data_root, output_root, launcher, seed_start, epochs, seed_count = sys.argv[1:]
percentage = float(percentage)
seed_start, epochs, seed_count = int(seed_start), int(epochs), int(seed_count)
epoch_budgets = ({0.2: 6000, 1: 5000, 2: 4000} if source == 'TinyImageNet'
                 else {0.2: 10000, 1: 8000, 2: 6000})
if (seed_count not in (1, 2) or not 0 <= seed_start < 6 or seed_start + seed_count > 6
        or (seed_count == 2 and seed_start % 2 != 0)
        or epochs != epoch_budgets.get(percentage)):
    raise ValueError('Invalid seed range or source/subset epoch budget')
seed_label = str(seed_start) if seed_count == 1 else f'{seed_start}..{seed_start + 1}'
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
# Scope the new budgets to this kmeans-only launcher. The saved configuration
# contains the effective schedule; full-data and other runners are unchanged.
protocol['ssl']['subset_epochs'] = epochs
protocol['ssl']['checkpoint_every_epochs'] = 500
protocol['subset_seeds'] = list(range(6))
validate_protocol(protocol)
if protocol['ssl']['batch_size'] != 256:
    raise ValueError('Measured runtime estimates require the batch-256 recipe')
if (source not in SOURCE_COUNTS or model not in ('ConvNet', 'VGG11', 'ResNet18')
        or percentage not in (0.2, 1, 2) or method not in ('simclr', 'barlowtwins')
        or protocol['subset_seeds'] != list(range(6))):
    raise ValueError('Expected the fixed k-means-selection benchmark, seeds 0..5')

# Only the source TRAIN split is needed. No downstream data or probes are loaded.
base = RawDataset(data_root, source, True)
if len(base) != SOURCE_COUNTS[source]:
    raise ValueError('Unexpected source training split size')
# Allocation-time or account changes must not invalidate an identical SSL run.
# Hash the executed training worker, not the surrounding scheduler/timing code.
worker_source = Path(launcher).read_text().split("<<'PY'\n", 1)[1].split("\nPY\n", 1)[0]
signature = {'protocol': protocol, 'code_hashes': code_hashes(),
             'ssl_worker_sha256': hashlib.sha256(worker_source.encode()).hexdigest(), 'source': source,
             'source_fingerprint': base.fingerprint, 'model': model,
             'selection_method': 'kmeans', 'ssl': method, 'subset': percentage}
signature_hash = digest_json(signature)
group = output_root / source / model / 'kmeans' / f'pt{percentage:g}' / method
print(f'K-means SSL only: {source}/{model}/{percentage:g}%/{method}, '
      f'seeds {seed_label}, epochs/seed={epochs}', flush=True)
for seed in range(seed_start, seed_start + seed_count):
    folder = group / f'seed_{seed:03d}'
    with exclusive_run(folder):
        # One canonical selection per source/percentage/seed, shared by all SSL
        # methods and training networks. Block on its file lock rather than race
        # two GPU k-means runs into overwriting each other's selected indices.
        cache = output_root / 'selections'
        cache.mkdir(parents=True, exist_ok=True)
        selection_lock = cache / f'{source}_pt{percentage:g}_seed{seed:03d}.lock'
        torch.cuda.synchronize()
        selection_started = time.monotonic()
        with selection_lock.open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            selected, selection = select_subset(base, 'kmeans', percentage, seed, protocol,
                                                cache, 0, 'cuda')
        torch.cuda.synchronize()
        selection_seconds = time.monotonic() - selection_started
        print(f'SELECTED {len(selected)} k-means exemplars: seed={seed}, '
              f'seconds_including_cache_wait={selection_seconds:.3f}', flush=True)
        identity = {'signature': signature_hash, 'seed': seed,
                    'selection': selection, 'selection_method': 'kmeans'}
        config = {'identity': identity, 'signature': signature}
        config_path = folder / 'config.json'
        if config_path.exists() and json.loads(config_path.read_text()) != config:
            raise ValueError(f'Existing checkpoint configuration differs: {folder}')
        atomic_json(config_path, config)
        if (folder / 'encoder.pt').exists():
            checked, _, saved = load_encoder(folder / 'encoder.pt', identity, device='cpu')
            if saved['epochs'] != epochs:
                raise ValueError(f'Wrong epoch budget in {folder / "encoder.pt"}')
            del checked, saved
            print(f'SKIP verified encoder: {folder / "encoder.pt"}', flush=True)
        else:
            atomic_json(folder / 'environment.json', {
                'python': sys.version, 'torch': str(torch.__version__),
                'torchvision': str(torchvision.__version__),
                'launcher_sha256': file_hash(launcher),
                'gpu': torch.cuda.get_device_name(), 'workers': 0, 'threads': 1,
                'job_id': os.environ['SLURM_JOB_ID'],
                'cluster': os.environ.get('SLURM_CLUSTER_NAME')})
            started = time.monotonic()
            checkpoint, metadata = train(selected, protocol, source, model, method,
                                          seed, folder, identity, 0, 'cuda', full=False)
            if not metadata['complete'] or metadata['epochs'] != epochs:
                raise RuntimeError('Trainer did not finish the requested epoch budget')
            atomic_json(folder / 'ssl_complete.json', {
                **metadata, 'checkpoint_file_sha256': file_hash(checkpoint),
                'selection_wall_seconds_including_cache_wait': selection_seconds,
                'training_wall_seconds': time.monotonic() - started})
            print(f'SAVED {checkpoint}', flush=True)
        # The resumable optimizer file is redundant after strict encoder validation.
        # encoder.pt and config.json are permanently retained for later probes.
        (folder / 'training.pt').unlink(missing_ok=True)
    del selected
    gc.collect()
    torch.cuda.empty_cache()
print(f'COMPLETE: {seed_count} retained encoder(s), seeds {seed_label}, in {group}', flush=True)
PY
    exit 0
fi

case ${1:---submit} in
    --submit) dry_run=0 ;;
    --dry-run) dry_run=1 ;;
    --help|-h)
        printf '%s\n' \
            'bash Evaluate_benchamrk/kmeans_ssl.sh --dry-run  # print all 198 jobs' \
            'bash Evaluate_benchamrk/kmeans_ssl.sh            # submit on Vulcan' \
            '198 jobs / 324 encoders: 126 two-seed jobs and 72 single-seed jobs.' \
            'Split only configurations whose two-seed walltime (including 15 min) exceeds 5h.' \
            'Currently single-seed: TinyImageNet 1% and 2%, all networks, both SSL methods.' \
            'Other configurations keep pairs 0-1, 2-3, 4-5; six seeds 0..5; no evaluation.' \
            'A single seed may still exceed 5h; no truncation, stages, or automatic epoch reduction.' \
            'CIFAR10/CIFAR100 epochs PER SEED: 0.2%=10000, 1%=8000, 2%=6000.' \
            'TinyImageNet epochs PER SEED: 0.2%=6000, 1%=5000, 2%=4000.' \
            'Both schedules apply to BOTH SSL methods and ALL three networks.' \
            'Networks: ConvNetBN, VGG11BN, ResNet18BN. Sources: CIFAR10, CIFAR100, TinyImageNet.' \
            'Selection: existing k-means on random ConvNetGAP-BN features; unique real images.' \
            'Selection runs inside each job, with a shared locked cache; no preparation jobs.' \
            'Overwrites resumable training.pt every 500 epochs; keeps final encoder.pt.' \
            'Resources: 1 L40S GPU, 1 CPU, 6G; walltime = SSL projection + selection proxy + 15 min.' \
            'Accounts: 178 jobs Wang, 20 Lu (~90/10, nearest whole jobs), verified before submission.' \
            'Uses the existing MKDT repository and Python environment on the destination server.' \
            'Defaults: BENCH_VENV=$HOME/ENV; BENCH_DATA_ROOT=<MKDT parent>/data.' \
            'Vulcan L40S inventory, both accounts and module availability are checked before submission.' \
            'Results: Evaluate_benchamrk/kmeans_ssl_checkpoints_cifar10k8k6k_tiny6k5k4k_6seeds (including logs and job IDs).' \
            'Earlier schedule directories are left untouched and are NOT resumed into this run.' \
            'Timing reference: this trainer on Killarney L40S, 1 CPU, workers=0, at 1000 epochs.' \
            'Longer schedules on Vulcan are projected from Killarney L40S; not measured k-means runs.' \
            'Legacy selection proxies/seed: CIFAR10=3s, CIFAR100=4s, TinyImageNet=9s.' \
            'Those old selection logs used a different cached-image pipeline; current selection is unprofiled.' \
            'Only CIFAR100/ConvNet/0.2%/Barlow uses the matching CIFAR10 timing (CUDA failure).' \
            'No automatic resubmission. Reruns can resume ONLY the new-schedule checkpoints.'
        exit 0 ;;
    *) echo 'Use --submit (default), --dry-run, or --help' >&2; exit 2 ;;
esac
[[ $# -le 1 ]] || { echo 'Unexpected extra arguments' >&2; exit 2; }
[[ -z ${SLURM_JOB_ID:-} ]] || { echo 'Run this launcher with bash on the login node, not sbatch.' >&2; exit 2; }
bench_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
launcher="$bench_dir/kmeans_ssl.sh"
output_root="$bench_dir/kmeans_ssl_checkpoints_cifar10k8k6k_tiny6k5k4k_6seeds"
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
batch=$(awk '/"batch_size":/ {gsub(/[^0-9]/, ""); print; exit}' "$bench_dir/protocol.json")
[[ $batch == 256 ]] || {
    echo 'Timing estimates require the existing batch-256 protocol.' >&2; exit 2;
}

estimate_ssl_seconds() {
    local dataset=$1 network=$2 method=$3 size=$4 budget=$5 count=$6 rates measured
    local seconds02 seconds1 seconds2
    # Mean full-seed wall seconds at 1000 epochs, columns .2%, 1%, 2%.
    # Evidence: random_ssl_checkpoints/SOURCE/MODEL/random/ptSIZE/SSL/seed_*/ssl_complete.json
    # All 131 measured seeds: THIS trainer, Killarney, NVIDIA L40S, 1 CPU, workers=0.
    # Covers model initialization, training, periodic checkpoint I/O and final validation.
    # CIFAR100/ConvNet/.2/Barlow failed before training: use matching CIFAR10 (165.042343).
    # Estimate = seed_count * new_epochs/1000 * measured_seconds. No padding here;
    # exactly 900 seconds is added below. These remain longer-schedule projections.
    case "$dataset/$network/$method" in
        CIFAR10/ConvNet/simclr) rates='152.064814 389.881640 1137.717049' ;;
        CIFAR10/ConvNet/barlowtwins) rates='165.042343 376.801052 1110.647621' ;;
        CIFAR10/VGG11/simclr) rates='170.470094 426.226296 1258.825002' ;;
        CIFAR10/VGG11/barlowtwins) rates='171.199426 401.636168 1175.011173' ;;
        CIFAR10/ResNet18/simclr) rates='191.189161 455.985794 1360.410589' ;;
        CIFAR10/ResNet18/barlowtwins) rates='191.342436 453.655637 1339.718976' ;;
        CIFAR100/ConvNet/simclr) rates='150.789988 389.865760 1133.004543' ;;
        CIFAR100/ConvNet/barlowtwins) rates='165.042343 378.600236 1112.378044' ;;
        CIFAR100/VGG11/simclr) rates='169.794964 406.332590 1207.140469' ;;
        CIFAR100/VGG11/barlowtwins) rates='172.018359 400.057883 1165.262329' ;;
        CIFAR100/ResNet18/simclr) rates='190.902911 458.932893 1361.934841' ;;
        CIFAR100/ResNet18/barlowtwins) rates='195.503562 455.843248 1331.635225' ;;
        TinyImageNet/ConvNet/simclr) rates='508.378851 1943.642959 4488.616121' ;;
        TinyImageNet/ConvNet/barlowtwins) rates='496.084897 1819.173800 4208.770114' ;;
        TinyImageNet/VGG11/simclr) rates='529.595413 2266.329713 4673.310012' ;;
        TinyImageNet/VGG11/barlowtwins) rates='503.130282 1904.018456 4439.264389' ;;
        TinyImageNet/ResNet18/simclr) rates='715.795388 2725.167625 6339.877118' ;;
        TinyImageNet/ResNet18/barlowtwins) rates='691.170080 2598.348825 6027.111621' ;;
        *) echo 'Missing timing reference' >&2; return 2 ;;
    esac
    read -r seconds02 seconds1 seconds2 <<< "$rates"
    case $size in 0.2) measured=$seconds02 ;; 1) measured=$seconds1 ;; 2) measured=$seconds2 ;; esac
    awk -v measured="$measured" -v epochs="$budget" -v count="$count" 'BEGIN {
            seconds=count*epochs/1000*measured;
            rounded=int(seconds); if(seconds>rounded) rounded++;
            print rounded;
        }'
}

selection_proxy_seconds() {
    # Historical preparation walltimes, rounded up; NOT current-selector profiles.
    # CIFAR10: benchmarks/cifar10/validation/timing.json: features + kmeans(1,2,5) = 2.733s.
    # CIFAR100: c100_prepare_kmeans-1421486.out: maximum 3.2s/seed.
    # TinyImageNet: tiny_prepare_kmeans-1421578.out: maximum 8.7s/seed.
    # Older jobs used cached 32px tensors. This native-image selector may take longer;
    # the 15-minute cushion is unchanged. Each job records actual selection time.
    case $1 in
        CIFAR10) printf '3\n' ;;
        CIFAR100) printf '4\n' ;;
        TinyImageNet) printf '9\n' ;;
        *) echo 'Missing selection timing proxy' >&2; return 2 ;;
    esac
}

if (( ! dry_run )); then
    cluster=$(scontrol show config | awk '$1 == "ClusterName" {print $3}')
    [[ $cluster == vulcan ]] || { echo 'Submit this launcher on Vulcan only.' >&2; exit 2; }
    [[ -r $venv/bin/activate && -d $data_root ]] || {
        echo 'Set BENCH_VENV and BENCH_DATA_ROOT to the existing destination paths.' >&2; exit 2;
    }
    venv=$(cd -- "$venv" && pwd -P)
    data_root=$(cd -- "$data_root" && pwd -P)
    for marker in cifar-10-batches-py/data_batch_{1..5} cifar-10-batches-py/batches.meta \
                  cifar-100-python/train cifar-100-python/meta tinyimagenet.pt; do
        [[ -r $data_root/$marker ]] || { echo "Missing training data: $data_root/$marker" >&2; exit 2; }
    done
    # Discover on Vulcan before submitting anything. No partition guessing.
    module spider python/3.11.5 >/dev/null 2>&1
    sinfo -s
    inventory=$(sinfo -h -o '%G' --Node)
    gpu_types=$(printf '%s\n' "$inventory" | tr ',' '\n' | sed -nE 's/^gpu:([^:]+):.*/\1/p' | sort -u)
    gpu_type=l40s
    grep -Fxq -- "$gpu_type" <<< "$gpu_types" || { echo "GPU type not available: $gpu_type" >&2; exit 2; }
    accounts=$(sacctmgr -nP show assoc "user=$(id -un)" format=Account)
    for account in "$wang" "$lu"; do
        printf '%s\n' "$accounts" | cut -d'|' -f1 | grep -Fxq -- "$account" || {
            echo "This account is not available here: $account" >&2; exit 2;
        }
    done
    mkdir -p "$output_root/logs"
    submission_log="$output_root/submitted_$(date -u +%Y%m%dT%H%M%SZ)_$$.tsv"
    printf 'job_id\taccount\tdataset\tmodel\tsubset_percent\tssl\tseeds\tepochs_per_seed\testimated_training_seconds\tselection_proxy_seconds\twalltime\n' > "$submission_log"
else
    gpu_type=l40s
fi

printf 'K-means SSL only: 198 jobs (126 paired + 72 single-seed), 324 retained encoders.\n'
printf 'Two-seed configurations over 5h are split; individual seeds can still exceed 5h.\n'
printf 'CIFAR10/CIFAR100 epochs/seed: 0.2%%=10000, 1%%=8000, 2%%=6000.\n'
printf 'TinyImageNet epochs/seed: 0.2%%=6000, 1%%=5000, 2%%=4000.\n'
printf 'Separate output directory; no reuse of checkpoints from earlier schedule directories.\n'
printf 'Killarney L40S SSL-rate projection + legacy selection proxy + exactly 00:15:00.\n'
printf 'Current native-image k-means selection on Vulcan is unprofiled; these are estimates.\n'
printf 'dataset\tmodel\tsubset_percent\tssl\tseeds\tepochs_per_seed\taccount\twalltime\n'
job_index=0
for source_name in CIFAR10 CIFAR100 TinyImageNet; do
    for model in ConvNet VGG11 ResNet18; do
        for subset in 0.2 1 2; do
            epochs=$(epochs_for_subset "$source_name" "$subset")
            for ssl in simclr barlowtwins; do
                seed_count=2
                training_seconds=$(estimate_ssl_seconds "$source_name" "$model" "$ssl" "$subset" "$epochs" "$seed_count")
                selection_per_seed=$(selection_proxy_seconds "$source_name")
                selection_seconds=$((seed_count * selection_per_seed))
                if (( training_seconds + selection_seconds + 900 > 5 * 3600 )); then
                    seed_count=1
                    training_seconds=$(estimate_ssl_seconds "$source_name" "$model" "$ssl" "$subset" "$epochs" "$seed_count")
                    selection_seconds=$((seed_count * selection_per_seed))
                fi
                seconds=$((training_seconds + selection_seconds + 900))
                printf -v walltime '%02d:%02d:%02d' "$((seconds / 3600))" "$((seconds / 60 % 60))" "$((seconds % 60))"
                for ((seed_start=0; seed_start<6; seed_start+=seed_count)); do
                    seed_end=$((seed_start + seed_count - 1))
                    seed_label=$seed_start
                    seed_tag=$seed_start
                    if (( seed_count == 2 )); then
                        seed_label="$seed_start..$seed_end"
                        seed_tag="$seed_start-$seed_end"
                    fi
                    account=$wang
                    if (( (job_index + 1) * 20 / 198 > job_index * 20 / 198 )); then account=$lu; fi
                    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                        "$source_name" "$model" "$subset" "$ssl" "$seed_label" "$epochs" "$account" "$walltime"
                    if (( ! dry_run )); then
                        name="kmeans_${source_name}_${model}_${subset/./p}_${ssl}_e${epochs}_s${seed_tag}"
                        job_id=$(sbatch --parsable --job-name="$name" --account="$account" \
                            --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=6G \
                            --gres="gpu:$gpu_type:1" --time="$walltime" --chdir="$bench_dir" \
                            --output="$output_root/logs/%x-%j.out" --error="$output_root/logs/%x-%j.err" \
                            "$launcher" --worker "$bench_dir" "$source_name" "$model" "$subset" "$ssl" \
                            "$venv" "$data_root" "$output_root" "$seed_start" "$epochs" "$seed_count")
                        job_id=${job_id%%;*}
                        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$job_id" "$account" \
                            "$source_name" "$model" "$subset" "$ssl" "$seed_label" "$epochs" \
                            "$training_seconds" "$selection_seconds" "$walltime" >> "$submission_log"
                        printf 'Submitted job %s\n' "$job_id"
                    fi
                    job_index=$((job_index + 1))
                done
            done
        done
    done
done
[[ $job_index == 198 ]]
if (( dry_run )); then
    printf 'Dry run only: no jobs submitted and no files written.\n'
else
    printf 'Submitted 198 jobs (178 Wang / 20 Lu). IDs: %s\nCheckpoints: %s\n' "$submission_log" "$output_root"
fi
