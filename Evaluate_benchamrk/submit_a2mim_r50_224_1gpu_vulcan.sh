#!/bin/bash
# Pinned original OpenMixup recipe, with the agreed one-GPU adaptation.
set -euo pipefail
if [[ ${1:-} == --worker ]]; then
    shift
    [[ $# == 5 ]] || exit 2
    stage=$1 unit=$2 data=$3 probe_venv=$4 bundle=$5
    : "${SLURM_JOB_ID:?Submit through Slurm}"
    bench=${unit%/a2mim_r50_224_1gpu/units/*}
    run="$bench/a2mim_r50_224_1gpu"
    module --force purge
    if [[ $stage == ssl || $stage == check ]]; then
        module load StdEnv/2023 gcc/12.3 python/3.10.13
        source "$run/venv/bin/activate"
        export PYTHONPATH="$run/upstream:$run"
        script=(python -u -X faulthandler "$run/train.py")
        [[ $stage != check ]] || script+=(--check)
        script+=("$unit" "$data")
    elif [[ $stage == probe ]]; then
        module load StdEnv/2023 gcc/12.3 cuda/12.6 python/3.11.5
        source "$probe_venv/bin/activate"
        script=(python -u "$run/probes.py" "$unit" "$data" "$bundle")
    else exit 2; fi
    export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
    export XDG_CACHE_HOME="$run/cache/xdg" HF_HOME="$run/cache/hf" TORCH_HOME="$run/cache/torch" TMPDIR="${SLURM_TMPDIR:?}"
    cd "$bench"
    exec "${script[@]}"
fi
if [[ ${1:-} == --dispatch ]]; then
    [[ $# == 2 ]] || exit 2
    bench=$2
else
    bench=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
fi
run="$bench/a2mim_r50_224_1gpu"
venv=${BENCH_VENV:-$HOME/ENV}
data=${BENCH_DATA_ROOT:-$bench/../../data}
[[ -r $venv/bin/activate && -d $data ]] || exit 2
venv=$(cd -- "$venv" && pwd -P)
data=$(cd -- "$data" && pwd -P)
mkdir -p "$run/logs" "$run/batches"
if [[ ${1:-} == --dispatch ]]; then
    : "${SLURM_JOB_ID:?Dispatcher must use Slurm}"
    module --force purge
    module load StdEnv/2023 gcc/12.3 cuda/12.6 python/3.11.5
    source "$venv/bin/activate"
    export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
    export XDG_CACHE_HOME="$run/cache/xdg" HF_HOME="$run/cache/hf" TORCH_HOME="$run/cache/torch" TMPDIR="${SLURM_TMPDIR:?}"
    [[ -s $run/probe_calibration.json ]] || python "$run/probes.py" --calibrate "$data"
    exec python "$run/launch.py" "$data" "$venv"
fi
[[ $# == 0 && -z ${SLURM_JOB_ID:-} ]] || exit 2
[[ $(scontrol show config | awk '$1=="ClusterName" {print $3}') == vulcan ]] || exit 2
inventory=$(sinfo -h -o '%G' --Node | sort -u)
[[ $inventory == *gpu:l40s:* ]] || exit 2
exec 9>"$run/.bootstrap.lock"
flock -n 9 || exit 2
cd "$bench"
if [[ ! -r $run/cache/wheels/mmcv_full-1.7.2-cp310-cp310-manylinux1_x86_64.whl && ! -r $run/cache/wheels/mmcv_full-1.7.2-cp310-cp310-linux_x86_64.whl ]]; then
    mkdir -p "$run/cache/wheels"
    curl -fL --retry 2 --max-time 180 \
        -o "$run/cache/wheels/mmcv_full-1.7.2-cp310-cp310-manylinux1_x86_64.whl" \
        https://download.openmmlab.com/mmcv/dist/cu118/torch2.0.0/mmcv_full-1.7.2-cp310-cp310-manylinux1_x86_64.whl
fi
setup=$(squeue -h -u "$(id -un)" -n a2r50_setup -o '%i' | head -n 1)
if [[ -z $setup ]]; then
    setup=$(sbatch --parsable --job-name=a2r50_setup --account=aip-boyuwang --nodes=1 --ntasks=1 \
        --cpus-per-task=1 --mem=6G --gres=gpu:l40s:1 --time=00:30:00 --chdir="$bench" \
        --output="$run/logs/setup-%j.out" --error="$run/logs/setup-%j.err" "$run/setup.sh")
    setup=${setup%%;*}
fi
dispatch=$(squeue -h -u "$(id -un)" -n a2r50_dispatch -o '%i' | head -n 1)
if [[ -z $dispatch ]]; then
    dispatch=$(sbatch --parsable --job-name=a2r50_dispatch --account=aip-yiweilu --nodes=1 --ntasks=1 \
        --cpus-per-task=1 --mem=6G --gres=gpu:l40s:1 --time=00:30:00 --chdir="$bench" \
        --dependency="afterok:$setup" --kill-on-invalid-dep=yes --export="ALL,BENCH_VENV=$venv,BENCH_DATA_ROOT=$data" \
        --output="$run/logs/dispatch-%j.out" --error="$run/logs/dispatch-%j.err" \
        "$bench/submit_a2mim_r50_224_1gpu_vulcan.sh" --dispatch "$bench")
    dispatch=${dispatch%%;*}
fi
printf 'Setup: %s\nAutomatic calibration and cohort submission: %s\n' "$setup" "$dispatch"
