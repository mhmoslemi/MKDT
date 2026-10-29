#!/bin/bash
set -euo pipefail
: "${SLURM_JOB_ID:?Run setup through Slurm}"
run="${SLURM_SUBMIT_DIR:?}/a2mim_r50_224_1gpu"
[[ -r $run/requirements.txt && -r $run/preflight.py ]]
module --force purge
module load StdEnv/2023 gcc/12.3 python/3.10.13
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PIP_CACHE_DIR="$run/cache/pip" XDG_CACHE_HOME="$run/cache/xdg" HF_HOME="$run/cache/hf"
export TORCH_HOME="$run/cache/torch" TMPDIR="${SLURM_TMPDIR:?}" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
mkdir -p "$run/cache"
exec 9>"$run/.environment.lock"
flock -n 9
[[ -r $run/venv/bin/activate ]] || virtualenv --no-download "$run/venv"
source "$run/venv/bin/activate"
convert_wheels() {
    local wheel
    for wheel in "$run"/cache/wheels/*manylinux*.whl; do
        [[ -f $wheel ]] || continue
        [[ ! -f ${wheel%%-manylinux*}-linux_x86_64.whl ]] || continue
        setrpaths --path "$wheel"
        python -m wheel tags --platform-tag linux_x86_64 "$wheel"
    done
}
python -m pip install --no-index 'torch==2.0.1+cu118' 'torchvision==0.15.2+cu118' || \
    PIP_NO_INDEX=0 python -m pip install --index-url https://download.pytorch.org/whl/cu118 'torch==2.0.1+cu118' 'torchvision==0.15.2+cu118'
if ! python -m pip install --no-index --only-binary=:all: -r "$run/requirements.txt"; then
    mkdir -p "$run/cache/wheels"
    PIP_CONFIG_FILE=/dev/null PIP_NO_INDEX=0 PYTHONPATH= python -m pip download --index-url https://pypi.org/simple \
        --platform manylinux2014_x86_64 --only-binary=:all: --no-deps --dest "$run/cache/wheels" -r "$run/requirements.txt"
    convert_wheels
    python -m pip install --no-index --find-links "$run/cache/wheels" -r "$run/requirements.txt"
fi
convert_wheels
# All MMCV dependencies are explicit above; use headless OpenCV, not a second GUI build.
if ! python -m pip install --no-index --no-deps --only-binary=:all: --find-links "$run/cache/wheels" 'mmcv-full==1.7.2'; then
    mkdir -p "$run/cache/wheels"
    python -m pip download --no-index --platform manylinux1_x86_64 --only-binary=:all: --no-deps \
        --find-links https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html \
        --dest "$run/cache/wheels" 'mmcv-full==1.7.2'
    convert_wheels
    python -m pip install --no-index --no-deps --find-links "$run/cache/wheels" 'mmcv-full==1.7.2'
fi
export PYTHONPATH="$run/upstream:$run"
python "$run/preflight.py"
