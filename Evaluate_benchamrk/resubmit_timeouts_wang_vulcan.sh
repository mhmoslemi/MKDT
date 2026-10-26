#!/bin/bash
# Same saved expert/SSL retries as Killarney; run them on Vulcan with Wang.
set -euo pipefail
bench=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
if [[ ${1:-} == --worker ]]; then
    shift
    stage=${1:-}
    [[ $stage == experts && $# == 9 || $stage == ssl && $# == 6 ]] || exit 2
    : "${SLURM_JOB_ID:?Submit through this launcher}"
    [[ ${SLURM_CLUSTER_NAME:-${CC_CLUSTER:-}} == vulcan && ${SLURM_JOB_ACCOUNT:-} == aip-boyuwang ]] || exit 2
    if [[ $stage == experts ]]; then
        bench=$2 code=$3 dataset=$4 start=$5 work=$6 synthetic=$7 data=$8 venv=$9
        arguments=(experts "$code" "$dataset" "$start" "$work" "$synthetic" "$data")
        export PYTHONPATH="$code:$(cd -- "$bench/.." && pwd -P)${PYTHONPATH:+:$PYTHONPATH}"
    else
        bench=$2 manifest=$3 venv=$4 data=$5 synthetic=$6
        arguments=(ssl "$manifest" "$data" "$synthetic/generated")
    fi
    module --force purge
    module load StdEnv/2023 python/3.11.5
    source "$venv/bin/activate"
    export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
    export HF_HOME="$bench/timeout_retries/cache/hf" XDG_CACHE_HOME="$bench/timeout_retries/cache/xdg"
    export TORCH_HOME="$bench/timeout_retries/cache/torch" WANDB_MODE=offline TMPDIR="${SLURM_TMPDIR:?}"
    cd "$bench"
    exec python -u timeout_retries/vulcan_worker.py "${arguments[@]}"
fi
export TIMEOUT_RETRY_CLUSTER=vulcan
exec bash "$bench/resubmit_timeouts_wang_killarney.sh" "$@"
