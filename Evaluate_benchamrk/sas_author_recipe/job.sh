#!/bin/bash
set -euo pipefail
: "${SLURM_JOB_ID:?Submit through Slurm}"
[[ ${SLURM_JOB_ACCOUNT:?} == aip-yiweilu ]]
module --force purge
module load StdEnv/2023 python/3.11.5
source /home/mmoslem3/ENV/bin/activate
run=/scratch/mmoslem3/MKDT/Evaluate_benchamrk/sas_author_recipe
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export XDG_CACHE_HOME="$run/cache" HF_HOME="$run/cache/hf" TORCH_HOME="$run/cache/torch"
export TMPDIR="${SLURM_TMPDIR:?}"
cd /scratch/mmoslem3/MKDT
stage=$1
shift
case "$stage" in
    proxy) exec torchrun --standalone --nproc_per_node=4 "$run/train_proxy.py";;
    groups|select)
        if [[ ! -f $run/deps/.complete ]]; then
            pip install --no-index --no-deps --target "$run/deps" ftfy wcwidth fast-pytorch-kmeans nvidia-ml-py
            touch "$run/deps/.complete"
        fi
        export PYTHONPATH="$run/deps${PYTHONPATH:+:$PYTHONPATH}"
        if [[ $stage == groups ]]; then
            exec python -u "$run/select_tiny.py" --groups-only
        fi
        exec python -u "$run/select_tiny.py";;
    evaluate) exec python -u "$run/worker.py" "$@";;
    *) exit 2;;
esac
