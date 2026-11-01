#!/bin/bash
set -euo pipefail
: "${SLURM_JOB_ID:?Submit through Slurm}"
[[ $# == 1 || $# == 2 ]]
module --force purge
module load StdEnv/2023 python/3.11.5
source /home/mmoslem3/ENV/bin/activate
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK" NUMEXPR_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export XDG_CACHE_HOME="${SLURM_TMPDIR:?}/cache" MPLCONFIGDIR="$SLURM_TMPDIR/matplotlib"
export TORCH_HOME="$SLURM_TMPDIR/torch" HF_HOME="$SLURM_TMPDIR/hf" TMPDIR="$SLURM_TMPDIR"
cd /scratch/mmoslem3/MKDT
if [[ $1 == qualitative ]]; then
    [[ $# == 1 ]]
    exec python -u Evaluate_benchamrk/mipa_components/qualitative.py
fi
if [[ $1 == examples ]]; then
    [[ $# == 1 ]]
    exec python -u Evaluate_benchamrk/mipa_components/qualitative.py --examples
fi
if [[ $1 == cleaner_examples ]]; then
    [[ $# == 1 ]]
    exec python -u Evaluate_benchamrk/mipa_components/qualitative.py --cleaner-examples
fi
exec python -u Evaluate_benchamrk/mipa_components/worker.py "$@"
