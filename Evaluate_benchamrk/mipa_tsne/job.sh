#!/bin/bash
set -euo pipefail
: "${SLURM_JOB_ID:?Submit through sbatch}"
module --force purge
module load StdEnv/2023 python/3.11.5
source /home/mmoslem3/ENV/bin/activate
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK" NUMEXPR_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export XDG_CACHE_HOME="${SLURM_TMPDIR:?}/cache" MPLCONFIGDIR="$SLURM_TMPDIR/matplotlib"
export TORCH_HOME="$SLURM_TMPDIR/torch" HF_HOME="$SLURM_TMPDIR/hf" TMPDIR="$SLURM_TMPDIR"
cd /scratch/mmoslem3/MKDT
if [[ $1 == embed && $# == 1 ]]; then
    datasets=(CIFAR10 CIFAR100)
    sizes=(0.2 1 2)
    exec python Evaluate_benchamrk/mipa_tsne/compare.py embed --dataset "${datasets[SLURM_ARRAY_TASK_ID / 3]}" --percentage "${sizes[SLURM_ARRAY_TASK_ID % 3]}"
fi
exec python Evaluate_benchamrk/mipa_tsne/compare.py "$@"
