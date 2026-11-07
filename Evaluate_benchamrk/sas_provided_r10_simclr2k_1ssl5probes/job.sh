#!/bin/bash
set -euo pipefail
: "${SLURM_JOB_ID:?Submit through Slurm}"
module --force purge
module load StdEnv/2023 python/3.11.5
source /home/mmoslem3/ENV/bin/activate
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export XDG_CACHE_HOME=/scratch/mmoslem3/MKDT/Evaluate_benchamrk/sas_provided_r10_simclr2k_1ssl5probes/cache
export HF_HOME="$XDG_CACHE_HOME/hf" TORCH_HOME="$XDG_CACHE_HOME/torch" TMPDIR="${SLURM_TMPDIR:?}"
cd /scratch/mmoslem3/MKDT/Evaluate_benchamrk
exec python -u sas_provided_r10_simclr2k_1ssl5probes/worker.py "$@"
