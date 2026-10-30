#!/bin/bash
set -euo pipefail
: "${SLURM_JOB_ID:?Submit through Slurm}"
module --force purge
module load StdEnv/2023 python/3.11.5
source /home/mmoslem3/ENV/bin/activate
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export XDG_CACHE_HOME=/scratch/mmoslem3/MKDT/Evaluate_benchamrk/mipa_fashion_singlechannel/cache
cd /scratch/mmoslem3/MKDT
exec python -u Evaluate_benchamrk/mipa_fashion_singlechannel/check_simclr.py
