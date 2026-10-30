#!/bin/bash
set -euo pipefail
: "${SLURM_JOB_ID:?Submit through Slurm}"
module --force purge
module load StdEnv/2023 python/3.11.5
source /home/mmoslem3/ENV/bin/activate
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TMPDIR="${SLURM_TMPDIR:?}" XDG_CACHE_HOME=/scratch/mmoslem3/MKDT/Evaluate_benchamrk/mipa_multilayer/cache
cd /scratch/mmoslem3/MKDT
python -B OT-SSL-DD/test_method1_features.py
exec python -B Evaluate_benchamrk/mipa_multilayer/check_cosine10.py
