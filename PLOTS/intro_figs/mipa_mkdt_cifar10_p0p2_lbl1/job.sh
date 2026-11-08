#!/bin/bash
#SBATCH --job-name=mipa_mkdt_figure
#SBATCH --account=aip-yiweilu
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=00:01:00
#SBATCH --chdir=/scratch/mmoslem3/MKDT/PLOTS/time_storage
#SBATCH --output=/scratch/mmoslem3/MKDT/PLOTS/time_storage/render-%j.out
set -euo pipefail
module --force purge
module load StdEnv/2023 python/3.11.5
source /home/mmoslem3/ENV/bin/activate
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR="${SLURM_TMPDIR:?}/matplotlib"
python time_stoare_compare.py
