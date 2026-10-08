#!/bin/bash
#SBATCH --job-name=download-downstream
#SBATCH --account=aip-boyuwang
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/mmoslem3/MKDT/benchmarks/downstream/download-%j.log
set -euo pipefail
: "${SLURM_JOB_ID:?Submit through Slurm}"
cd /scratch/mmoslem3/MKDT
module load StdEnv/2023 python/3.11.5
source "$HOME/ENV/bin/activate"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2
export XDG_CACHE_HOME="$SCRATCH/cache"
python benchmarks/downstream/download_datasets.py
