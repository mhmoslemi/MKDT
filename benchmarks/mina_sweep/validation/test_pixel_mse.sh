#!/bin/bash
#SBATCH --account=aip-boyuwang
#SBATCH --job-name=mina_mse_test
#SBATCH --time=0-00:05:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --output=mina_sweep/validation/pixel_mse-%j.out
set -euo pipefail
cd /scratch/mmoslem3/MKDT
module load StdEnv/2023 python/3.11.5
source "$HOME/ENV/bin/activate"
export OMP_NUM_THREADS=1
export XDG_CACHE_HOME="$SCRATCH/cache"
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR="$SLURM_TMPDIR"
PYTHONPATH="$PWD/OT-SSL-DD${PYTHONPATH:+:$PYTHONPATH}" python3 _extra/test_mina_if_pixel_mse.py
