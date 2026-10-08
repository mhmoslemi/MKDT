#!/bin/bash
#SBATCH --account=aip-boyuwang
#SBATCH --job-name=downstream_validate
#SBATCH --time=0-00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=15G
#SBATCH --gres=gpu:l40s:1
#SBATCH --output=benchmarks/downstream/validation/validate-%j.out
set -euo pipefail
cd /scratch/mmoslem3/MKDT
module load StdEnv/2023 python/3.11.5
source "$HOME/ENV/bin/activate"
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export XDG_CACHE_HOME="$SCRATCH/cache"
export TMPDIR="$SLURM_TMPDIR"
PYTHONPATH="$PWD/benchmarks:$PWD/OT-SSL-DD${PYTHONPATH:+:$PYTHONPATH}" python3 _extra/test_downstream.py
python3 benchmarks/downstream/make_jobs.py
python3 benchmarks/downstream/validation/smoke.py
