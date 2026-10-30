#!/bin/bash
#SBATCH --job-name=iic_init_preview
#SBATCH --account=aip-yiweilu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=00:05:00
#SBATCH --chdir=/scratch/mmoslem3/MKDT
#SBATCH --output=/scratch/mmoslem3/MKDT/Evaluate_benchamrk/iic_init/logs/iic_init_preview-%j.out
#SBATCH --error=/scratch/mmoslem3/MKDT/Evaluate_benchamrk/iic_init/logs/iic_init_preview-%j.err
set -euo pipefail
: "${SLURM_JOB_ID:?Submit through sbatch}"
module --force purge
module load StdEnv/2023 python/3.11.5
source /home/mmoslem3/ENV/bin/activate
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export XDG_CACHE_HOME="${SLURM_TMPDIR:?}/cache"
export TORCH_HOME="$XDG_CACHE_HOME/torch" HF_HOME="$XDG_CACHE_HOME/hf"
export TMPDIR="$SLURM_TMPDIR"
exec python -u Evaluate_benchamrk/iic_init/preview_initialization.py --device cuda
