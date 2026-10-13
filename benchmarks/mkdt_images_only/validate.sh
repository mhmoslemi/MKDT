#!/bin/bash
#SBATCH --account=aip-boyuwang
#SBATCH --job-name=mkdt_images_validate
#SBATCH --time=0-00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=benchmarks/mkdt_images_only/validate-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the MKDT project directory}"
module load StdEnv/2023 python/3.11.5
source "${VENV_ACTIVATE:-$HOME/ENV/bin/activate}"
export PYTHONDONTWRITEBYTECODE=1
python3 -m py_compile benchmarks/benchmark_mkdt_images.py benchmarks/mkdt_images_only/report.py
python3 benchmarks/mkdt_images_only/validate.py
