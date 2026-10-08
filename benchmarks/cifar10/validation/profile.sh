#!/bin/bash
#SBATCH --account=aip-boyuwang
#SBATCH --job-name=cifar_bench_profile
#SBATCH --time=0-00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=15G
#SBATCH --gres=gpu:l40s:1
#SBATCH --output=benchmarks/cifar10/validation/profile-%j.out
set -euo pipefail
cd /scratch/mmoslem3/MKDT
module load StdEnv/2023 python/3.11.5
source "$HOME/ENV/bin/activate"
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export XDG_CACHE_HOME="$SCRATCH/cache"
srun python3 OT-SSL-DD/benchmark_cifar10.py profile --output benchmarks/cifar10/validation/timing.json
