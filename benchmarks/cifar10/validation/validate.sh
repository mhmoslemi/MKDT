#!/bin/bash
#SBATCH --account=aip-boyuwang
#SBATCH --job-name=cifar_bench_validate
#SBATCH --time=0-00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=15G
#SBATCH --gres=gpu:l40s:1
#SBATCH --output=benchmarks/cifar10/validation/validate-%j.out
set -euo pipefail
cd /scratch/mmoslem3/MKDT
module load StdEnv/2023 python/3.11.5
source "$HOME/ENV/bin/activate"
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export XDG_CACHE_HOME="$SCRATCH/cache"
export TMPDIR="$SLURM_TMPDIR"
python3 OT-SSL-DD/test_benchmark_cifar10.py
python3 benchmarks/cifar10/make_jobs.py
python3 benchmarks/cifar10/report.py --results benchmarks/cifar10/results --output benchmarks/cifar10/tables.md
python3 benchmarks/cifar10/validation/smoke.py
