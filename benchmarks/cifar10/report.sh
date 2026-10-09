#!/bin/bash
#SBATCH --account=aip-yiweilu
#SBATCH --job-name=cifar_bench_tables
#SBATCH --time=0-00:05:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?}}"
source "$PROJECT_ROOT/benchmarks/cifar10/environment.sh"
python3 benchmarks/cifar10/report.py --results "$BENCH_RUN_ROOT" --output "$BENCH_RUN_ROOT/tables.md"
