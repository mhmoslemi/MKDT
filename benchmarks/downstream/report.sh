#!/bin/bash
#SBATCH --account=aip-boyuwang
#SBATCH --job-name=downstream_tables
#SBATCH --time=0-00:05:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?}}"
source "$PROJECT_ROOT/benchmarks/downstream/environment.sh"
python3 benchmarks/downstream/report.py --results "$DOWNSTREAM_RUN_ROOT"
