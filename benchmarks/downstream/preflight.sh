#!/bin/bash
#SBATCH --account=aip-boyuwang
#SBATCH --job-name=ds_preflight
#SBATCH --time=0-00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=15G
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the MKDT project directory}}"
source "$PROJECT_ROOT/benchmarks/downstream/environment.sh"
python3 OT-SSL-DD/benchmark_downstream.py preflight \
    --device cpu --data-path "$DATA_ROOT" --targets "$TARGET_DATASETS"
