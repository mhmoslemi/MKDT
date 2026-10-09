#!/bin/bash
#SBATCH --account=aip-boyuwang
#SBATCH --job-name=cross_tables
#SBATCH --time=0-00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the MKDT project directory}}"
source "$PROJECT_ROOT/benchmarks/cross_downstream/environment.sh"
python3 benchmarks/cross_downstream/report.py \
    --results "$CROSS_RUN_ROOT" --output-dir "$CROSS_RUN_ROOT"
