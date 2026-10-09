#!/bin/bash
#SBATCH --account=aip-boyuwang
#SBATCH --job-name=cross_update_tables
#SBATCH --time=0-00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --output=benchmarks/cross_downstream/update-tables-%j.out

set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the MKDT project directory}}"
: "${CROSS_RUN_ROOT:?Set CROSS_RUN_ROOT to the submitted results batch}"
export PROJECT_ROOT
source "$PROJECT_ROOT/benchmarks/cross_downstream/environment.sh"
python3 benchmarks/cross_downstream/report.py \
    --results "$CROSS_RUN_ROOT" --output-dir "$CROSS_RUN_ROOT"
mkdir -p "$PROJECT_ROOT/CVPR_tables"
cp "$CROSS_RUN_ROOT/cifar100_downstream.tex" "$PROJECT_ROOT/CVPR_tables/cifar100_downstream.tex"
cp "$CROSS_RUN_ROOT/tinyimagenet_downstream.tex" "$PROJECT_ROOT/CVPR_tables/tinyimagenet_downstream.tex"
