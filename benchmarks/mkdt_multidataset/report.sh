#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=0-00:10:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
source "$PROJECT_ROOT/benchmarks/mkdt_multidataset/environment.sh"
python3 benchmarks/mkdt_multidataset/report.py \
    --batch "$MKDT_MULTI_BATCH" --templates "$PROJECT_ROOT/CVPR_tables" \
    --output "$MKDT_MULTI_BATCH/tables"
