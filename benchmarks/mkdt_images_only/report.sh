#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=0-00:10:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
source "$PROJECT_ROOT/benchmarks/mkdt_images_only/environment.sh"
python3 benchmarks/mkdt_images_only/report.py \
    --batch "$MKDT_BATCH_ROOT" \
    --templates "$PROJECT_ROOT/CVPR_tables" \
    --output "$MKDT_BATCH_ROOT/tables" \
    --apply
