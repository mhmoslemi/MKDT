#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=0-00:10:00

set -euo pipefail
source "$PROJECT_ROOT/benchmarks/full_data_refresh/environment.sh"
python3 benchmarks/full_data_refresh/report.py --batch "$FULL_BATCH"
