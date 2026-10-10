#!/bin/bash
#SBATCH --account=aip-boyuwang
#SBATCH --job-name=synthetic_new_tables
#SBATCH --time=0-00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
source "$PROJECT_ROOT/benchmarks/synthetic_new/environment.sh"
python3 benchmarks/benchmark_synthetic_new.py report \
    --batch "$SYNTH_BATCH_ROOT" \
    --templates "$SYNTH_BATCH_ROOT/table_templates" \
    --output "$SYNTH_BATCH_ROOT/tables"
