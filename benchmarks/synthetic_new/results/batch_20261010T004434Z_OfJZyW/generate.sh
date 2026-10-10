#!/bin/bash
#SBATCH --account=aip-boyuwang
#SBATCH --job-name=synthetic_new_generate
#SBATCH --time=0-00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
: "${PROJECT_ROOT:?}"
: "${SYNTH_BATCH_ROOT:?}"
: "${SYNTHETIC_INPUT_ROOT:?}"
cd "$PROJECT_ROOT"
module load StdEnv/2023 python/3.11.5
source "${VENV_ACTIVATE:-$HOME/ENV/bin/activate}"
python3 benchmarks/synthetic_new/make_jobs.py \
    --batch "$SYNTH_BATCH_ROOT" --project-root "$PROJECT_ROOT"
python3 benchmarks/benchmark_synthetic_new.py preflight \
    --synthetic-root "$SYNTHETIC_INPUT_ROOT" --device cpu
