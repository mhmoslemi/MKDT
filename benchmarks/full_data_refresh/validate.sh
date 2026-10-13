#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=0-00:05:00

set -euo pipefail
cd "${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
module load StdEnv/2023 python/3.11.5
source "${VENV_ACTIVATE:-$HOME/ENV/bin/activate}"
python3 -m py_compile \
    benchmarks/benchmark_full_refresh.py \
    benchmarks/full_data_refresh/report.py \
    benchmarks/benchmark_mkdt_multidataset.py \
    benchmarks/mkdt_multidataset/report.py \
    buffer.py distill.py
python3 benchmarks/benchmark_full_refresh.py --help >/dev/null
python3 benchmarks/benchmark_mkdt_multidataset.py --help >/dev/null
echo VALIDATION_OK
