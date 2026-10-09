#!/bin/bash
#SBATCH --account=aip-boyuwang
#SBATCH --job-name=main_generate_jobs
#SBATCH --time=0-00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --output=benchmarks/main/generate-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the MKDT project directory}"
module load StdEnv/2023 python/3.11.5
export PYTHONDONTWRITEBYTECODE=1
python3 benchmarks/main/make_jobs.py
python3 - <<'PY'
import ast
from pathlib import Path

for name in ('benchmarks/benchmark_dataset.py', 'benchmarks/main/make_jobs.py', 'benchmarks/main/report.py'):
    ast.parse(Path(name).read_text(), filename=name)
print('Python syntax validation passed.')
PY
