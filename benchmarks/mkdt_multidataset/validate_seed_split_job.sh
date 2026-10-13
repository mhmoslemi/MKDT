#!/bin/bash
set -euo pipefail

: "${PROJECT_ROOT:?}"
: "${MKDT_MULTI_BATCH:?}"
: "${MKDT_CODE_ROOT:?}"
module load StdEnv/2023 python/3.11.5
source "${VENV_ACTIVATE:-$HOME/ENV/bin/activate}"
python3 - <<'PY'
import ast
from pathlib import Path

paths = (
    Path('/scratch/mmoslem3/MKDT/benchmarks/benchmark_mkdt_multidataset.py'),
    Path('/scratch/mmoslem3/MKDT/benchmarks/mkdt_multidataset/results/batch_20261010T160046Z_Y6dqZS/code/benchmarks/benchmark_mkdt_multidataset.py'),
)
for path in paths:
    ast.parse(path.read_text(), filename=str(path))
    print(f'validated {path}')
PY
bash -n \
    "$PROJECT_ROOT/benchmarks/full_data_refresh/seed_array_job.sh" \
    "$PROJECT_ROOT/benchmarks/mkdt_multidataset/seed_stage_job.sh"
