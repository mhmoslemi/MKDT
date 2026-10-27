#!/bin/bash
set -euo pipefail
module --force purge
module load StdEnv/2023 python/3.11.5
source "${BENCH_VENV:-$HOME/ENV}/bin/activate"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd -- "${1:?Benchmark directory required}"
shift
exec python -u partial_probe_jobs/check_tiny_encoders.py "$@"
