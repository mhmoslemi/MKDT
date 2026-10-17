#!/bin/bash
set -euo pipefail
BATCH=$(readlink -f -- "${1:?batch directory}")
source "$BATCH/environment.sh"
bash "$BATCH/code/Evaluate_benchamrk/check.sh"
exec python -u "$BATCH/validate.py" "$BATCH" "$BENCH_DATA_ROOT"

