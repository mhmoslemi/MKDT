#!/bin/bash
# Portable launch helper. Invoke a scripts/*.sh runner with Bash.
set -euo pipefail
BENCH_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${BENCH_THREADS:-4}"
export MKL_NUM_THREADS="${BENCH_THREADS:-4}"

run_benchmark() {
    local runner_path="$1"
    shift
    local log_name
    log_name="$(basename -- "$runner_path" .sh)"
    mkdir -p "$BENCH_DIR/logs"
    "${BENCH_PYTHON:-python3}" -u "$BENCH_DIR/run.py" \
        --data-root "${BENCH_DATA_ROOT:-$BENCH_DIR/../../data}" \
        --output-root "$BENCH_DIR/results" \
        --workers "${BENCH_WORKERS:-4}" --threads "${BENCH_THREADS:-4}" \
        "$@" 2>&1 | tee -a "$BENCH_DIR/logs/$log_name.log"
}
