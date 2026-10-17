#!/bin/bash
set -euo pipefail
bench_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONDONTWRITEBYTECODE=1
"${BENCH_PYTHON:-python3}" "$bench_dir/plan.py" inputs
