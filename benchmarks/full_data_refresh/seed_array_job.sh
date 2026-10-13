#!/bin/bash
set -euo pipefail

: "${SLURM_ARRAY_TASK_ID:?This wrapper must run as a Slurm array task.}"
: "${PROJECT_ROOT:?}"
[[ $# -eq 5 ]] || {
    echo "Usage: $0 <source> <model> <ssl> <ssl_epochs> <config>" >&2
    exit 2
}

exec "$PROJECT_ROOT/benchmarks/full_data_refresh/job.sh" \
    pretrain "$1" "$2" "$3" "$4" "$5" "$SLURM_ARRAY_TASK_ID" 1
