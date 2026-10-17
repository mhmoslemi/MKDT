#!/bin/bash
set -euo pipefail
BATCH=$(readlink -f -- "${1:?batch}")
task=${2:?task key}
source "$BATCH/environment.sh"
exec python -u "$BATCH/stages.py" "$BATCH" "$task"

