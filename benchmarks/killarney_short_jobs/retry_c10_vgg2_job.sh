#!/bin/bash
set -uo pipefail

PROJECT_ROOT=$(readlink -f /home/mmoslem3/scratch/MKDT)
BUNDLE="$PROJECT_ROOT/benchmarks/killarney_short_jobs"

# The original run encountered one stochastic non-finite trajectory after
# 3,568/5,000 iterations.  Retry the unchanged, faithful configuration in a
# fresh Python process; do not alter its optimizer or learning rates.
last_status=1
for attempt in 1 2 3; do
    echo "CIFAR10 VGG11 2% distillation attempt $attempt/3"
    if "$BUNDLE/synthesis_job.sh" CIFAR10 2; then
        exit 0
    else
        last_status=$?
        echo "Attempt $attempt failed with status $last_status" >&2
    fi
done

echo "CIFAR10 VGG11 2% distillation failed after 3 unchanged attempts" >&2
exit "$last_status"
