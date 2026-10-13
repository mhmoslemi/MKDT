#!/bin/sh
set -eu

SUBMIT_PART=1
export SUBMIT_PART
PROJECT_ROOT=$(readlink -f /home/mmoslem3/scratch/MKDT)
exec sh "$PROJECT_ROOT/benchmarks/killarney_short_jobs/submit_all.sh"
