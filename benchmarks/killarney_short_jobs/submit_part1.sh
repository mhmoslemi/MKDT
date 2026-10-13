#!/bin/sh
set -eu

SUBMIT_PART=1
export SUBMIT_PART
exec sh /home/mmoslem3/scratch/MKDT/benchmarks/killarney_short_jobs/submit_all.sh

