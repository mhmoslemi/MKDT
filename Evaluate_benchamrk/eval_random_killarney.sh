#!/bin/bash
# Run with bash ON KILLARNEY after random-selection SSL finishes.
set -euo pipefail
bench_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
exec bash "$bench_dir/eval_jobs/launch.sh" random "$@"
