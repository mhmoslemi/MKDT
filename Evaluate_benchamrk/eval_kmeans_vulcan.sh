#!/bin/bash
# Run with bash AFTER k-means SSL finishes. Does not submit SSL jobs.
set -euo pipefail
bench_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
exec bash "$bench_dir/eval_jobs/launch.sh" kmeans "$@"
