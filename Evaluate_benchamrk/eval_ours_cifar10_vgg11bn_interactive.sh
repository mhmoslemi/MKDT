#!/bin/bash
set -euo pipefail
bench_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
export OURS_EVAL_LOG_SUFFIX=_VGG11BN
exec bash "$bench_dir/eval_ours_cifar10_interactive.sh" --model VGG11BN "$@"
