#!/bin/bash
# Reuse the saved MKDT 0.2% recipe: CIFAR10 on Wang, CIFAR100 on Lu.
set -euo pipefail
bench=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
export MKDT_SUBMIT_CLUSTER=vulcan
exec bash "$bench/submit_mkdt_02_wang_killarney.sh" "$@"
