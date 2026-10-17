#!/bin/bash
set -euo pipefail
runner_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$runner_dir/../common.sh"
run_benchmark "${BASH_SOURCE[0]}" --source CIFAR10 --model ConvNet --method random --subset 2 --ssl barlowtwins --seeds 0 1 2 3 4 5 6 7 8 9 "$@"
