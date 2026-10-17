#!/bin/bash
set -euo pipefail
runner_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$runner_dir/../common.sh"
run_benchmark "${BASH_SOURCE[0]}" --source CIFAR100 --model VGG11 --method full --subset 100 --ssl barlowtwins --seeds 2 "$@"
