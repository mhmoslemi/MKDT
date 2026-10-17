#!/bin/bash
set -euo pipefail
runner_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$runner_dir/../common.sh"
run_benchmark "${BASH_SOURCE[0]}" --source TinyImageNet --model VGG11 --method ours --subset 5 --ssl barlowtwins --seeds 0 1 2 3 4 5 6 7 8 9 "$@"
