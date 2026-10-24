#!/bin/bash
set -euo pipefail
bench_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
exec bash "$bench_dir/eval_ours_2k10k_interactive.sh" CIFAR10
