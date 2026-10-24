#!/bin/bash
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
bash "$script_dir/run_ours_cifar10.sh"
bash "$script_dir/run_ours_cifar100.sh"
bash "$script_dir/run_ours_tinyimagenet.sh"
