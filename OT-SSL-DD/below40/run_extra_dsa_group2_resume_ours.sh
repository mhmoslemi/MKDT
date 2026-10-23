#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
source ./dsa_launch_common.sh

RUN_DIR="$PWD/result_ours_20261022T151749Z_dsa_" run_method ours.py \
    res_JointViewMMD_CIFAR10_ResNet18BN_0.2percent_final.pt \
    --resume "$PWD/result_ours_20261022T151749Z_dsa_/res_JointViewMMD_CIFAR10_ResNet18BN_0.2percent_final.pt" \
    --Iteration 3000 --save_every 500 \
    --random_models ResNet18BN --teacher_source pretrained --teacher_seed 0 --num_eval 0
