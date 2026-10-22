#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
source ./dsa_launch_common.sh

RUN_DIR="$PWD/result_mina_IF_20261022T212321Z_dsa_" run_method mina_IF.py \
    res_IIC_CIFAR10_ResNet18BN_0.2percent_final.pt \
    --resume "$PWD/result_mina_IF_20261022T212321Z_dsa_/res_IIC_CIFAR10_ResNet18BN_0.2percent_final.pt" \
    --random_models ResNet18BN --teacher_source pretrained --teacher_seed 0 --num_eval 0
