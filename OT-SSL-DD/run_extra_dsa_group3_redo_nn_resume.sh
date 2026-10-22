#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
source ./dsa_launch_common.sh

RUN_DIR="$PWD/result_nn_20261022T180159Z_dsa_" run_method nn.py \
    res_IIC_CIFAR10_ResNet18BN_0.2percent_final.pt \
    --resume "$PWD/result_nn_20261022T180159Z_dsa_/res_IIC_CIFAR10_ResNet18BN_0.2percent_final.pt" \
    --Iteration 1000 --lr_img 0.0002 --lr_img_min 0.0002 \
    --random_models ResNet18BN --ssl_method simclr
run_method was-dd.py - \
    --teacher_source pretrained --teacher_seeds 0 --teacher_ssl simclr --optimizer adam \
    --experiment positive-negative --ssl-method simclr --ssl-epochs 2000 \
    --probe-epochs 100 --label-percentage 1 --runs 1
