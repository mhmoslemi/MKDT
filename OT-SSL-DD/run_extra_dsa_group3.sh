#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
source ./dsa_launch_common.sh

run_method main_grapg.py res_SSLMetaLearning_CIFAR10_ResNet18BN_0.2percent.pt \
    --image_optimizer adam --num_eval 0
run_method method3.py syn_cifar10_0.002.pt --num_eval 0
run_method nn.py res_IIC_CIFAR10_ResNet18BN_0.2percent_final.pt \
    --random_models ResNet18BN --lr_img_min 0.001 --ssl_method simclr
run_method was-dd.py - \
    --teacher_source pretrained --teacher_seeds 0 --teacher_ssl simclr --optimizer adam \
    --experiment positive-negative --ssl-method simclr --ssl-epochs 2000 \
    --probe-epochs 100 --label-percentage 1 --runs 1
