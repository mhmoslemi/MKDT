#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

source ./dsa_launch_common.sh

RUN_DIR="$PWD/result_main_SSL_20261022T043128Z_dsa_" run_method main_SSL.py - \
    --resume "$PWD/result_main_SSL_20261022T043128Z_dsa_/res_OT-SSL_CIFAR10_ResNet18BN_0.2percent.pt" --Iteration 5500 \
    --teacher_seed 0 --teacher_ssl simclr --ssl-method simclr \
    --ssl-epochs 2000 --probe-epochs 100 --label-percentage 1 --runs 1
run_method main_old_SSL.py - \
    --teacher_seed 0 --teacher_ssl simclr --ssl-method simclr \
    --ssl-epochs 2000 --probe-epochs 100 --label-percentage 1 --runs 1
run_method method1.py res_IIC_CIFAR10_ResNet18BN_0.2percent_final.pt \
    --random_models ResNet18BN --teacher_source pretrained --teacher_seed 0 --num_eval 0
run_method mehtod2.py res_IIC_CIFAR10_ResNet18BN_0.2percent_final.pt \
    --random_models ResNet18BN --teacher_source pretrained --teacher_seed 0 --num_eval 0
run_method mina_IF.py res_IIC_CIFAR10_ResNet18BN_0.2percent_final.pt \
    --random_models ResNet18BN --teacher_source pretrained --teacher_seed 0 --num_eval 0
