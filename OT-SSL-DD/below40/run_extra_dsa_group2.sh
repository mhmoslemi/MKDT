#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

source ./dsa_launch_common.sh

RUN_DIR="$PWD/result_main_SSL_new_20261022T043132Z_dsa_" run_method main_SSL_new.py - \
    --resume "$PWD/result_main_SSL_new_20261022T043132Z_dsa_/res_Scattering-SSL_CIFAR10_ResNet18BN_0.2percent.pt" --Iteration 7500 \
    --ssl-method simclr --ssl-epochs 2000 --probe-epochs 100 --label-percentage 1 --runs 1
run_method final.py res_AFMM_paired_CIFAR10_ResNet18BN_0.2percent_final.pt \
    --random_models ResNet18BN --teacher_source pretrained --teacher_seed 0 --num_eval 0
run_method mina_IF_intro.py res_IIC_CIFAR10_ResNet18BN_0.2percent.pt \
    --random_models ResNet18BN --teacher_source pretrained --teacher_seed 0 --num_eval 0
run_method ours.py res_JointViewMMD_CIFAR10_ResNet18BN_0.2percent_final.pt \
    --random_models ResNet18BN --teacher_source pretrained --teacher_seed 0 --num_eval 0
