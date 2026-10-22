#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

common_args=(--dataset CIFAR10 --model ResNet18BN --percentage 0.2)
train_args=(--Iteration 8000 --dsa)
eval_args=(--ssl-method simclr --ssl-epochs 2000 --probe-epochs 100
           --label-percentage 1 --runs 1 --seed 0)

# new_idea.py and cluster-ddd.py already evaluate their final checkpoint.
python -u new_idea.py "${common_args[@]}" "${train_args[@]}" "${eval_args[@]}" \
    --teacher_source pretrained --teacher_seeds 0 \
    --save_path "$PWD/result_new_idea2_dsa_"

python -u Gen-dd.py "${common_args[@]}" "${train_args[@]}" \
    --teacher_seeds 0 \
    --vae_path "$PWD/gen_dd_cache/residual_vae_CIFAR10_z256_e400_beta0.01_seed0.pt" \
    --save_path "$PWD/result_GenerativeFeatureMatching_dsa_"
python -u eval.py "${common_args[@]}" "${eval_args[@]}" \
    --synthetic-checkpoint "$PWD/result_GenerativeFeatureMatching_dsa_/res_GFM_CIFAR10_ResNet18BN_0.2percent_exp0.pt" \
    --save-path "$PWD/result_GenerativeFeatureMatching_dsa_/evaluation"

python -u Gen-dd2.py "${common_args[@]}" "${train_args[@]}" \
    --teacher_source pretrained --teacher_seeds 0 \
    --save_path "$PWD/result_SADM2_dsa_"
python -u eval.py "${common_args[@]}" "${eval_args[@]}" \
    --synthetic-checkpoint "$PWD/result_SADM2_dsa_/res_SADM_CIFAR10_ResNet18BN_0.2percent_exp0.pt" \
    --save-path "$PWD/result_SADM2_dsa_/evaluation"

python -u DD-jepa.py --dataset CIFAR10 --model ResNet18BN --num_syn 100 \
    --Iteration 8000 --dsa --save_path "$PWD/result_DDJEPA_dsa_"
python -u eval.py "${common_args[@]}" "${eval_args[@]}" \
    --synthetic-checkpoint "$PWD/result_DDJEPA_dsa_/res_DDJEPA_CIFAR10_ResNet18BN_0.2percent_final.pt" \
    --save-path "$PWD/result_DDJEPA_dsa_/evaluation"

python -u cluster-ddd.py "${common_args[@]}" "${train_args[@]}" "${eval_args[@]}" \
    --teacher_source pretrained --teacher_seeds 0 \
    --save_path "$PWD/result_CRMMD2_dsa_"
