#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

common_args=(--dataset CIFAR10 --model ResNet18BN --percentage 0.2)
train_args=(--Iteration 8000 --dsa --lr_img 0.005)
eval_args=(--ssl-method simclr --ssl-epochs 2000 --probe-epochs 100
           --label-percentage 1 --runs 1 --seed 0)

stamp=$(date -u +%Y%m%dT%H%M%SZ)
sadm_dir="$PWD/result_SADM2_fresh_${stamp}_dsa_"
jepa_dir="$PWD/result_DDJEPA_fresh_${stamp}_dsa_"
cluster_dir="$PWD/result_CRMMD2_fresh_${stamp}_dsa_"

python -u Gen-dd2.py "${common_args[@]}" "${train_args[@]}" \
    --image_optimizer adam --teacher_source pretrained --teacher_seeds 0 --save_path "$sadm_dir"
python -u eval.py "${common_args[@]}" "${eval_args[@]}" \
    --synthetic-checkpoint "$sadm_dir/res_SADM_CIFAR10_ResNet18BN_0.2percent_exp0.pt" \
    --save-path "$sadm_dir/evaluation"

python -u DD-jepa.py --dataset CIFAR10 --model ResNet18BN --num_syn 100 \
    "${train_args[@]}" --save_path "$jepa_dir"
python -u eval.py "${common_args[@]}" "${eval_args[@]}" \
    --synthetic-checkpoint "$jepa_dir/res_DDJEPA_CIFAR10_ResNet18BN_0.2percent_final.pt" \
    --save-path "$jepa_dir/evaluation"

# cluster-ddd.py evaluates its final checkpoint itself.
python -u cluster-ddd.py "${common_args[@]}" "${train_args[@]}" "${eval_args[@]}" \
    --optimizer adam --teacher_source pretrained --teacher_seeds 0 --save_path "$cluster_dir"
