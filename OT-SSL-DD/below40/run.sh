#!/bin/bash
if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi
set -e

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

dataset=CIFAR10
model=ResNet18
ssl_method=simclr
num_exp=1
save_path="$PWD/result_GenerativeFeatureMatching"

for percentage in 0.2 1 2; do
    python Gen-dd.py --resume --dataset "$dataset" --percentage "$percentage" --model "$model" --teacher_ssl "$ssl_method" --teacher_seeds 0 --num_exp "$num_exp" --save_path "$save_path" --teacher_path /scratch/mmoslem3/MKDT/Evaluate_benchamrk/full_ssl_checkpoints_1000ep_3seeds --vae_path /scratch/mmoslem3/MKDT/OT-SSL-DD/gen_dd_cache/residual_vae_CIFAR10_z256_e400_beta0.01_seed0.pt --vae_epochs 350
done

for percentage in 0.2 1 2; do
    for ((exp=0; exp<num_exp; exp++)); do
        python eval.py --synthetic-checkpoint "$save_path/res_GFM_${dataset}_${model}_${percentage}percent_exp${exp}.pt" --dataset "$dataset" --percentage "$percentage" --model "$model" --ssl-method "$ssl_method" --seed "$exp"
    done
done
