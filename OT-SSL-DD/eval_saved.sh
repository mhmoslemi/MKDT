#!/bin/sh
set -eu

cd -- "$(dirname -- "$0")"

for percentage in 0.2 1 2; do
    checkpoint="$PWD/result_GenerativeFeatureMatching/res_GFM_CIFAR10_ResNet18_${percentage}percent_exp0.pt"
    if [ ! -f "$checkpoint" ]; then
        printf 'Missing checkpoint: %s\n' "$checkpoint" >&2
        exit 1
    fi
    python eval.py --synthetic-checkpoint "$checkpoint" --dataset CIFAR10 --percentage "$percentage" --model ResNet18 --ssl-method simclr --label-percentage 1 --seed 0
done
