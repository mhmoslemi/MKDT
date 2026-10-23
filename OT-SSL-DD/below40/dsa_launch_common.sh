#!/usr/bin/env bash

run_stamp=$(date -u +%Y%m%dT%H%M%SZ)

run_method() {
    local script="$1"
    local checkpoint="$2"
    shift 2
    local output_dir="${RUN_DIR:-$PWD/result_${script%.py}_${run_stamp}_dsa_}"
    mkdir -p "$output_dir"

    python -u "$script" \
        --dataset CIFAR10 --model ResNet18BN --percentage 0.2 \
        --Iteration 8000 --lr_img 0.001 --dsa --seed 0 \
        --save_path "$output_dir" "$@" 2>&1 | tee -a "$output_dir/run.log"

    # A dash marks a method that already performs the reference evaluation itself.
    if [[ "$checkpoint" != - ]]; then
        python -u eval.py \
            --synthetic-checkpoint "$output_dir/$checkpoint" \
            --dataset CIFAR10 --model ResNet18BN --percentage 0.2 \
            --ssl-method simclr --ssl-epochs 2000 --probe-epochs 100 \
            --label-percentage 1 --runs 1 --seed 0 \
            --save-path "$output_dir/evaluation" 2>&1 | tee -a "$output_dir/run.log"
    fi
}
