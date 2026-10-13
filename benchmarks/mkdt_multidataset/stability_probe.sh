#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=00:20:00

set -euo pipefail

export PROJECT_ROOT=/scratch/mmoslem3/MKDT
export MKDT_MULTI_BATCH="$PROJECT_ROOT/benchmarks/mkdt_multidataset/results/batch_20261010T160046Z_Y6dqZS"
export MKDT_CODE_ROOT="$MKDT_MULTI_BATCH/code"
export MKDT_SYNTH_ROOT="$MKDT_MULTI_BATCH/synthetic"
export TARGET_CACHE_ROOT="$MKDT_MULTI_BATCH/targets"
export DATA_ROOT=/scratch/mmoslem3/data
source "$PROJECT_ROOT/benchmarks/mkdt_multidataset/environment.sh"

target="$MKDT_SYNTH_ROOT/work/target_rep/barlow_twins/CIFAR10_target_rep_train.pt"

run_probe() {
    local model=$1 size=$2 max_start=$3 name=$4
    srun --unbuffered python3 distill.py \
        --dataset CIFAR10 --model "$model" --iters 50 --eval_it 50 \
        --train_labels_path "$target" --expert_epochs 2 \
        --lr_img 1000 --syn_steps 40 --batch_syn 64 \
        --image_init_idx_path "$MKDT_SYNTH_ROOT/work/init/CIFAR10/$model/CIFAR10_barlow_twins_${size}_high_loss_indices.pkl" \
        --min_start_epoch 1 --max_start_epoch "$max_start" \
        --expert_dir "$MKDT_SYNTH_ROOT/work/buffers/CIFAR10/$model" \
        --data_path "$DATA_ROOT" --run_id "$name" \
        --save_dir "$MKDT_SYNTH_ROOT/debug_stability/$name"
}

run_probe ResNet18 1 2 r18_min1_max2
run_probe ResNet18 5 5 r18_min1_max5
run_probe VGG11 1 2 vgg_min1_max2
run_probe VGG11 5 5 vgg_min1_max5
