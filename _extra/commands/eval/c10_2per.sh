#!/bin/bash

cleanup() {
    echo "Terminating all running processes..."
    pkill -P $$
    exit
}

trap cleanup SIGINT

SUBSET_FRAC="default"

# Parse command-line arguments to extract --subset_frac value
while [[ $# -gt 0 ]]; do
  case "$1" in
    --subset_frac)
      SUBSET_FRAC="$2"
      EXTRA_ARGS+="--subset_frac $2 "
      shift 2
      ;;
    *)
      EXTRA_ARGS+="$1 "
      shift
      ;;
  esac
done

DATE_TIME=$(date "+%Y-%m-%d_%H-%M-%S")

RESULT_DIR="experiment_results/cifar10_2per_${SUBSET_FRAC}_labeled_data/$DATE_TIME/"

mkdir -p "$RESULT_DIR"

GPUS=(1 2 3 4 5 6 7)

### CIFAR 10 2%

# No Pre-Training 
python eval.py --pre_epoch 20 --no_pretrain --train_dataset CIFAR10 --label_path /home/jennyni/MKDT/target_rep/barlow_twins/CIFAR10_target_rep_train.pt --device ${GPUS[0]} $EXTRA_ARGS > "$RESULT_DIR/cifar10_nopretrain.txt" &

# Random
python eval.py --subset_path /home/jennyni/ssl-mtt/init/cifar10/cifar10_random100ipc.pkl --pre_epoch 20 --train_dataset CIFAR10 --label_path /home/jennyni/MKDT/target_rep/barlow_twins/CIFAR10_target_rep_train.pt --device ${GPUS[1]} $EXTRA_ARGS > "$RESULT_DIR/cifar10_random_2per.txt" &

# SAS subset
python eval.py --subset_path /home/jennyni/ssl-mtt/sas_subset/cifar10-cl-core-idx_1000.pkl --pre_epoch 20 --train_dataset CIFAR10 --label_path /home/jennyni/MKDT/target_rep/barlow_twins/CIFAR10_target_rep_train.pt --device ${GPUS[2]} $EXTRA_ARGS > "$RESULT_DIR/cifar10_sas_2per.txt" &

# MKDT High Loss
python eval.py --result_dir /home/jennyni/ssl-mtt/logged_files/CIFAR10/2024-05-05_11:49:29None --pre_epoch 20 --train_dataset CIFAR10 --label_path /home/jennyni/MKDT/target_rep/barlow_twins/CIFAR10_target_rep_train.pt --distilled_steps 5000 --device ${GPUS[3]} $EXTRA_ARGS > "$RESULT_DIR/cifar10_mkdt_high_2per.txt" &

# KRR-ST
python eval.py --train_dataset CIFAR10 --result_dir /home/jennyni/ssl-mtt/synthetic_data/CIFAR10/krr_st --pre_epoch 20 --train_dataset CIFAR100 --label_path /home/jennyni/MKDT/target_rep/barlow_twins/CIFAR10_target_rep_train.pt --use_krrst --device ${GPUS[4]} $EXTRA_ARGS > "$RESULT_DIR/cifar10_krrst_2per.txt" &

wait
