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

RESULT_DIR="experiment_results/tiny_2per_${SUBSET_FRAC}_labeled_data/$DATE_TIME/"

mkdir -p "$RESULT_DIR"

GPUS=(1 2 3 4 5 6 7)

### Tiny 2%

# No Pre-Training
python eval.py --train_dataset Tiny --train_model ConvNetD4 --label_path /home/jennyni/MKDT/target_rep/barlow_twins/Tiny_target_rep_train.pt --no_pretrain --device ${GPUS[0]} $EXTRA_ARGS > "$RESULT_DIR/tiny_nopretrain.txt" &

# Random
python eval.py --train_dataset Tiny --train_model ConvNetD4 --label_path /home/jennyni/MKDT/target_rep/barlow_twins/Tiny_target_rep_train.pt --subset_path /home/jennyni/ssl-mtt/init/tiny_imagenet/tiny_random_2perc.pkl --device ${GPUS[1]} $EXTRA_ARGS > "$RESULT_DIR/tiny_random_2per.txt" &

# MKDT High Loss
python eval.py --result_dir /home/jennyni/ssl-mtt/logged_files/Tiny/2024-05-04_22:48:05None --train_dataset Tiny --train_model ConvNetD4 --label_path /home/jennyni/MKDT/target_rep/barlow_twins/Tiny_target_rep_train.pt --distilled_steps 5000 --device ${GPUS[2]} $EXTRA_ARGS > "$RESULT_DIR/tiny_mkdt_high_2per.txt" &

# KRRST
python eval.py --result_dir /home/jennyni/ssl-mtt/synthetic_data/Tiny/krr_st --train_dataset Tiny --train_model ConvNetD4 --label_path /home/jennyni/MKDT/target_rep/barlow_twins/Tiny_target_rep_train.pt --use_krrst --device ${GPUS[3]} $EXTRA_ARGS > "$RESULT_DIR/tiny_krrst_2per.txt" 

# Full Data 
# python eval.py --train_dataset Tiny --train_model ConvNetD4 --label_path /home/jennyni/MKDT/target_rep/barlow_twins/Tiny_target_rep_train.pt --device ${GPUS[6]} $EXTRA_ARGS > "$RESULT_DIR/tiny_full.txt" 

wait
