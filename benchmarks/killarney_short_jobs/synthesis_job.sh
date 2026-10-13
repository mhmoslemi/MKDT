#!/bin/bash
set -euo pipefail

[[ $# -eq 2 ]] || {
    echo "Usage: $0 <CIFAR10|CIFAR100> <1|2|5>" >&2
    exit 2
}

PROJECT_ROOT=$(readlink -f /home/mmoslem3/scratch/MKDT)
source "$PROJECT_ROOT/benchmarks/killarney_short_jobs/environment.sh"

dataset=$1
size=$2
case "$dataset" in
    CIFAR10|CIFAR100) ;;
    *) echo "Invalid dataset: $dataset" >&2; exit 2 ;;
esac
case "$size" in
    1|2|5) ;;
    *) echo "Invalid percentage: $size" >&2; exit 2 ;;
esac

model=VGG11
target_rep="$MKDT_SYNTH_ROOT/work/target_rep/barlow_twins/${dataset}_target_rep_train.pt"
buffer_dir="$MKDT_SYNTH_ROOT/work/buffers/$dataset/$model"
init_file="$MKDT_SYNTH_ROOT/work/init/$dataset/$model/${dataset}_barlow_twins_${size}_high_loss_indices.pkl"
output="$MKDT_SYNTH_ROOT/generated/VGG11/$dataset/${dataset}_${size}pct"

[[ -f "$target_rep" ]] || { echo "Missing target representations: $target_rep" >&2; exit 1; }
[[ -f "$buffer_dir/replay_buffer_99.pt" ]] || { echo "Missing VGG11 expert bank: $buffer_dir" >&2; exit 1; }
[[ -f "$init_file" ]] || { echo "Missing initialization: $init_file" >&2; exit 1; }

checkpoint_is_finite() {
    python3 - "$1" "$2" "$3" <<'PY'
import sys
import torch

for path in sys.argv[1:]:
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, torch.Tensor) or not torch.isfinite(value).all().item():
        raise SystemExit(f"Non-finite or invalid synthetic checkpoint: {path}")
PY
}

if [[ -f "$output/images_5000.pt" ]]; then
    if checkpoint_is_finite "$output/images_5000.pt" "$output/labels_5000.pt" "$output/synlr_5000.pt"; then
        echo "Finite synthetic checkpoint already exists: $output/images_5000.pt"
        exit 0
    fi
    quarantine="${output}.invalid_$(date -u +%Y%m%dT%H%M%SZ)"
    echo "Moving invalid checkpoint directory to: $quarantine" >&2
    mv "$output" "$quarantine"
fi

if [[ "$size" == 5 ]]; then
    lr_img=10000
    max_start=5
else
    lr_img=1000
    max_start=2
fi

mkdir -p "$output"
srun --unbuffered python3 distill.py \
    --dataset "$dataset" \
    --model "$model" \
    --iters 5000 \
    --eval_it 1000 \
    --train_labels_path "$target_rep" \
    --expert_epochs 2 \
    --lr_img "$lr_img" \
    --syn_steps 40 \
    --image_init_idx_path "$init_file" \
    --min_start_epoch 1 \
    --max_start_epoch "$max_start" \
    --expert_dir "$buffer_dir" \
    --data_path "$DATA_ROOT" \
    --run_id "${dataset}_${model}_${size}pct" \
    --save_dir "$output" \
    --batch_syn 64

checkpoint_is_finite "$output/images_5000.pt" "$output/labels_5000.pt" "$output/synlr_5000.pt"
