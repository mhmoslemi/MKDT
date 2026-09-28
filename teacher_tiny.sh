#!/usr/bin/env bash

# Train the MKDT Barlow Twins ResNet-18 teacher on Tiny ImageNet.
#
# The default input is the serialized dataset on the training server:
#   /home/mmoslem3/scratch/data/tinyimagenet.pt
#
# Examples:
#   bash teacher_tiny.sh
#   bash teacher_tiny.sh --device 1
#   bash teacher_tiny.sh --resume
#   bash teacher_tiny.sh --data-file /different/path/tinyimagenet.pt
#   bash teacher_tiny.sh --dry-run

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# MKDT Appendix A.1 specifies ResNet-18, Adam, batch size 256, LR 0.01,
# cosine annealing, weight decay 1e-6, and a 1024-dimensional projection.
# The referenced Tiny ImageNet ResNet-18 setup supplies the remaining values:
# 2000 epochs, 10 warmup epochs, and lambda_BT = 1/1024 ~= 0.0009765.
BATCH_SIZE=256
LEARNING_RATE=0.01
WEIGHT_DECAY=1e-6
PROJECTOR_DIM=1024
BARLOW_LAMBDA=0.0009765
WARMUP_EPOCHS=10

DATA_FILE="${DATA_FILE:-/home/mmoslem3/scratch/data/tinyimagenet.pt}"
DATA_PATH="${DATA_PATH:-/home/mmoslem3/scratch/data}"
CKPT_DIR="${CKPT_DIR:-$SCRIPT_DIR/krrst_teacher_ckpt}"
DEVICE="${DEVICE:-0}"
EPOCHS="${EPOCHS:-2000}"
NUM_WORKERS="${NUM_WORKERS:-16}"
SEED="${SEED:-0}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-10}"
TINY_INPUT_FORMAT="${TINY_INPUT_FORMAT:-auto}"
RESUME=0
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: bash teacher_tiny.sh [options]

Options:
  --data-file PATH        Tiny ImageNet .pt file
                          (default: /home/mmoslem3/scratch/data/tinyimagenet.pt)
  --ckpt-dir PATH         Checkpoint directory (default: ./krrst_teacher_ckpt)
  --device N              CUDA device number (default: 0)
  --epochs N              Training epochs (reference default: 2000)
  --num-workers N         DataLoader workers (default: 16)
  --seed N                Random seed (default: 0)
  --checkpoint-every N    Save every N epochs (default: 10)
  --input-format FORMAT   auto, uint8, zero_one, zero_255, or normalized
                          (default: auto; normalized assumes ImageNet stats)
  --resume                Continue from the full training checkpoint
  --dry-run               Print the resolved command without training
  -h, --help              Show this help

The .pt file may contain an image tensor, (images, labels), a dictionary using
common image keys, or an indexable PyTorch Dataset. Only load trusted .pt files.
EOF
}

need_value() {
  if [[ $# -lt 2 || -z "${2:-}" ]]; then
    echo "Missing value for $1" >&2
    usage >&2
    exit 2
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-file)
      need_value "$@"
      DATA_FILE="$2"
      shift 2
      ;;
    --ckpt-dir)
      need_value "$@"
      CKPT_DIR="$2"
      shift 2
      ;;
    --device)
      need_value "$@"
      DEVICE="$2"
      shift 2
      ;;
    --epochs)
      need_value "$@"
      EPOCHS="$2"
      shift 2
      ;;
    --num-workers)
      need_value "$@"
      NUM_WORKERS="$2"
      shift 2
      ;;
    --seed)
      need_value "$@"
      SEED="$2"
      shift 2
      ;;
    --checkpoint-every)
      need_value "$@"
      CHECKPOINT_EVERY="$2"
      shift 2
      ;;
    --input-format)
      need_value "$@"
      TINY_INPUT_FORMAT="$2"
      shift 2
      ;;
    --resume)
      RESUME=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

case "$TINY_INPUT_FORMAT" in
  auto|uint8|zero_one|zero_255|normalized) ;;
  *)
    echo "--input-format must be auto, uint8, zero_one, zero_255, or normalized" >&2
    exit 2
    ;;
esac

command=(
  python3 train_teacher.py
  --dataset Tiny
  --data_path "$DATA_PATH"
  --data_file "$DATA_FILE"
  --tiny_input_format "$TINY_INPUT_FORMAT"
  --ckpt_dir "$CKPT_DIR"
  --device "$DEVICE"
  --epochs "$EPOCHS"
  --batch_size "$BATCH_SIZE"
  --lr "$LEARNING_RATE"
  --weight_decay "$WEIGHT_DECAY"
  --proj_dim "$PROJECTOR_DIM"
  --lambd "$BARLOW_LAMBDA"
  --warmup_epochs "$WARMUP_EPOCHS"
  --num_workers "$NUM_WORKERS"
  --seed "$SEED"
  --checkpoint_every "$CHECKPOINT_EVERY"
)

if [[ "$RESUME" -eq 1 ]]; then
  command+=(--resume)
fi

echo "MKDT Barlow Twins Tiny ImageNet teacher settings"
echo "  data file:          $DATA_FILE"
echo "  checkpoint dir:     $CKPT_DIR"
echo "  device:             $DEVICE"
echo "  epochs:             $EPOCHS"
echo "  batch size:         $BATCH_SIZE"
echo "  learning rate:      $LEARNING_RATE"
echo "  schedule:           10-epoch warmup + cosine decay"
echo "  weight decay:       $WEIGHT_DECAY"
echo "  projection dim:     $PROJECTOR_DIM"
echo "  representation dim: 512"
echo "  Barlow lambda:      $BARLOW_LAMBDA"
echo "  input format:       $TINY_INPUT_FORMAT"
printf 'Command:'
printf ' %q' "${command[@]}"
printf '\n'

if [[ "$DRY_RUN" -eq 0 ]]; then
  "${command[@]}"
fi
