#!/usr/bin/env bash

# Train the Barlow Twins ResNet-18 teacher used by MKDT on CIFAR-10,
# CIFAR-100, or both datasets.
#
# Examples:
#   bash teacher.sh                         # train CIFAR-10, then CIFAR-100
#   bash teacher.sh --dataset CIFAR10       # train only CIFAR-10
#   bash teacher.sh --dataset CIFAR100 --resume
#   bash teacher.sh --data-path /path/to/data --device 1
#   bash teacher.sh --dry-run               # print commands without training

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Paper settings from MKDT Appendix A.1:
#   ResNet-18, Adam, batch size 256, learning rate 0.01,
#   cosine learning-rate schedule, weight decay 1e-6,
#   projection dimension 1024, teacher representation dimension 512.
#
# Appendix A.1 refers to the ResNet-18 Barlow Twins setup of Bandara et al.
# for details it does not repeat. That setup uses 1000 epochs, a 10-epoch
# linear warmup, and lambda_BT = 0.0078125 for CIFAR-10/100.
BATCH_SIZE=256
LEARNING_RATE=0.01
WEIGHT_DECAY=1e-6
PROJECTOR_DIM=1024
BARLOW_LAMBDA=0.0078125
WARMUP_EPOCHS=10

# Operational settings. Override these with command-line flags or environment
# variables where noted.
DATASET="both"
DATA_PATH="${DATA_PATH:-$SCRIPT_DIR/data}"
CKPT_DIR="${CKPT_DIR:-$SCRIPT_DIR/krrst_teacher_ckpt}"
DEVICE="${DEVICE:-0}"
EPOCHS="${EPOCHS:-1000}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SEED="${SEED:-0}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-10}"
RESUME=0
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: bash teacher.sh [options]

Options:
  --dataset NAME          CIFAR10, CIFAR100, or both (default: both)
  --data-path PATH        Dataset download/storage directory (default: ./data)
  --ckpt-dir PATH         Checkpoint directory (default: ./krrst_teacher_ckpt)
  --device N              CUDA device number (default: 0)
  --epochs N              Training epochs (paper-reference default: 1000)
  --num-workers N         DataLoader workers (default: 8)
  --seed N                Random seed (default: 0)
  --checkpoint-every N    Save every N epochs (default: 10)
  --resume                Continue from the full training checkpoint
  --dry-run               Print the resolved command without running it
  -h, --help              Show this help

The script expects PyTorch, torchvision, and tqdm in the active environment.
It downloads CIFAR-10/100 automatically when they are not already present.
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
    --dataset)
      need_value "$@"
      DATASET="$2"
      shift 2
      ;;
    --data-path)
      need_value "$@"
      DATA_PATH="$2"
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

DATASET_KEY="$(printf '%s' "$DATASET" | tr '[:lower:]' '[:upper:]')"
case "$DATASET_KEY" in
  CIFAR10)
    DATASETS=("CIFAR10")
    ;;
  CIFAR100)
    DATASETS=("CIFAR100")
    ;;
  BOTH)
    DATASETS=("CIFAR10" "CIFAR100")
    ;;
  *)
    echo "--dataset must be CIFAR10, CIFAR100, or both; got: $DATASET" >&2
    exit 2
    ;;
esac

run_teacher() {
  local dataset="$1"
  local command=(
    python3 train_teacher.py
    --dataset "$dataset"
    --data_path "$DATA_PATH"
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

  echo
  echo "Training $dataset teacher"
  printf 'Command:'
  printf ' %q' "${command[@]}"
  printf '\n'

  if [[ "$DRY_RUN" -eq 0 ]]; then
    "${command[@]}"
  fi
}

echo "MKDT Barlow Twins teacher settings"
echo "  datasets:           ${DATASETS[*]}"
echo "  data path:          $DATA_PATH"
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

for dataset in "${DATASETS[@]}"; do
  run_teacher "$dataset"
done
