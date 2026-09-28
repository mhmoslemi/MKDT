#!/bin/bash
# End-to-end MKDT pipeline for the COBRA datasets, starting *after* the Barlow
# Twins teachers have already been trained (see teacher.sh / train_teacher.py).
#
# For every requested dataset this runs, in order:
#   1. get_target_rep.py  - extract the teacher's target representations
#   2. buffer.py           - collect student expert trajectories
#   3. sort_dset.py         - pick the high-loss subset used to init the synthetic set
#   4. distill.py           - run MTT-style distillation (one run per --percents entry)
#
# Resumable by default: each step is skipped if its output already exists.
# Pass --Force to wipe and redo everything from scratch, even if outputs exist.
#
# --smoke writes real (tiny) files under the exact same paths a full run
# uses. Never follow a --smoke run for a dataset with a real run on that same
# dataset unless you pass --Force on the real run too, or the real run will
# "resume" on top of the smoke-quality (1 expert, 3 epochs) artifacts.
#
# Usage:
#   ./run.sh [--Force] [--datasets "DS1 DS2 ..."] [--percents "2 5"]
#            [--device N] [--data-path PATH] [--num-experts N]
#            [--train-epochs N] [--smoke]
#
# Examples:
#   ./run.sh                                   # resume/run everything, paper hparams
#   ./run.sh --Force                            # wipe everything, start clean
#   ./run.sh --datasets "CIFAR10_S_90 BFFHQ"    # only these datasets
#   ./run.sh --smoke --datasets CIFAR10_S_90    # tiny sanity run, no real output

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# --- COBRA dataset -> image size, mirrors utils.py COBRA_64PX_KEYS ----------
COBRA_64PX_KEYS="UTKface CelebA BFFHQ"
ALL_COBRA_KEYS="CIFAR10_S_90 Colored_MNIST_foreground Colored_MNIST_background Colored_FashionMNIST_foreground Colored_FashionMNIST_background UTKface CelebA BFFHQ"

# --- defaults -----------------------------------------------------------
FORCE=false
DATASETS=""
PERCENTS="2 5"
DEVICE=0
DATA_PATH="/home/mmoslem3/scratch/data"
CKPT_DIR="$REPO_ROOT/krrst_teacher_ckpt"
RESULT_DIR="$REPO_ROOT/results"
BUFFER_ROOT="$REPO_ROOT/buffers_barlow_twins"
INIT_ROOT="$REPO_ROOT/init"
LOG_DIR="$REPO_ROOT/logs"
SSL_ALGO="barlow_twins"
NUM_EXPERTS=100
TRAIN_EPOCHS=20
SMOKE=false

while [ $# -gt 0 ]; do
  case "$1" in
    --Force) FORCE=true; shift ;;
    --datasets) DATASETS="$2"; shift 2 ;;
    --percents) PERCENTS="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --data-path) DATA_PATH="$2"; shift 2 ;;
    --num-experts) NUM_EXPERTS="$2"; shift 2 ;;
    --train-epochs) TRAIN_EPOCHS="$2"; shift 2 ;;
    --smoke) SMOKE=true; shift ;;
    -h|--help) grep '^#' "$0" | sed 's/^#//'; exit 0 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

mkdir -p "$LOG_DIR"

# Tiny overrides so the whole pipeline can be exercised in minutes to check
# for crashes. Output is not meant to be usable for real distillation.
# train_epochs must stay > max_start_epoch + expert_epochs used below, else
# distill.py indexes past the end of the (short) expert trajectory.
if $SMOKE; then
  NUM_EXPERTS=1
  TRAIN_EPOCHS=3
  echo "[smoke] num_experts=$NUM_EXPERTS train_epochs=$TRAIN_EPOCHS (distill iters/syn_steps/max_start_epoch scaled down too)"
fi

source /home/mmoslem3/ENV/bin/activate

# distill.py calls wandb.init(); wandb isn't logged in on this machine, which
# would otherwise block waiting for an API key. Log locally instead - override
# by exporting WANDB_MODE=online yourself before running if you want live logging.
export WANDB_MODE="${WANDB_MODE:-offline}"

model_for_dataset() {
  case " $COBRA_64PX_KEYS " in
    *" $1 "*) echo "ConvNetD4" ;;
    *) echo "ConvNet" ;;
  esac
}

# Hyperparameters mirror commands/distill/barlow_twins/{CIFAR10,Tiny}/*.sh:
# 32px datasets follow the CIFAR10 preset, 64px datasets follow the Tiny preset.
# NOTE: the repo's Tiny/5_per.sh is itself a copy-paste of 2_per.sh (same
# max_start_epoch/lr_img/init file) - that looks like a bug upstream, so for
# the 5% 64px case we instead extrapolate the same 2%->5% scaling CIFAR uses
# (max_start_epoch x2.5, lr_img x10) rather than reproduce that bug.
distill_hparams() {
  local dataset="$1" pct="$2" model
  model=$(model_for_dataset "$dataset")
  EVAL_IT=1000
  if [ "$model" = "ConvNetD4" ]; then
    SYN_STEPS=10; ITERS=10000; EXPERT_EPOCHS=2
    if [ "$pct" = "2" ]; then MAX_START_EPOCH=2; LR_IMG=100000
    else MAX_START_EPOCH=5; LR_IMG=1000000; fi
  else
    SYN_STEPS=40; ITERS=5000; EXPERT_EPOCHS=2
    if [ "$pct" = "2" ]; then MAX_START_EPOCH=2; LR_IMG=1000
    else MAX_START_EPOCH=5; LR_IMG=10000; fi
  fi
  if $SMOKE; then
    # eval_it=1 guarantees the final-iteration checkpoint (our completion
    # marker) is written even when ITERS is this small.
    ITERS=6; SYN_STEPS=2; EXPERT_EPOCHS=1; MAX_START_EPOCH=1; EVAL_IT=1
  fi
}

# --- pick datasets: auto-detect from saved teacher checkpoints unless given -
if [ -z "$DATASETS" ]; then
  found=""
  for key in $ALL_COBRA_KEYS; do
    if [ -f "$CKPT_DIR/barlow_twins_resnet18_${key}.pt" ]; then
      found="$found $key"
    else
      echo "skip: no saved teacher checkpoint for '$key' ($CKPT_DIR/barlow_twins_resnet18_${key}.pt not found)"
    fi
  done
  DATASETS="$found"
fi

if [ -z "${DATASETS// /}" ]; then
  echo "No datasets to process (no teacher checkpoints found in $CKPT_DIR). Run teacher.sh first."
  exit 1
fi

echo "Datasets: $DATASETS"
echo "Percents: $PERCENTS"
echo "Force: $FORCE   Smoke: $SMOKE   Device: $DEVICE"
echo

process_dataset() {
  local DATASET="$1"
  local MODEL TEACHER_CKPT TARGET_REP BUFFER_DIR EXISTING NEEDED
  MODEL=$(model_for_dataset "$DATASET")
  TEACHER_CKPT="$CKPT_DIR/barlow_twins_resnet18_${DATASET}.pt"
  TARGET_REP="$RESULT_DIR/$SSL_ALGO/${DATASET}_target_rep_train.pt"
  BUFFER_DIR="$BUFFER_ROOT/$DATASET/$MODEL"
  local INIT_DIR="$INIT_ROOT/$(echo "$DATASET" | tr '[:upper:]' '[:lower:]')"

  echo "=== $DATASET (model=$MODEL) ==="

  # --- Step 1: target representation ---
  if $FORCE; then rm -f "$TARGET_REP"; fi
  if [ -f "$TARGET_REP" ]; then
    echo "[1/4] target rep exists, skipping: $TARGET_REP"
  else
    echo "[1/4] extracting target representation..."
    python get_target_rep.py \
      --dataset "$DATASET" --ssl_algorithm "$SSL_ALGO" \
      --data_path "$DATA_PATH" --result_dir "$RESULT_DIR" \
      --teacher_ckpt "$TEACHER_CKPT" --device "$DEVICE" \
      2>&1 | tee "$LOG_DIR/${DATASET}_target_rep.log"
    if [ ! -f "$TARGET_REP" ]; then
      echo "[1/4] FAILED: $TARGET_REP was not produced"; return 1
    fi
  fi

  # --- Step 2: expert trajectories ---
  if $FORCE; then rm -rf "$BUFFER_DIR"; fi
  EXISTING=0
  while [ -f "$BUFFER_DIR/replay_buffer_${EXISTING}.pt" ]; do EXISTING=$((EXISTING + 1)); done
  NEEDED=$((NUM_EXPERTS - EXISTING))
  if [ "$NEEDED" -le 0 ]; then
    echo "[2/4] have $EXISTING/$NUM_EXPERTS expert trajectories, skipping"
  else
    echo "[2/4] collecting $NEEDED more expert trajectories ($EXISTING/$NUM_EXPERTS so far)..."
    python buffer.py \
      --dataset "$DATASET" --model "$MODEL" --num_experts "$NEEDED" \
      --buffer_path "$BUFFER_ROOT" --train_labels_path "$TARGET_REP" \
      --train_epochs "$TRAIN_EPOCHS" --data_path "$DATA_PATH" --device "$DEVICE" \
      2>&1 | tee -a "$LOG_DIR/${DATASET}_buffer.log"
    EXISTING=0
    while [ -f "$BUFFER_DIR/replay_buffer_${EXISTING}.pt" ]; do EXISTING=$((EXISTING + 1)); done
    if [ "$EXISTING" -lt "$NUM_EXPERTS" ]; then
      echo "[2/4] FAILED: only $EXISTING/$NUM_EXPERTS trajectories on disk after buffer.py"; return 1
    fi
  fi

  # --- Step 3: high-loss subset for synthetic-image initialization ---
  local need_sort=false
  for PCT in $PERCENTS; do
    [ -f "$INIT_DIR/${DATASET}_${SSL_ALGO}_${PCT}_high_loss_indices.pkl" ] || need_sort=true
  done
  if $FORCE; then rm -rf "$INIT_DIR"; need_sort=true; fi
  if ! $need_sort; then
    echo "[3/4] high-loss subset indices exist, skipping"
  else
    echo "[3/4] computing high-loss subset..."
    python sort_dset.py \
      --dataset "$DATASET" --model "$MODEL" --num_buffers "$NUM_EXPERTS" \
      --ssl_algo "$SSL_ALGO" --data_path "$DATA_PATH" \
      --train_labels_path "$TARGET_REP" --buffer_dir "$BUFFER_DIR" \
      --output_dir "$INIT_DIR" --device "$DEVICE" \
      2>&1 | tee "$LOG_DIR/${DATASET}_sort.log"
    for PCT in $PERCENTS; do
      if [ ! -f "$INIT_DIR/${DATASET}_${SSL_ALGO}_${PCT}_high_loss_indices.pkl" ]; then
        echo "[3/4] FAILED: missing ${PCT}% high-loss indices for $DATASET"; return 1
      fi
    done
  fi

  # --- Step 4: distillation, one run per requested percent ---
  for PCT in $PERCENTS; do
    local SYN_STEPS ITERS EXPERT_EPOCHS MAX_START_EPOCH LR_IMG EVAL_IT
    distill_hparams "$DATASET" "$PCT"
    local SAVE_DIR="$REPO_ROOT/logged_files/$DATASET/${DATASET}_${PCT}pct"
    local INIT_PKL="$INIT_DIR/${DATASET}_${SSL_ALGO}_${PCT}_high_loss_indices.pkl"
    local FINAL_IMG="$SAVE_DIR/images_${ITERS}.pt"

    if $FORCE; then rm -rf "$SAVE_DIR"; fi
    if [ -f "$FINAL_IMG" ]; then
      echo "[4/4] ${PCT}% distillation already complete, skipping: $FINAL_IMG"
      continue
    fi
    if [ -d "$SAVE_DIR" ]; then
      echo "[4/4] ${PCT}%: found a partial/killed run in $SAVE_DIR - distill.py has no mid-run resume, restarting this run from iteration 0"
    fi
    echo "[4/4] distilling ${PCT}% (model=$MODEL iters=$ITERS syn_steps=$SYN_STEPS max_start_epoch=$MAX_START_EPOCH lr_img=$LR_IMG)..."
    python distill.py \
      --dataset "$DATASET" --model "$MODEL" --iters "$ITERS" --eval_it "$EVAL_IT" \
      --train_labels_path "$TARGET_REP" --expert_epochs "$EXPERT_EPOCHS" \
      --lr_img "$LR_IMG" --syn_steps "$SYN_STEPS" \
      --image_init_idx_path "$INIT_PKL" --max_start_epoch "$MAX_START_EPOCH" \
      --expert_dir "$BUFFER_DIR" --data_path "$DATA_PATH" \
      --run_id "${DATASET}_${PCT}pct" --save_dir "$SAVE_DIR" \
      2>&1 | tee "$LOG_DIR/${DATASET}_distill_${PCT}pct.log"
    if [ ! -f "$FINAL_IMG" ]; then
      echo "[4/4] FAILED: ${PCT}% distillation for $DATASET did not produce $FINAL_IMG"; return 1
    fi
  done

  echo "=== $DATASET done ==="
  echo
  return 0
}

FAILED_DATASETS=""
for DATASET in $DATASETS; do
  if ! process_dataset "$DATASET"; then
    echo "!!! $DATASET failed, see logs in $LOG_DIR - continuing with remaining datasets"
    FAILED_DATASETS="$FAILED_DATASETS $DATASET"
  fi
done

echo "All requested datasets processed."
if [ -n "$FAILED_DATASETS" ]; then
  echo "Failed:$FAILED_DATASETS"
  exit 1
fi
