#!/bin/bash
#SBATCH --account=aip-boyuwang
#SBATCH --job-name=mkdt_teacher_tiny
#SBATCH --time=4-21:25:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=14G
#SBATCH --gres=gpu:h100:1
#SBATCH --output=/home/mmoslem3/scratch/MKDT/slurm-%x-%j.out

# Measured runtime: 3.5 minutes/epoch x 2000 epochs = 4-20:40:00.
# Requested walltime includes an additional 00:45:00 margin.

set -euo pipefail

PROJECT_ROOT=/home/mmoslem3/scratch/MKDT
DATA_FILE=/home/mmoslem3/scratch/data/tinyimagenet.pt
CHECKPOINT_ROOT=/home/mmoslem3/scratch/MKDT/krrst_teacher_ckpt

[[ -f "$PROJECT_ROOT/teacher_tiny.sh" ]] || { echo "FATAL: missing $PROJECT_ROOT/teacher_tiny.sh" >&2; exit 1; }
[[ -f "$PROJECT_ROOT/train_teacher.py" ]] || { echo "FATAL: missing $PROJECT_ROOT/train_teacher.py" >&2; exit 1; }
[[ -f /home/mmoslem3/ENV/bin/activate ]] || { echo "FATAL: missing /home/mmoslem3/ENV/bin/activate" >&2; exit 1; }
[[ -r "$DATA_FILE" ]] || { echo "FATAL: missing or unreadable: $DATA_FILE" >&2; exit 1; }
mkdir -p "$CHECKPOINT_ROOT"
[[ -w "$CHECKPOINT_ROOT" ]] || { echo "FATAL: not writable: $CHECKPOINT_ROOT" >&2; exit 1; }

cd "$PROJECT_ROOT"
source /home/mmoslem3/ENV/bin/activate
export PYTHONUNBUFFERED=1

python3 -c 'import sys, torch, torchvision; ok=torch.cuda.is_available(); name=torch.cuda.get_device_name(0) if ok else "unavailable"; print("PyTorch:", torch.__version__, "CUDA:", ok, "GPU:", name, flush=True); sys.exit(None if ok and "H100" in name.upper() else "Expected an allocated H100")'

echo "Starting Tiny ImageNet teacher on $(hostname) at $(date)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-not-set}"

srun --unbuffered bash teacher_tiny.sh \
  --data-file "$DATA_FILE" \
  --ckpt-dir "$CHECKPOINT_ROOT" \
  --device 0 \
  --epochs 2000 \
  --num-workers 1 \
  --seed 0 \
  --checkpoint-every 10 \
  --input-format auto

echo "Finished Tiny ImageNet teacher at $(date)"
