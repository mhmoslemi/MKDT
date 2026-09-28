#!/bin/bash
#SBATCH --account=aip-boyuwang
#SBATCH --job-name=mkdt_teacher_cifar10
#SBATCH --time=0-12:41:40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --gres=gpu:h100:1
#SBATCH --output=/home/mmoslem3/scratch/MKDT/slurm-%x-%j.out

# Measured runtime: 43 seconds/epoch x 1000 epochs = 11:56:40.
# Requested walltime includes an additional 00:45:00 margin.

set -euo pipefail

PROJECT_ROOT=/home/mmoslem3/scratch/MKDT
DATA_ROOT=/home/mmoslem3/scratch/data
CHECKPOINT_ROOT=/home/mmoslem3/scratch/MKDT/krrst_teacher_ckpt

[[ -f "$PROJECT_ROOT/teacher.sh" ]] || { echo "FATAL: missing $PROJECT_ROOT/teacher.sh" >&2; exit 1; }
[[ -f "$PROJECT_ROOT/train_teacher.py" ]] || { echo "FATAL: missing $PROJECT_ROOT/train_teacher.py" >&2; exit 1; }
[[ -f /home/mmoslem3/ENV/bin/activate ]] || { echo "FATAL: missing /home/mmoslem3/ENV/bin/activate" >&2; exit 1; }
mkdir -p "$DATA_ROOT" "$CHECKPOINT_ROOT"
[[ -w "$DATA_ROOT" ]] || { echo "FATAL: not writable: $DATA_ROOT" >&2; exit 1; }
[[ -w "$CHECKPOINT_ROOT" ]] || { echo "FATAL: not writable: $CHECKPOINT_ROOT" >&2; exit 1; }

cd "$PROJECT_ROOT"
source /home/mmoslem3/ENV/bin/activate
export PYTHONUNBUFFERED=1

python3 -c 'import sys, torch, torchvision; ok=torch.cuda.is_available(); name=torch.cuda.get_device_name(0) if ok else "unavailable"; print("PyTorch:", torch.__version__, "CUDA:", ok, "GPU:", name, flush=True); sys.exit(None if ok and "H100" in name.upper() else "Expected an allocated H100")'

echo "Starting CIFAR-10 teacher on $(hostname) at $(date)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-not-set}"

srun --unbuffered bash teacher.sh \
  --dataset CIFAR10 \
  --data-path "$DATA_ROOT" \
  --ckpt-dir "$CHECKPOINT_ROOT" \
  --device 0 \
  --epochs 1000 \
  --num-workers 1 \
  --seed 0 \
  --checkpoint-every 10

echo "Finished CIFAR-10 teacher at $(date)"
