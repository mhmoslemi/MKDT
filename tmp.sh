#!/bin/bash
#SBATCH --account=aip-boyuwang
#SBATCH --job-name=rfc_cifar10_5000
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --gres=gpu:1
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$SCRATCH/MKDT}}"
DATA_ROOT="${DATA_ROOT:-$SCRATCH/data}"
SAVE_ROOT="${SAVE_ROOT:-$PROJECT_ROOT/results2}"
VENV_ACTIVATE="${VENV_ACTIVATE:-$HOME/ENV/bin/activate}"

[[ -f "$PROJECT_ROOT/OT-SSL-DD/main_idea.py" ]] || { echo "FATAL: missing $PROJECT_ROOT/OT-SSL-DD/main_idea.py" >&2; exit 1; }
[[ -f "$VENV_ACTIVATE" ]] || { echo "FATAL: missing Python environment: $VENV_ACTIVATE" >&2; exit 1; }
mkdir -p "$DATA_ROOT" "$SAVE_ROOT"

cd "$PROJECT_ROOT"
source "$VENV_ACTIVATE"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

python3 -c 'import sys, torch, torchvision; ok=torch.cuda.is_available(); name=torch.cuda.get_device_name(0) if ok else "unavailable"; print("PyTorch:", torch.__version__, "CUDA:", ok, "GPU:", name, flush=True); sys.exit(0 if ok else "CUDA is unavailable in the allocated job")'

echo "Starting random-feature CIFAR-10 5000-iteration job on $(hostname) at $(date)"
echo "Project: $PROJECT_ROOT"
echo "Data: $DATA_ROOT"
echo "Output: $SAVE_ROOT"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-not-set}"

srun --unbuffered python3 OT-SSL-DD/main_idea.py --dataset CIFAR10 --data_path "$DATA_ROOT" --save_path "$SAVE_ROOT" --Iteration 5000 --device cuda --num_eval 6 --num_random_networks 2 --num_aug_pairs 5

echo "Finished random-feature CIFAR-10 5000-iteration job at $(date)"
