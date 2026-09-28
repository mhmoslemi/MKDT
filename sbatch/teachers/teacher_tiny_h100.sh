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

cd /home/mmoslem3/scratch/MKDT
source /home/mmoslem3/ENV/bin/activate
export PYTHONUNBUFFERED=1

python3 -c 'import torch; print("PyTorch:", torch.__version__, "CUDA:", torch.cuda.is_available())'

echo "Starting Tiny ImageNet teacher on $(hostname) at $(date)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-not-set}"

srun --unbuffered bash teacher_tiny.sh \
  --device 0 \
  --epochs 2000 \
  --num-workers 1 \

echo "Finished Tiny ImageNet teacher at $(date)"
