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

cd /home/mmoslem3/scratch/MKDT
source /home/mmoslem3/ENV/bin/activate
export PYTHONUNBUFFERED=1

python3 -c 'import torch; print("PyTorch:", torch.__version__, "CUDA:", torch.cuda.is_available())'

echo "Starting CIFAR-10 teacher on $(hostname) at $(date)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-not-set}"

srun --unbuffered bash teacher.sh \
  --dataset CIFAR10 \
  --device 0 \
  --epochs 1000 \
  --num-workers 1 \
  --resume

echo "Finished CIFAR-10 teacher at $(date)"
