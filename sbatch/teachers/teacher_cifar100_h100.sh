#!/bin/bash
#SBATCH --account=aip-boyuwang
#SBATCH --job-name=mkdt_teacher_cifar100
#SBATCH --time=0-09:55:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --gres=gpu:h100:1
#SBATCH --output=/home/mmoslem3/scratch/MKDT/slurm-%x-%j.out

# Measured runtime: 33 seconds/epoch x 1000 epochs = 09:10:00.
# Requested walltime includes an additional 00:45:00 margin.

set -euo pipefail

cd /home/mmoslem3/scratch/MKDT

echo "Starting CIFAR-100 teacher on $(hostname) at $(date)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-not-set}"

srun --unbuffered bash teacher.sh \
  --dataset CIFAR100 \
  --device 0 \
  --epochs 1000 \
  --num-workers 1 \
  --resume

echo "Finished CIFAR-100 teacher at $(date)"
