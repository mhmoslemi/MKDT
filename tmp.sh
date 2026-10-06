#!/bin/bash
#SBATCH --account=aip-boyuwang
#SBATCH --job-name=mkdt_teacher_cifar10
#SBATCH --time=0-0:21:40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --gres=gpu:1
#SBATCH --output=/home/mmoslem3/scratch/MKDT/TMP.out



set -euo pipefail

PROJECT_ROOT=/home/mmoslem3/scratch/MKDT


cd "$PROJECT_ROOT"
source /home/mmoslem3/ENV/bin/activate
export PYTHONUNBUFFERED=1






srun python OT-SSL-DD/main_idea.py --Iteration 2 --save_path tmp

