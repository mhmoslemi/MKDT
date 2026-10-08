#!/bin/bash
#SBATCH --account=aip-boyuwang
#SBATCH --job-name=mina_04_batch_128
#SBATCH --time=0-03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --gres=gpu:l40s:1
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
RUN_TAG=04_batch_128
LR_IMG=0.005
BATCH_REAL=128
NUM_RANDOM_NETWORKS=20
TEMPERATURE=0.2
DISTILL_AUG_STRATEGY=color_crop_cutout_flip_scale_rotate
DISTILL_AUG_MODE=S
PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$SCRATCH/MKDT}}"
source "$PROJECT_ROOT/benchmarks/mina_sweep/run.sh"
