#!/bin/bash
# Shared body sourced by the individual Slurm scripts.
set -euo pipefail
: "${SLURM_JOB_ID:?Submit the job with sbatch; do not run training on the login node.}"
: "${RUN_TAG:?}" "${LR_IMG:?}" "${BATCH_REAL:?}" "${NUM_RANDOM_NETWORKS:?}" "${TEMPERATURE:?}" "${DISTILL_AUG_STRATEGY:?}"

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$SCRATCH/MKDT}}"
DATA_ROOT="${DATA_ROOT:-$SCRATCH/data}"
SAVE_ROOT="${SAVE_ROOT:-$PROJECT_ROOT/results_mina_sweep}"
VENV_ACTIVATE="${VENV_ACTIVATE:-$HOME/ENV/bin/activate}"
# mina_IF.py clears save_path on startup. Give every submission a fresh directory.
SAVE_PATH="$SAVE_ROOT/$RUN_TAG/job_$SLURM_JOB_ID"
[[ -f "$PROJECT_ROOT/OT-SSL-DD/mina_IF.py" ]] || { echo 'FATAL: missing mina_IF.py' >&2; exit 1; }
[[ -f "$VENV_ACTIVATE" ]] || { echo "FATAL: missing environment: $VENV_ACTIVATE" >&2; exit 1; }
mkdir -p "$DATA_ROOT" "$SAVE_ROOT/$RUN_TAG"
mkdir "$SAVE_PATH" || { echo "FATAL: refusing to reuse $SAVE_PATH" >&2; exit 1; }

cd "$PROJECT_ROOT"
module load StdEnv/2023 python/3.11.5
source "$VENV_ACTIVATE"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export HF_HOME="$SCRATCH/hf_cache"
export XDG_CACHE_HOME="$SCRATCH/cache"

python3 -c 'import sys, torch, torchvision; ok=torch.cuda.is_available(); name=torch.cuda.get_device_name(0) if ok else "unavailable"; print("PyTorch:", torch.__version__, "CUDA:", ok, "GPU:", name, flush=True); sys.exit(0 if ok else "CUDA is unavailable in the allocated job")'

# Preserve mina_IF.py defaults (including 2000 iterations, seed 0, num_eval 3).
# Explicitly hold SSL augmentation at the baseline: its implicit default otherwise
# follows distill_aug_strategy, confounding the distillation augmentation sweep.
train_command=(python3 OT-SSL-DD/mina_IF.py
    --data_path "$DATA_ROOT" --save_path "$SAVE_PATH" --device cuda
    --lr_img "$LR_IMG" --batch_real "$BATCH_REAL"
    --num_random_networks "$NUM_RANDOM_NETWORKS" --temperature "$TEMPERATURE"
    --distill_aug_strategy "$DISTILL_AUG_STRATEGY"
    --distill_aug_mode "${DISTILL_AUG_MODE:-S}"
    --ssl_aug_strategy color_crop_cutout_flip_scale_rotate)
echo "Starting $RUN_TAG at $(date); output: $SAVE_PATH"
printf 'Command: '; printf '%q ' "${train_command[@]}"; printf '\n'
srun --unbuffered "${train_command[@]}"
echo "Finished $RUN_TAG at $(date)"
