#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --gres=gpu:l40s:1
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
[[ $# -ge 1 ]] || { echo "Usage: $0 <prepare|distill> [1|2|5]" >&2; exit 2; }
: "${PROJECT_ROOT:?}"
: "${MKDT_SYNTH_ROOT:?}"

cd "$PROJECT_ROOT"
module load StdEnv/2023 python/3.11.5
source "${VENV_ACTIVATE:-$HOME/ENV/bin/activate}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export HF_HOME="$SCRATCH/hf_cache" XDG_CACHE_HOME="$SCRATCH/cache"
export WANDB_MODE=offline

data_root="${DATA_ROOT:-$SCRATCH/data}"
target_rep="$MKDT_SYNTH_ROOT/results/barlow_twins/CIFAR10_target_rep_train.pt"
buffer_root="$MKDT_SYNTH_ROOT/buffers"
buffer_dir="$buffer_root/CIFAR10/ConvNet"
init_dir="$MKDT_SYNTH_ROOT/init/cifar10"
teacher="$PROJECT_ROOT/krrst_teacher_ckpt/teacher_cifar10.pt"

case "$1" in
    prepare)
        [[ -f "$teacher" ]] || { echo "Missing official CIFAR10 teacher: $teacher" >&2; exit 1; }
        mkdir -p "$(dirname "$target_rep")" "$buffer_dir" "$init_dir"
        if [[ ! -f "$target_rep" ]]; then
            srun --unbuffered python3 get_target_rep.py \
                --dataset CIFAR10 --ssl_algorithm barlow_twins \
                --data_path "$data_root" --result_dir "$MKDT_SYNTH_ROOT/results" \
                --teacher_ckpt "$teacher" --batch_size 256 --device 0
        fi

        existing=0
        while [[ -f "$buffer_dir/replay_buffer_${existing}.pt" ]]; do
            (( existing += 1 ))
        done
        if (( existing < 100 )); then
            srun --unbuffered python3 buffer.py \
                --dataset CIFAR10 --model ConvNet --num_experts "$((100 - existing))" \
                --buffer_path "$buffer_root" --train_labels_path "$target_rep" \
                --train_epochs 20 --batch_train 256 --lr_teacher 0.1 \
                --mom 0.9 --l2 1e-4 --criterion mse \
                --data_path "$data_root" --device 0
        fi

        srun --unbuffered python3 sort_dset.py \
            --dataset CIFAR10 --model ConvNet --num_buffers 100 \
            --ssl_algo barlow_twins --batch_train 256 \
            --data_path "$data_root" --device 0 \
            --train_labels_path "$target_rep" --buffer_dir "$buffer_dir" \
            --output_dir "$init_dir"
        ;;
    distill)
        [[ $# -eq 2 ]] || { echo "distill requires a size: 1, 2, or 5" >&2; exit 2; }
        size=$2
        case "$size" in
            1|2) lr_img=1000; max_start_epoch=2 ;;
            5) lr_img=10000; max_start_epoch=5 ;;
            *) echo "Invalid MKDT percentage: $size" >&2; exit 2 ;;
        esac
        init_file="$init_dir/CIFAR10_barlow_twins_${size}_high_loss_indices.pkl"
        output="$MKDT_SYNTH_ROOT/CIFAR10_${size}pct"
        [[ -f "$target_rep" ]] || { echo "Missing target representations: $target_rep" >&2; exit 1; }
        [[ -f "$buffer_dir/replay_buffer_99.pt" ]] || { echo "Missing 100 expert trajectories" >&2; exit 1; }
        [[ -f "$init_file" ]] || { echo "Missing MKDT initialization: $init_file" >&2; exit 1; }
        if [[ -f "$output/images_5000.pt" ]]; then
            echo "MKDT ${size}% synthetic images already complete: $output/images_5000.pt"
            exit 0
        fi
        mkdir -p "$output"
        srun --unbuffered python3 distill.py \
            --dataset CIFAR10 --model ConvNet --iters 5000 --eval_it 1000 \
            --train_labels_path "$target_rep" --expert_epochs 2 \
            --lr_img "$lr_img" --syn_steps 40 \
            --image_init_idx_path "$init_file" \
            --max_start_epoch "$max_start_epoch" \
            --expert_dir "$buffer_dir" --data_path "$data_root" \
            --run_id "CIFAR10_${size}pct" --save_dir "$output"
        ;;
    *) echo "Invalid phase: $1" >&2; exit 2 ;;
esac
