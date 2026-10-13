#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --gres=gpu:l40s:1
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
[[ $# -ge 2 ]] || { echo "Usage: $0 <target|experts|sort|distill> <dataset> [model] [arg]" >&2; exit 2; }
source "$PROJECT_ROOT/benchmarks/mkdt_multidataset/environment.sh"

phase=$1
dataset=$2
case "$dataset" in
    CIFAR10) teacher="$PROJECT_ROOT/krrst_teacher_ckpt/teacher_cifar10.pt" ;;
    CIFAR100) teacher="$PROJECT_ROOT/krrst_teacher_ckpt/barlow_twins_resnet18_cifar100.pt" ;;
    Tiny) teacher="$PROJECT_ROOT/krrst_teacher_ckpt/barlow_twins_resnet18_tinyimagenet.pt" ;;
    *) echo "Invalid MKDT dataset: $dataset" >&2; exit 2 ;;
esac

target_rep="$MKDT_SYNTH_ROOT/work/target_rep/barlow_twins/${dataset}_target_rep_train.pt"

case "$phase" in
    target)
        [[ -f $teacher ]] || { echo "Missing teacher: $teacher" >&2; exit 1; }
        if [[ -f $target_rep ]]; then
            echo "Target representations already exist: $target_rep"
            exit 0
        fi
        mkdir -p "$(dirname "$target_rep")"
        srun --unbuffered python3 get_target_rep.py \
            --dataset "$dataset" --ssl_algorithm barlow_twins \
            --data_path "$DATA_ROOT" \
            --result_dir "$MKDT_SYNTH_ROOT/work/target_rep" \
            --teacher_ckpt "$teacher" --batch_size 256 --device 0
        ;;
    experts)
        [[ $# -eq 4 ]] || { echo "experts requires model and start index" >&2; exit 2; }
        model=$3
        start=$4
        [[ -f $target_rep ]] || { echo "Missing target representations: $target_rep" >&2; exit 1; }
        destination="$MKDT_SYNTH_ROOT/work/buffers/$dataset/$model"
        end=$((start + 19))
        complete=true
        for index in $(seq "$start" "$end"); do
            [[ -f "$destination/replay_buffer_${index}.pt" ]] || complete=false
        done
        if $complete; then
            echo "Expert shard already exists: $dataset $model $start-$end"
            exit 0
        fi
        srun --unbuffered python3 buffer.py \
            --dataset "$dataset" --model "$model" --num_experts 20 \
            --start_index "$start" --buffer_path "$MKDT_SYNTH_ROOT/work/buffers" \
            --train_labels_path "$target_rep" --train_epochs 20 \
            --batch_train 256 --lr_teacher 0.1 --mom 0.9 --l2 1e-4 \
            --criterion mse --data_path "$DATA_ROOT" --device 0
        ;;
    sort)
        [[ $# -eq 3 ]] || { echo "sort requires model" >&2; exit 2; }
        model=$3
        buffer_dir="$MKDT_SYNTH_ROOT/work/buffers/$dataset/$model"
        init_dir="$MKDT_SYNTH_ROOT/work/init/$dataset/$model"
        for index in $(seq 0 99); do
            [[ -f "$buffer_dir/replay_buffer_${index}.pt" ]] || {
                echo "Missing replay buffer $index in $buffer_dir" >&2
                exit 1
            }
        done
        mkdir -p "$init_dir"
        srun --unbuffered python3 sort_dset.py \
            --dataset "$dataset" --model "$model" --num_buffers 100 \
            --ssl_algo barlow_twins --batch_train 256 \
            --data_path "$DATA_ROOT" --device 0 \
            --train_labels_path "$target_rep" --buffer_dir "$buffer_dir" \
            --output_dir "$init_dir"
        ;;
    distill)
        [[ $# -eq 5 ]] || { echo "distill requires model, percentage, and logical generator" >&2; exit 2; }
        model=$3
        size=$4
        generator=$5
        [[ $size == 1 || $size == 2 || $size == 5 ]] || { echo "Invalid percentage: $size" >&2; exit 2; }
        buffer_dir="$MKDT_SYNTH_ROOT/work/buffers/$dataset/$model"
        init_file="$MKDT_SYNTH_ROOT/work/init/$dataset/$model/${dataset}_barlow_twins_${size}_high_loss_indices.pkl"
        if [[ $dataset == Tiny ]]; then
            iters=10000
            syn_steps=10
            final=10000
            if [[ $size == 5 ]]; then lr_img=1000000; max_start=5; else lr_img=100000; max_start=2; fi
        else
            iters=5000
            syn_steps=40
            final=5000
            if [[ $size == 5 ]]; then lr_img=10000; max_start=5; else lr_img=1000; max_start=2; fi
        fi
        source_name=$dataset
        [[ $dataset == Tiny ]] && source_name=TinyImageNet
        case "$generator:$model" in
            ConvNet:ConvNet|ConvNet:ConvNetD4|ResNet18BN:ResNet18) ;;
            *) echo "Invalid logical/implementation generator pair: $generator $model" >&2; exit 2 ;;
        esac
        output="$MKDT_SYNTH_ROOT/generated/$generator/$source_name/${dataset}_${size}pct"
        [[ -f $target_rep ]] || { echo "Missing target representations: $target_rep" >&2; exit 1; }
        [[ -f $buffer_dir/replay_buffer_99.pt ]] || { echo "Missing expert trajectories" >&2; exit 1; }
        [[ -f $init_file ]] || { echo "Missing high-loss initialization: $init_file" >&2; exit 1; }
        if [[ -f "$output/images_${final}.pt" ]]; then
            echo "Synthetic images already exist: $output/images_${final}.pt"
            exit 0
        fi
        mkdir -p "$output"
        command=(srun --unbuffered python3 distill.py
            --dataset "$dataset" --model "$model" --iters "$iters" --eval_it 1000
            --train_labels_path "$target_rep" --expert_epochs 2
            --lr_img "$lr_img" --syn_steps "$syn_steps"
            --image_init_idx_path "$init_file" --max_start_epoch "$max_start"
            --expert_dir "$buffer_dir" --data_path "$DATA_ROOT"
            --run_id "${dataset}_${model}_${size}pct" --save_dir "$output")
        if [[ $model == ResNet18 ]]; then
            if [[ $dataset == Tiny ]]; then command+=(--batch_syn 32); else command+=(--batch_syn 64); fi
        fi
        "${command[@]}"
        ;;
    *) echo "Invalid phase: $phase" >&2; exit 2 ;;
esac
