#!/bin/bash
set -euo pipefail

[[ $# -ge 2 ]] || {
    echo "Usage: $0 <experts|sort|distill> Tiny [model] [arg]" >&2
    exit 2
}

PROJECT_ROOT=$(readlink -f /home/mmoslem3/scratch/MKDT)
source "$PROJECT_ROOT/benchmarks/killarney_short_jobs/environment.sh"

phase=$1
dataset=$2
[[ $dataset == Tiny ]] || { echo "Part 3 only accepts Tiny" >&2; exit 2; }

target_rep="$MKDT_SYNTH_ROOT/work/target_rep/barlow_twins/Tiny_target_rep_train.pt"
[[ -s $target_rep ]] || { echo "Missing target representations: $target_rep" >&2; exit 1; }

case "$phase" in
    experts)
        [[ $# -eq 4 ]] || { echo "experts requires model and start index" >&2; exit 2; }
        model=$3
        start=$4
        [[ $model == VGG11 ]] || { echo "Part 3 only builds Tiny VGG11 experts" >&2; exit 2; }
        destination="$MKDT_SYNTH_ROOT/work/buffers/Tiny/VGG11"
        end=$((start + 19))
        complete=true
        for index in $(seq "$start" "$end"); do
            [[ -s "$destination/replay_buffer_${index}.pt" ]] || complete=false
        done
        if $complete; then
            echo "Expert shard already exists: Tiny VGG11 $start-$end"
            exit 0
        fi
        srun --unbuffered python3 buffer.py \
            --dataset Tiny --model VGG11 --num_experts 20 \
            --start_index "$start" --buffer_path "$MKDT_SYNTH_ROOT/work/buffers" \
            --train_labels_path "$target_rep" --train_epochs 20 \
            --batch_train 256 --lr_teacher 0.1 --mom 0.9 --l2 1e-4 \
            --criterion mse --data_path "$DATA_ROOT" --device 0 \
            --progress_percent_step 2
        ;;
    sort)
        [[ $# -eq 3 && $3 == VGG11 ]] || { echo "sort requires VGG11" >&2; exit 2; }
        buffer_dir="$MKDT_SYNTH_ROOT/work/buffers/Tiny/VGG11"
        init_dir="$MKDT_SYNTH_ROOT/work/init/Tiny/VGG11"
        for index in $(seq 0 99); do
            [[ -s "$buffer_dir/replay_buffer_${index}.pt" ]] || {
                echo "Missing replay buffer $index in $buffer_dir" >&2
                exit 1
            }
        done
        complete=true
        for size in 1 2 5; do
            [[ -s "$init_dir/Tiny_barlow_twins_${size}_high_loss_indices.pkl" ]] || complete=false
        done
        if $complete; then
            echo "Sorted Tiny VGG11 initializations already exist: $init_dir"
            exit 0
        fi
        mkdir -p "$init_dir"
        srun --unbuffered python3 sort_dset.py \
            --dataset Tiny --model VGG11 --num_buffers 100 \
            --ssl_algo barlow_twins --batch_train 256 \
            --data_path "$DATA_ROOT" --device 0 \
            --train_labels_path "$target_rep" --buffer_dir "$buffer_dir" \
            --output_dir "$init_dir"
        ;;
    distill)
        [[ $# -eq 5 ]] || {
            echo "distill requires model, percentage, and logical generator" >&2
            exit 2
        }
        model=$3
        size=$4
        generator=$5
        [[ $size == 1 || $size == 2 || $size == 5 ]] || {
            echo "Invalid percentage: $size" >&2
            exit 2
        }
        case "$generator:$model" in
            ConvNet:ConvNetD4|VGG11:VGG11) ;;
            *) echo "Invalid logical/implementation generator pair: $generator $model" >&2; exit 2 ;;
        esac

        buffer_dir="$MKDT_SYNTH_ROOT/work/buffers/Tiny/$model"
        init_file="$MKDT_SYNTH_ROOT/work/init/Tiny/$model/Tiny_barlow_twins_${size}_high_loss_indices.pkl"
        output="$MKDT_SYNTH_ROOT/generated/$generator/TinyImageNet/Tiny_${size}pct"
        [[ -s "$buffer_dir/replay_buffer_99.pt" ]] || {
            echo "Missing expert trajectories: $buffer_dir" >&2
            exit 1
        }
        [[ -s $init_file ]] || { echo "Missing initialization: $init_file" >&2; exit 1; }

        checkpoint_is_finite() {
            python3 - "$1" "$2" "$3" <<'PY'
import sys
import torch

for path in sys.argv[1:]:
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, torch.Tensor) or not torch.isfinite(value).all().item():
        raise SystemExit(f"Non-finite or invalid synthetic checkpoint: {path}")
PY
        }

        if [[ -s "$output/images_10000.pt" ]]; then
            if checkpoint_is_finite "$output/images_10000.pt" \
                "$output/labels_10000.pt" "$output/synlr_10000.pt"; then
                echo "Finite synthetic checkpoint already exists: $output/images_10000.pt"
                exit 0
            fi
            quarantine="${output}.invalid_$(date -u +%Y%m%dT%H%M%SZ)"
            echo "Moving invalid checkpoint directory to: $quarantine" >&2
            mv "$output" "$quarantine"
        fi

        if [[ $size == 5 ]]; then
            lr_img=1000000
            max_start=5
        else
            lr_img=100000
            max_start=2
        fi
        min_start=0
        [[ $model == VGG11 ]] && min_start=1

        mkdir -p "$output"
        command=(srun --unbuffered python3 distill.py
            --dataset Tiny --model "$model" --iters 10000 --eval_it 1000
            --train_labels_path "$target_rep" --expert_epochs 2
            --lr_img "$lr_img" --syn_steps 10
            --image_init_idx_path "$init_file" --min_start_epoch "$min_start"
            --max_start_epoch "$max_start" --expert_dir "$buffer_dir"
            --data_path "$DATA_ROOT" --run_id "Tiny_${model}_${size}pct"
            --save_dir "$output")
        [[ $model == VGG11 ]] && command+=(--batch_syn 32)
        "${command[@]}"

        checkpoint_is_finite "$output/images_10000.pt" \
            "$output/labels_10000.pt" "$output/synlr_10000.pt"
        ;;
    *)
        echo "Invalid phase: $phase" >&2
        exit 2
        ;;
esac
