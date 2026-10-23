#!/bin/bash
set -euo pipefail
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}" OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
output_root="$script_dir/ours_syntehtic"
teacher_root="$script_dir/../Evaluate_benchamrk/full_ssl_checkpoints_1000ep_3seeds"
dataset=TinyImageNet
mkdir -p "$output_root/logs"
missing=0

for model in ResNet18BN VGG11BN ConvNetBN; do
    teacher_seed=0
    if [[ "$model" == ResNet18BN ]]; then
        teacher_seed=1
    fi
    printf -v teacher_seed_dir 'seed_%03d' "$teacher_seed"
    teacher="$teacher_root/$dataset/${model%BN}/full/pt100/simclr/$teacher_seed_dir/encoder.pt"
    if [[ ! -s "$teacher" ]]; then
        echo "MISSING teacher: $teacher" | tee -a "$output_root/logs/${dataset}_missing.log"
        missing=1
        continue
    fi
    for percentage in 0.2 1 2; do
        case "$percentage" in
            0.2) iterations=10000 ;;
            1) iterations=8000 ;;
            2) iterations=6000 ;;
        esac
        log="$output_root/logs/${dataset}_${model}_pt${percentage}.log"
        echo "=== $dataset / $model / $percentage% / $iterations iterations / teacher seed $teacher_seed: $(date -u +%FT%TZ) ===" | tee -a "$log"
        python -u "$script_dir/method1.py" \
            --dataset "$dataset" --model "$model" --random_models "$model" \
            --percentage "$percentage" --Iteration "$iterations" --seed 0 \
            --lr_img 0.001 --batch_real 1024 --num_random_networks 1 \
            --temperature 0.2 --disp_weight 0 --pair_weight 0 \
            --dsa --distill_aug_strategy color_crop_cutout_flip_scale_rotate --distill_aug_mode M \
            --teacher_source pretrained --teacher_ssl simclr --teacher_seed "$teacher_seed" --teacher_path "$teacher_root" \
            --device cuda --num_eval 0 --resume --save_every 100 --feature_batch 128 \
            --save_path "$output_root/$dataset/$model/pt$percentage" 2>&1 | tee -a "$log"
    done
done
exit "$missing"
