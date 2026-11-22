#!/bin/bash
set -euo pipefail

ROOT=/scratch/mmoslem3/MKDT
ACCOUNT=aip-bghaddar
LOGS="$ROOT/KRR/logs"
mkdir -p "$LOGS" "$ROOT/KRR/evaluations"

submit_ssl() {
    sbatch --parsable --account="$ACCOUNT" --time="$5" --cpus-per-task=4 --mem=32G \
        --gres=gpu:l40s:1 --job-name="$1" --output="$LOGS/%x-%j.out" --error="$LOGS/%x-%j.err" \
        "$ROOT/KRR/slurm_job.sh" eval_ssl "$2" 2 "$4"
}

submit_probes() {
    local prefix=$1 dataset=$2 x_syn=$3 dependency=$4 time=$5
    local labels seed
    for labels in 1 5; do
        for seed in 0 1 2 3 4; do
            sbatch --parsable --account="$ACCOUNT" --time="$time" --cpus-per-task=4 --mem=16G \
                --gres=gpu:l40s:1 --dependency="afterok:$dependency" \
                --job-name="${prefix}_l${labels}s${seed}" --output="$LOGS/%x-%j.out" --error="$LOGS/%x-%j.err" \
                "$ROOT/KRR/slurm_job.sh" eval_probe "$dataset" 2 "$x_syn" "$labels" "$seed"
        done
    done
}

c100_x="$ROOT/KRR/synthetic_data/cifar100/krr_st/x_syn.pt"
tiny_x="$ROOT/KRR/synthetic_data/tinyimagenet/krr_st/x_syn.pt"
c100_ssl=$(submit_ssl krrs_c100_p2 cifar100 2 "$c100_x" 02:00:00)
tiny_ssl=$(submit_ssl krrs_tiny_p2 tinyimagenet 2 "$tiny_x" 06:00:00)
echo "CIFAR100 SSL $c100_ssl"
submit_probes krrp_c100 cifar100 "$c100_x" "$c100_ssl" 00:20:00
echo "TinyImageNet SSL $tiny_ssl"
submit_probes krrp_tiny tinyimagenet "$tiny_x" "$tiny_ssl" 00:30:00
