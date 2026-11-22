#!/bin/bash
set -euo pipefail

ROOT=/scratch/mmoslem3/MKDT
ACCOUNT=aip-bghaddar
LOGS="$ROOT/KRR/logs"
mkdir -p "$LOGS" "$ROOT/KRR/runs" "$ROOT/KRR/evaluations"

submit_distill() {
    local name=$1 dataset=$2 count=$3 lr=$4 time=$5
    sbatch --parsable --account="$ACCOUNT" --time="$time" --cpus-per-task=4 --mem=32G \
        --gres=gpu:l40s:1 --job-name="$name" --output="$LOGS/%x-%j.out" --error="$LOGS/%x-%j.err" \
        "$ROOT/KRR/slurm_job.sh" distill "$dataset" "$count" "$lr" "$name"
}

submit_eval() {
    local name=$1 dataset=$2 percentage=$3 x_syn=$4 time=$5 dependency=${6:-}
    local dep=()
    if [[ -n "$dependency" ]]; then
        dep=(--dependency="afterok:$dependency")
    fi
    sbatch --parsable --account="$ACCOUNT" --time="$time" --cpus-per-task=4 --mem=32G \
        --gres=gpu:l40s:1 --job-name="$name" --output="$LOGS/%x-%j.out" --error="$LOGS/%x-%j.err" \
        "${dep[@]}" "$ROOT/KRR/slurm_job.sh" eval "$dataset" "$percentage" "$x_syn"
}

c100_p01=$(submit_distill krr80_c100_p0p1 cifar100 50 1e-4 12:00:00)
c100_p1=$(submit_distill krr80_c100_p1 cifar100 500 1e-4 1-00:00:00)
tiny_p01=$(submit_distill krr80_tiny_p0p1 tinyimagenet 100 1e-5 1-00:00:00)
tiny_p1=$(submit_distill krr80_tiny_p1 tinyimagenet 1000 1e-5 3-00:00:00)

c100_eval_p01=$(submit_eval krre_c100_p0p1 cifar100 0.1 "$ROOT/KRR/runs/krr80_c100_p0p1/x_syn.pt" 12:00:00 "$c100_p01")
c100_eval_p1=$(submit_eval krre_c100_p1 cifar100 1 "$ROOT/KRR/runs/krr80_c100_p1/x_syn.pt" 12:00:00 "$c100_p1")
c100_eval_p2=$(submit_eval krre_c100_p2 cifar100 2 "$ROOT/KRR/synthetic_data/cifar100/krr_st/x_syn.pt" 12:00:00)
tiny_eval_p01=$(submit_eval krre_tiny_p0p1 tinyimagenet 0.1 "$ROOT/KRR/runs/krr80_tiny_p0p1/x_syn.pt" 1-00:00:00 "$tiny_p01")
tiny_eval_p1=$(submit_eval krre_tiny_p1 tinyimagenet 1 "$ROOT/KRR/runs/krr80_tiny_p1/x_syn.pt" 1-00:00:00 "$tiny_p1")
tiny_eval_p2=$(submit_eval krre_tiny_p2 tinyimagenet 2 "$ROOT/KRR/synthetic_data/tinyimagenet/krr_st/x_syn.pt" 1-00:00:00)

printf 'distill_c100_0.1\t%s\neval_c100_0.1\t%s\n' "$c100_p01" "$c100_eval_p01"
printf 'distill_c100_1\t%s\neval_c100_1\t%s\neval_c100_2_released\t%s\n' "$c100_p1" "$c100_eval_p1" "$c100_eval_p2"
printf 'distill_tiny_0.1\t%s\neval_tiny_0.1\t%s\n' "$tiny_p01" "$tiny_eval_p01"
printf 'distill_tiny_1\t%s\neval_tiny_1\t%s\neval_tiny_2_released\t%s\n' "$tiny_p1" "$tiny_eval_p1" "$tiny_eval_p2"
