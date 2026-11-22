#!/bin/bash
set -euo pipefail

ROOT=/scratch/mmoslem3/MKDT
ACCOUNT=aip-bghaddar
LOGS="$ROOT/KRR/logs"
mkdir -p "$LOGS" "$ROOT/KRR/runs" "$ROOT/KRR/evaluations"

submit_distill() {
    sbatch --parsable --account="$ACCOUNT" --time="$3" --cpus-per-task=4 --mem=6G \
        --gres=gpu:l40s:1 --dependency="afterok:$4" --job-name="$1" \
        --output="$LOGS/%x-%j.out" --error="$LOGS/%x-%j.err" \
        "$ROOT/KRR/slurm_job.sh" distill cifar10 "$2" 1e-4 "$1"
}

submit_ssl() {
    sbatch --parsable --account="$ACCOUNT" --time="$5" --cpus-per-task=4 --mem=6G \
        --gres=gpu:l40s:1 --dependency="afterok:$4" --job-name="$1" \
        --output="$LOGS/%x-%j.out" --error="$LOGS/%x-%j.err" \
        "$ROOT/KRR/slurm_job.sh" eval_ssl cifar10 "$2" "$ROOT/KRR/runs/krr80_c10_p$3/x_syn.pt"
}

submit_probes() {
    local percentage=$1 run_tag=$2 dependency=$3 labels seed
    for labels in 1 5; do
        for seed in 0 1 2 3 4; do
            sbatch --parsable --account="$ACCOUNT" --time=00:08:00 --cpus-per-task=4 --mem=6G \
                --gres=gpu:l40s:1 --dependency="afterok:$dependency" \
                --job-name="krrp_c10_p${percentage}_l${labels}s${seed}" \
                --output="$LOGS/%x-%j.out" --error="$LOGS/%x-%j.err" \
                "$ROOT/KRR/slurm_job.sh" eval_probe cifar10 "$percentage" \
                "$ROOT/KRR/runs/krr80_c10_p$run_tag/x_syn.pt" "$labels" "$seed"
        done
    done
}

for percentage in 0p2 1 2; do
    [[ ! -e "$ROOT/KRR/runs/krr80_c10_p$percentage/x_syn.pt" ]] || { echo "Output exists for $percentage" >&2; exit 2; }
done

selftest=$(sbatch --parsable --account="$ACCOUNT" --time=00:05:00 --cpus-per-task=2 --mem=4G \
    --job-name=krr_c10_selftest --output="$LOGS/%x-%j.out" --error="$LOGS/%x-%j.err" \
    "$ROOT/KRR/slurm_job.sh" selftest)

d02=$(submit_distill krr80_c10_p0p2 100 06:00:00 "$selftest")
d1=$(submit_distill krr80_c10_p1 500 14:00:00 "$selftest")
d2=$(submit_distill krr80_c10_p2 1000 1-06:00:00 "$selftest")
s02=$(submit_ssl krrs_c10_p0p2 0.2 0p2 "$d02" 00:25:00)
s1=$(submit_ssl krrs_c10_p1 1 1 "$d1" 00:31:00)
s2=$(submit_ssl krrs_c10_p2 2 2 "$d2" 00:51:00)

echo "selftest $selftest"
echo "0.2% distill $d02 ssl $s02"
submit_probes 0.2 0p2 "$s02"
echo "1% distill $d1 ssl $s1"
submit_probes 1 1 "$s1"
echo "2% distill $d2 ssl $s2"
submit_probes 2 2 "$s2"
