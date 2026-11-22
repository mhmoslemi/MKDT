#!/bin/bash
set -euo pipefail

ROOT=/scratch/mmoslem3/MKDT
source /home/mmoslem3/ENV/bin/activate
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export XDG_CACHE_HOME="$ROOT/KRR/cache"
mkdir -p "$XDG_CACHE_HOME"
cd "$ROOT"

mode=$1
shift
if [[ "$mode" == selftest ]]; then
    python -m py_compile KRR/selfsup_dd/train.py KRR/evaluate_krr.py KRR/subset_x_syn.py
    python KRR/evaluate_krr.py --self-test
    (cd KRR/selfsup_dd && python -c 'import train; assert set(train.DATA_NAMES) == {"cifar10", "cifar100", "tinyimagenet"}')
elif [[ "$mode" == subset_krr || "$mode" == subset_tiny_krr ]]; then
    python -u KRR/subset_x_syn.py "$1" "$2" "$3"
elif [[ "$mode" == distill || "$mode" == distill_resume || "$mode" == distill_continue ]]; then
    dataset=$1
    count=$2
    lr=$3
    run_name=$4
    destination="$ROOT/KRR/runs/$run_name"
    if [[ "$mode" == distill && -e "$destination/x_syn.pt" ]]; then
        echo "Refusing to replace existing synthetic data: $destination/x_syn.pt" >&2
        exit 2
    fi
    if [[ "$mode" == distill_continue && -e "$destination/x_syn.pt" ]]; then
        echo "Refusing to replace existing continuation: $destination/x_syn.pt" >&2
        exit 2
    fi
    extra=(--end_step "${5:-80000}")
    if [[ "$mode" == distill_resume ]]; then
        extra+=(--resume "$destination/resume.pt")
    elif [[ "$mode" == distill_continue ]]; then
        extra+=(--init_run "$ROOT/KRR/runs/$6" --start_step "$7")
    fi
    cd "$ROOT/KRR/selfsup_dd"
    python -u train.py --data_name "$dataset" --num_images "$count" --outer_lr "$lr" \
        --outer_iteration "${8:-80000}" --outer_batch_size 1024 --seed 0 --num_workers "${SLURM_CPUS_PER_TASK:-1}" \
        --data_root /scratch/mmoslem3/data --teacher_dir "$ROOT/KRR/teacher_ckpt" \
        --save_dir "$ROOT/KRR/runs" --exp_name "$run_name" --save_every 2000 \
        --print_every 100 --skip_meta_test "${extra[@]}"
elif [[ "$mode" == eval ]]; then
    dataset=$1
    percentage=$2
    x_syn=$3
    python -u KRR/evaluate_krr.py "$dataset" "$percentage" "$x_syn" \
        --data-root /scratch/mmoslem3/data --output "$ROOT/KRR/evaluations" \
        --workers "${SLURM_CPUS_PER_TASK:-1}"
elif [[ "$mode" == eval_ssl ]]; then
    python -u KRR/evaluate_krr.py "$1" "$2" "$3" --stage ssl \
        --data-root /scratch/mmoslem3/data --output "${4:-$ROOT/KRR/evaluations}" \
        --workers "${SLURM_CPUS_PER_TASK:-1}"
elif [[ "$mode" == eval_probe ]]; then
    python -u KRR/evaluate_krr.py "$1" "$2" "$3" --stage probe --labels "$4" --probe-seed "$5" \
        --data-root /scratch/mmoslem3/data --output "${6:-$ROOT/KRR/evaluations}" \
        --workers "${SLURM_CPUS_PER_TASK:-1}"
else
    echo "Unknown mode: $mode" >&2
    exit 2
fi
