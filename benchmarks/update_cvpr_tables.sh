#!/bin/bash
#SBATCH --account=aip-boyuwang
#SBATCH --job-name=update_cvpr_tables
#SBATCH --time=0-00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --output=benchmarks/update-cvpr-tables-%j.out

set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the MKDT project directory}}"
cd "$PROJECT_ROOT"
module load StdEnv/2023 python/3.11.5
export PYTHONDONTWRITEBYTECODE=1

CIFAR100_RESULTS="${CIFAR100_RESULTS:-$PROJECT_ROOT/benchmarks/cifar100/results/batch_20261009T163141Z_eMeTVJ}"
TINYIMAGENET_RESULTS="${TINYIMAGENET_RESULTS:-$PROJECT_ROOT/benchmarks/tinyimagenet/results/batch_20261009T163143Z_D8Lnde}"
CROSS_RESULTS="${CROSS_RESULTS:-$PROJECT_ROOT/benchmarks/cross_downstream/results/batch_20261009T171449Z_o91BKP}"
SYNTHETIC_RESULTS="${SYNTHETIC_RESULTS:-$PROJECT_ROOT/benchmarks/synthetic_new/results/batch_20261010T004533Z_wSzguy}"

python3 benchmarks/main/report.py \
    --dataset CIFAR100 --results "$CIFAR100_RESULTS" \
    --output "$CIFAR100_RESULTS/cifar100_main.tex"
python3 benchmarks/main/report.py \
    --dataset TinyImageNet --results "$TINYIMAGENET_RESULTS" \
    --output "$TINYIMAGENET_RESULTS/tinyimagenet_main.tex"
python3 benchmarks/cross_downstream/report.py \
    --results "$CROSS_RESULTS" --output-dir "$CROSS_RESULTS"

mkdir -p "$PROJECT_ROOT/CVPR_tables"
cp "$CIFAR100_RESULTS/cifar100_main.tex" "$PROJECT_ROOT/CVPR_tables/cifar100_main.tex"
cp "$TINYIMAGENET_RESULTS/tinyimagenet_main.tex" "$PROJECT_ROOT/CVPR_tables/tinyimagenet_main.tex"
cp "$CROSS_RESULTS/cifar100_downstream.tex" "$PROJECT_ROOT/CVPR_tables/cifar100_downstream.tex"
cp "$CROSS_RESULTS/tinyimagenet_downstream.tex" "$PROJECT_ROOT/CVPR_tables/tinyimagenet_downstream.tex"

# Overlay complete synthetic-data results into the Ours rows after refreshing
# the baseline tables, so incomplete synthetic configurations remain blank.
python3 benchmarks/benchmark_synthetic_new.py report \
    --batch "$SYNTHETIC_RESULTS" --templates "$PROJECT_ROOT/CVPR_tables" \
    --output "$SYNTHETIC_RESULTS/tables_current"
cp "$SYNTHETIC_RESULTS/tables_current/cifar10_main.tex" "$PROJECT_ROOT/CVPR_tables/cifar10_main.tex"
cp "$SYNTHETIC_RESULTS/tables_current/cifar100_main.tex" "$PROJECT_ROOT/CVPR_tables/cifar100_main.tex"
cp "$SYNTHETIC_RESULTS/tables_current/cifar10_downstream.tex" "$PROJECT_ROOT/CVPR_tables/cifar10_downstream.tex"
cp "$SYNTHETIC_RESULTS/tables_current/cifar100_downstream.tex" "$PROJECT_ROOT/CVPR_tables/cifar100_downstream.tex"
