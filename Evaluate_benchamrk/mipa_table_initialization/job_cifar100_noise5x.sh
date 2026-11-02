#!/bin/bash
set -euo pipefail
: "${SLURM_JOB_ID:?Submit through Slurm}"
module --force purge
module load StdEnv/2023 python/3.11.5
source /home/mmoslem3/ENV/bin/activate
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK" NUMEXPR_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export XDG_CACHE_HOME="${SLURM_TMPDIR:?}/cache" MPLCONFIGDIR="$SLURM_TMPDIR/matplotlib"
export TORCH_HOME="$SLURM_TMPDIR/torch" HF_HOME="$SLURM_TMPDIR/hf" TMPDIR="$SLURM_TMPDIR"
cd /scratch/mmoslem3/MKDT
python -u Evaluate_benchamrk/mipa_table_initialization/cifar100_noise5x.py
cd /scratch/mmoslem3/MKDT/OT-SSL-DD
pdflatex -interaction=nonstopmode -halt-on-error -jobname=noise_preview -output-directory="$SLURM_TMPDIR" /scratch/mmoslem3/MKDT/Evaluate_benchamrk/mipa_components/pairs_export.hokJTv/preview.tex
if rg 'Overfull \\hbox' "$SLURM_TMPDIR/noise_preview.log"; then exit 1; fi
cp "$SLURM_TMPDIR/noise_preview.pdf" cifar100_1percent_pairs/figure_preview.pdf
pdftoppm -singlefile -scale-to 1800 -png "$SLURM_TMPDIR/noise_preview.pdf" cifar100_1percent_pairs/figure_preview
