#!/bin/bash
set -euo pipefail
: "${SLURM_JOB_ID:?Run through Slurm}"
module --force purge
module load StdEnv/2023 python/3.11.5
source /home/mmoslem3/ENV/bin/activate
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export XDG_CACHE_HOME="${SLURM_TMPDIR:?}/cache" TMPDIR="$SLURM_TMPDIR"
python /scratch/mmoslem3/MKDT/Evaluate_benchamrk/mipa_components/pairs_export.hokJTv/export.py
cd /scratch/mmoslem3/MKDT/OT-SSL-DD
pdflatex -interaction=nonstopmode -halt-on-error -output-directory="$SLURM_TMPDIR" /scratch/mmoslem3/MKDT/Evaluate_benchamrk/mipa_components/pairs_export.hokJTv/preview.tex
if rg 'Overfull \\hbox' "$SLURM_TMPDIR/preview.log"; then exit 1; fi
cp "$SLURM_TMPDIR/preview.pdf" /scratch/mmoslem3/MKDT/Evaluate_benchamrk/mipa_components/pairs_export.hokJTv/preview.pdf
pdftoppm -singlefile -scale-to 1600 -png "$SLURM_TMPDIR/preview.pdf" /scratch/mmoslem3/MKDT/Evaluate_benchamrk/mipa_components/pairs_export.hokJTv/preview
