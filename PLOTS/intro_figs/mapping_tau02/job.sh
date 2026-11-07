#!/bin/bash
set -euo pipefail
module --force purge
module load StdEnv/2023 python/3.11.5
source /home/mmoslem3/ENV/bin/activate
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export XDG_CACHE_HOME=/scratch/mmoslem3/MKDT/PLOTS/intro_figs/mapping_tau02/cache
export TORCH_HOME="$XDG_CACHE_HOME/torch" HF_HOME="$XDG_CACHE_HOME/hf" TMPDIR="${SLURM_TMPDIR:?}"
cd /scratch/mmoslem3/MKDT/PLOTS/intro_figs
python mapping_tau02/compute.py
pdflatex -interaction=nonstopmode -halt-on-error intro_final.tex > compile.stdout 2>&1
pdflatex -interaction=nonstopmode -halt-on-error intro_final.tex >> compile.stdout 2>&1
pdflatex -interaction=nonstopmode -halt-on-error intro_artwork.tex > compile_artwork.stdout 2>&1
pdflatex -interaction=nonstopmode -halt-on-error intro_artwork.tex >> compile_artwork.stdout 2>&1
pdftoppm -singlefile -scale-to 1800 -png intro_final.pdf intro_final_preview
printf 'MAPPING_AND_FIGURE_COMPLETE\n'
