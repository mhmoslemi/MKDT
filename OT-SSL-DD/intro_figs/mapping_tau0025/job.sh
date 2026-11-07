#!/bin/bash
#SBATCH --job-name=intro_mapping_tau0025
#SBATCH --account=aip-yiweilu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=00:02:00
#SBATCH --chdir=/scratch/mmoslem3/MKDT/OT-SSL-DD/intro_figs
#SBATCH --output=/scratch/mmoslem3/MKDT/OT-SSL-DD/intro_figs/mapping_tau0025/%x-%j.out
set -euo pipefail
module --force purge
module load StdEnv/2023 python/3.11.5
source /home/mmoslem3/ENV/bin/activate
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export XDG_CACHE_HOME=/scratch/mmoslem3/MKDT/OT-SSL-DD/intro_figs/mapping_tau0025/cache
export TMPDIR="${SLURM_TMPDIR:?}"
python mapping_tau0025/compute.py
pdflatex -interaction=nonstopmode -halt-on-error intro_final.tex > compile.stdout 2>&1
pdflatex -interaction=nonstopmode -halt-on-error intro_final.tex >> compile.stdout 2>&1
pdflatex -interaction=nonstopmode -halt-on-error intro_artwork.tex > compile_artwork.stdout 2>&1
pdflatex -interaction=nonstopmode -halt-on-error intro_artwork.tex >> compile_artwork.stdout 2>&1
pdftoppm -singlefile -scale-to 1800 -png intro_final.pdf intro_final_preview
cp intro.tex intro_final.pdf intro_artwork.pdf mapping_tau0025/
printf 'MAPPING_AND_FIGURE_COMPLETE\n'
