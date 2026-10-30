#!/bin/bash
# Read-only preflight, on a Slurm CPU node rather than the shared login node.
set -euo pipefail
: "${SLURM_JOB_ID:?Submit through Slurm}"
module --force purge
module load StdEnv/2023 python/3.11.5
source /home/mmoslem3/ENV/bin/activate
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd /scratch/mmoslem3/MKDT
python - <<'PY'
import ast
from pathlib import Path
path = Path('Evaluate_benchamrk/mipa_similarity_ssl1k3probes/worker.py')
ast.parse(path.read_text(), filename=str(path))
print('Syntax passed')
PY
while IFS= read -r manifest; do
    python Evaluate_benchamrk/mipa_similarity_ssl1k3probes/worker.py "$manifest" /scratch/mmoslem3/data --check
done < "$1/preflight.tsv"
