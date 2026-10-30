#!/bin/bash
set -euo pipefail
module --force purge
module load StdEnv/2023 python/3.11.5
source /home/mmoslem3/ENV/bin/activate
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd /scratch/mmoslem3/MKDT
python - <<'PY'
import ast
from pathlib import Path
for name in ('Evaluate_benchamrk/mipa_similarity_ablation/worker.py', '/scratch/mmoslem3/MKDT/Evaluate_benchamrk/mipa_similarity_ablation/batches/batch_20261030T010548Z_aPHws0/method1.py'):
    compile(ast.parse(Path(name).read_text()), name, 'exec')
print('Syntax passed')
PY
while IFS=$'\t' read -r source manifest; do
    python -u Evaluate_benchamrk/mipa_similarity_ablation/worker.py "$manifest" /scratch/mmoslem3/data --check
done < /scratch/mmoslem3/MKDT/Evaluate_benchamrk/mipa_similarity_ablation/batches/batch_20261030T010548Z_aPHws0/preflight.tsv
