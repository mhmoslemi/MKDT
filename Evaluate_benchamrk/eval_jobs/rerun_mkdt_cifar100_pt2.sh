#!/bin/bash
set -euo pipefail
: "${SLURM_JOB_ID:?Run through sbatch}"
output=${1:?Pass a new evaluation output directory}
module load StdEnv/2023 python/3.11.5
source /home/mmoslem3/ENV/bin/activate
cd /scratch/mmoslem3/MKDT
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export XDG_CACHE_HOME="$SLURM_TMPDIR/cache" TORCH_HOME="$SLURM_TMPDIR/torch"
python -u - "$output" <<'PY'
import sys
sys.path.insert(0, 'Evaluate_benchamrk/eval_jobs')
import interactive_mkdt_2k as mkdt

mkdt.MODELS = ('ResNet18',)
mkdt.SIZES = (2,)
mkdt.evaluation_targets = lambda source, model: (source,)
sys.argv = [sys.argv[0], 'CIFAR100', '--data-root', '/scratch/mmoslem3/data', '--output', sys.argv[1]]
raise SystemExit(mkdt.main())
PY
