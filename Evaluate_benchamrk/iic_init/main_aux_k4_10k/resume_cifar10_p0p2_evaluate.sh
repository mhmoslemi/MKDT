#!/bin/bash
set -euo pipefail
: "${SLURM_JOB_ID:?Submit through Slurm}"
module --force purge
module load StdEnv/2023 python/3.11.5
source /home/mmoslem3/ENV/bin/activate
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export XDG_CACHE_HOME="${SLURM_TMPDIR:?}/cache"
export TORCH_HOME="$XDG_CACHE_HOME/torch" HF_HOME="$XDG_CACHE_HOME/hf" MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
export TMPDIR="$SLURM_TMPDIR"
cd /scratch/mmoslem3/MKDT/Evaluate_benchamrk
exec python -u - <<'PY'
import json
from pathlib import Path
import runpy
import sys
import torch

folder = Path('../OT-SSL-DD/ours_syntehtic_iic_main_aux_k4_10k/CIFAR10/ResNet18BN/pt0.2/evaluation/simclr/ssl_seed_000')
checkpoint = folder / 'training.pt'
assert checkpoint.is_file(), 'Resume checkpoint missing; refusing to restart from scratch'
config = json.loads((folder / 'config.json').read_text())
assert config['signature']['protocol']['ssl']['subset_epochs'] == 2000
state = torch.load(checkpoint, map_location='cpu', weights_only=True)
assert state['identity'] == config['identity'], 'Checkpoint identity changed'
assert 0 < state['epoch'] <= 2000 and state['steps_per_epoch'] == 1
assert all(key in state for key in ('model', 'optimizer', 'rng'))
print('RESUME confirmed: saved SSL epoch %d/2000; model, optimizer and RNG state retained.' % state['epoch'], flush=True)
del state
sys.argv = ['iic_init/distill_main_aux.py', '--job',
            'iic_init/main_aux_k4_10k/jobs/CIFAR10_p0p2.json', '--stage', 'evaluate']
runpy.run_path('iic_init/distill_main_aux.py', run_name='__main__')
PY
