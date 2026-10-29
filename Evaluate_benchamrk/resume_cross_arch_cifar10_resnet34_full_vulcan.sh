#!/bin/bash
#SBATCH --job-name=xa_C10_R34_full_finish
#SBATCH --account=aip-yiweilu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=16:55:00
#SBATCH --chdir=/scratch/mmoslem3/MKDT/Evaluate_benchamrk
#SBATCH --output=/scratch/mmoslem3/MKDT/Evaluate_benchamrk/cross_arch_ablation/logs/%x-%j.out
#SBATCH --error=/scratch/mmoslem3/MKDT/Evaluate_benchamrk/cross_arch_ablation/logs/%x-%j.err
set -euo pipefail
: "${SLURM_JOB_ID:?Submit this script with sbatch}"

module --force purge
module load StdEnv/2023 python/3.11.5
source /home/mmoslem3/ENV/bin/activate
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export HF_HOME=/scratch/mmoslem3/MKDT/Evaluate_benchamrk/cross_arch_ablation/cache/hf
export XDG_CACHE_HOME=/scratch/mmoslem3/MKDT/Evaluate_benchamrk/cross_arch_ablation/cache/xdg
export TORCH_HOME=/scratch/mmoslem3/MKDT/Evaluate_benchamrk/cross_arch_ablation/cache/torch
export TMPDIR="${SLURM_TMPDIR:?}"
cd /scratch/mmoslem3/MKDT/Evaluate_benchamrk

python -u - <<'PY'
import json
from pathlib import Path
import torch

folder = Path('cross_arch_ablation/results/CIFAR10/ResNet34/full/pt100/simclr/ssl_seed_000')
saved = torch.load(folder / 'training.pt', map_location='cpu', weights_only=True, mmap=True)
assert saved['identity'] == json.loads((folder / 'config.json').read_text())['identity']
assert 200 <= saved['epoch'] <= 1000, 'Expected the saved epoch-200 checkpoint or a later save'
assert saved['steps'] == saved['epoch'] * saved['steps_per_epoch']
assert 'optimizer' in saved and 'rng' in saved
print('RESUME VERIFIED: epoch %d / 1000, retaining optimizer and RNG state' % saved['epoch'], flush=True)
PY

job_batch=cross_arch_ablation/batches/batch_20261026T062122Z_smjkx2
python -u cross_arch_ablation/worker.py "$job_batch/xa_CIFAR10_ResNet34_full_p100_s0_e1000.json" /scratch/mmoslem3/data
python -u cross_arch_ablation/worker.py "$job_batch/xa_CIFAR10_ResNet34_full_p100_s0_probe.json" /scratch/mmoslem3/data

python -u - <<'PY'
import fcntl
import json
import os
from pathlib import Path

run = Path('cross_arch_ablation')
summary = json.loads((run / 'results/CIFAR10/ResNet34/full/pt100/simclr/summary.json').read_text())
assert summary['complete'] and summary['source'] == 'CIFAR10' and summary['model'] == 'ResNet34'
assert summary['kind'] == 'full' and summary['subset'] == 100
cells = {(cell['target'], cell['labels']): cell for cell in summary['cells']}
table = Path('../CVPR_tables/ablations/cross_arch_combined.tex')
with (run / '.report.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    text = table.read_text()
    for labels in (1, 5):
        tag = 'crossarch backbone=ResNet34 kind=full pt=100 lbl=%d' % labels
        matches = [line for line in text.splitlines() if line.endswith('% ' + tag)]
        assert len(matches) == 1, 'Missing or duplicate combined-table row'
        old = matches[0]
        fields = old.split('&')
        assert len(fields) == 10
        for column, target in enumerate(('CIFAR10', 'CIFAR100', 'TinyImageNet'), start=4):
            cell = cells[target, labels]
            assert cell['complete'] and cell['completed_probes'] == cell['expected_probes'] == 5
            color = r'\cellcolor{gray!12}' if column == 4 else ''
            fields[column] = ' ' + color + r'\dsacc{%.2f}{%.2f}' % (cell['mean_percent'], cell['std_percent']) + ' '
        text = text.replace(old, '&'.join(fields))
    temporary = table.with_name('.%s.%d.tmp' % (table.name, os.getpid()))
    temporary.write_text(text)
    temporary.replace(table)
print('COMPLETE: 1000 SSL epochs, 30 frozen probes, six full-data combined-table cells filled')
PY
