"""Continue the saved TinyImageNet 2% run with its exact original parameters."""
import os
from pathlib import Path
import runpy
import sys

import torch

bench = Path(__file__).resolve().parent.parent
script = bench.parent / 'OT-SSL-DD/method1.py'
folder = script.parent / 'ours_syntehtic/TinyImageNet/ResNet18BN/pt2'
checkpoint = folder / 'res_IIC_TinyImageNet_ResNet18BN_2percent_final.pt'
if (not os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_JOB_ACCOUNT') != 'aip-yiweilu'
        or os.environ.get('SLURM_CLUSTER_NAME', os.environ.get('CC_CLUSTER')) != 'vulcan'
        or not torch.cuda.is_available() or torch.cuda.device_count() != 1):
    raise RuntimeError('Use the Lu Vulcan GPU launcher')
saved = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
params = saved['run_args']
if ([params[k] for k in ('dataset', 'model', 'percentage', 'Iteration')]
        != ['TinyImageNet', 'ResNet18BN', 2, 6000] or not 0 < saved['iteration'] <= 6000
        or any(k not in saved for k in ('optimizer', 'scheduler', 'rng', 'objective_rng', 'initial_data'))):
    raise ValueError('Not the original resumable TinyImageNet 2% checkpoint')
print('CONTINUE saved iteration %d -> 6000 with original run_args, Adam and RNG state' % saved['iteration'], flush=True)
argv = [str(script)]
for key, value in params.items():
    if isinstance(value, bool):
        argv.append('--' + (key if value else 'no-' + key))
    else:
        argv += ['--' + key] + [str(v) for v in (value if isinstance(value, list) else [value])]
argv += ['--resume', '--device', 'cuda', '--data_path', sys.argv[1], '--save_path', str(folder),
         '--save_every', '100', '--num_eval', '0']
del saved
sys.path.insert(0, str(script.parent))
sys.argv = argv
runpy.run_path(str(script), run_name='__main__')
final = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
if not final['complete'] or final['iteration'] != 6000 or final['run_args'] != params:
    raise ValueError('Generation did not finish the original 6000-iteration run')
