"""Verify the exact frozen weights behind the failed TinyImageNet probes."""
import argparse
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent
sys.path.insert(0, str(BENCH))

import torch
from common import atomic_json, file_hash, load_encoder, state_hash

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--check-worker', action='store_true')
args = parser.parse_args()
if not os.environ.get('SLURM_JOB_ID'):
    parser.error('Run this check through Slurm')
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
groups = json.loads((HERE / 'remaining_cells.json').read_text())
selected = ('mkdt_TinyImageNet_p1_simclr_down', 'ours_TinyImageNet_p1_simclr_down',
            'mkdt_TinyImageNet_p2_simclr')
results = []
for group in (g for g in groups if g['id'] in selected):
    for seed in (0, 1):
        folder = BENCH / group['checkpoint_group'] / f'ssl_seed_{seed:03d}'
        config = json.loads((folder / 'config.json').read_text())
        marker = json.loads((folder / 'ssl_complete.json').read_text())
        encoder, _, saved = load_encoder(folder / 'encoder.pt', config['identity'], 'cpu')
        actual = state_hash(encoder)
        assert marker['complete'] and marker['identity'] == config['identity']
        assert saved['epochs'] == marker['epochs'] == group['epochs'] == 2000
        assert actual == saved['encoder_sha256'] == marker['encoder_sha256']
        record = {'group': group['id'], 'seed': seed, 'weights_match': True,
                  'encoder_sha256': actual, 'actual_file_sha256': file_hash(folder / 'encoder.pt'),
                  'recorded_file_sha256': marker['checkpoint_file_sha256']}
        del encoder
        if args.check_worker:
            from worker import saved_encoder
            assert saved_encoder(folder, config, marker) == folder / 'encoder.pt'
            wrong = {**marker, 'encoder_sha256': '0' * 64}
            try:
                saved_encoder(folder, config, wrong)
            except ValueError:
                pass
            else:
                raise AssertionError('Changed encoder weights must still be rejected')
        results.append(record)
        print(json.dumps(record), flush=True)
assert len(results) == 6
destination = 'encoder_validation.json' if args.check_worker else 'encoder_diagnosis.json'
atomic_json(HERE / destination, {'complete': True, 'worker_checked': args.check_worker,
                               'encoders': results})
