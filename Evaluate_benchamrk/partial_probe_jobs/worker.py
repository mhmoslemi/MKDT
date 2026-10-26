"""Finish the approved frozen linear probes per cell; no encoder optimization."""
import argparse
import contextlib
import fcntl
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent
sys.path.insert(0, str(BENCH))

import torch

from common import (atomic_json, atomic_save, build_encoder, code_hashes, digest_json,
                    exclusive_run, file_hash, load_encoder, state_hash)
from data import RawDataset, SOURCE_COUNTS
from evaluate import linear_probe
from run import evaluation_targets


def read(path):
    return json.loads(Path(path).read_text())


def inside(value):
    path = (BENCH / value).resolve()
    if not path.is_relative_to(BENCH) or path == BENCH:
        raise ValueError('Probe/checkpoint paths must stay inside Evaluate_benchamrk')
    return path


def saved_encoder(folder, config, marker):
    checkpoint = folder / 'encoder.pt'
    if checkpoint.exists():
        if file_hash(checkpoint) != marker['checkpoint_file_sha256']:
            raise ValueError('Saved encoder file changed')
    else:
        # Export the already-final encoder weights; never continue SSL training.
        cache = BENCH / 'partial_probe_results/recovered_encoders' / marker['encoder_sha256']
        checkpoint = cache / 'encoder.pt'
        cache.mkdir(parents=True, exist_ok=True)
        with (cache / '.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not checkpoint.exists():
                original = folder / 'training.pt'
                state = torch.load(original, map_location='cpu', weights_only=True, mmap=True)
                if state['identity'] != config['identity'] or state['epoch'] != marker['epochs']:
                    raise ValueError('Rolling state is not the completed encoder; no SSL fallback')
                signature = config['signature']
                encoder, _ = build_encoder(signature['model'], signature['protocol']['input_size'][signature['source']])
                weights = {k.removeprefix('encoder.'): v for k, v in state['model'].items() if k.startswith('encoder.')}
                encoder.load_state_dict(weights, strict=True)
                if state_hash(encoder) != marker['encoder_sha256']:
                    raise ValueError('Recovered weights do not match the completed encoder')
                saved = {'kind': 'ssl_reference_encoder', 'complete': True, 'identity': config['identity'],
                    'source': signature['source'], 'model': signature['model'], 'method': signature['ssl'],
                    'seed': config['identity']['seed'], 'epochs': state['epoch'], 'steps': state['steps'],
                    'input_size': signature['protocol']['input_size'][signature['source']],
                    'normalization': signature['protocol']['normalization'][signature['source']],
                    'effective_batch': min(int(SOURCE_COUNTS[signature['source']] * signature['subset'] / 100),
                                           signature['protocol']['ssl']['batch_size']),
                    'encoder': encoder.state_dict(), 'encoder_sha256': marker['encoder_sha256']}
                atomic_save(checkpoint, saved)
                atomic_json(cache / 'recovery.json', {'source_state': str(original),
                    'source_state_sha256': file_hash(original), 'epoch': state['epoch'],
                    'encoder_sha256': marker['encoder_sha256'], 'ssl_updates': 0})
                del state, encoder, weights
    encoder, _, saved = load_encoder(checkpoint, config['identity'], 'cpu')
    del encoder
    signature = config['signature']
    expected = {'source': signature['source'], 'model': 'ResNet18', 'method': signature['ssl'],
        'seed': config['identity']['seed'], 'epochs': marker['epochs'], 'encoder_sha256': marker['encoder_sha256'],
        'input_size': signature['protocol']['input_size'][signature['source']],
        'normalization': signature['protocol']['normalization'][signature['source']]}
    if any(saved.get(k) != v for k, v in expected.items()):
        raise ValueError('Saved encoder metadata differs from the requested frozen encoder')
    return checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('data_root', type=Path)
    args = parser.parse_args()
    if (not os.environ.get('SLURM_JOB_ID')
            or os.environ.get('SLURM_CLUSTER_NAME', os.environ.get('CC_CLUSTER')) not in ('vulcan', 'killarney')
            or os.environ.get('SLURM_JOB_ACCOUNT') not in ('aip-boyuwang', 'aip-yiweilu')
            or not torch.cuda.is_available() or torch.cuda.device_count() != 1):
        parser.error('Use a Wang/Lu single-GPU launcher on Vulcan or Killarney')
    if not args.manifest.resolve().is_relative_to(HERE / 'batches'):
        parser.error('Unexpected probe job manifest')
    job = read(args.manifest)
    definitions = read(HERE / 'cells.json')
    if job.get('remaining'):
        definitions += read(HERE / 'remaining_cells.json')
    if job.get('account', 'aip-boyuwang') != os.environ.get('SLURM_JOB_ACCOUNT'):
        raise ValueError('Job account differs from its submission plan')
    groups = [g for g in definitions if g['id'] == job['group_id']]
    if len(groups) != 1:
        raise ValueError('Unapproved probe completion group')
    group, cell = groups[0], job['cell']
    if (any(cell[k] != group[k] for k in ('source', 'kind', 'subset', 'ssl'))
            or job['cell_key'] != '|'.join(map(str, (cell['source'], 'ResNet18', cell['kind'],
                                                    cell['subset'], cell['ssl'], cell['target'], cell['labels'])))
            or [cell['target'], cell['labels']] not in group['cells']
            or job['ssl_epochs'] != group['epochs'] or job['code_hashes'] != code_hashes()
            or len(job['samples']) != (10 if group['scheme'] == 'ten_ssl' else 6)
            or job['scheme'] != group['scheme']
            or (group['scheme'] == 'ten_ssl' and group['kind'] != 'random')
            or job['repeated_encoders'] != (group['scheme'] in ('one_ssl', 'four_ssl'))):
        raise ValueError('Probe job scope, epochs or code changed')
    pairs = ([(0, p) for p in range(6)] if group['scheme'] == 'one_ssl' else
             [(0, 0), (1, 1), (2, 2), (4, 4), (0, 3), (1, 5)] if group['scheme'] == 'four_ssl' else
             [(s, s) for s in range(10)] if group['scheme'] == 'ten_ssl' else
             [(s, p) for s in (0, 1) for p in (0, 1, 2)])
    if [(s['ssl_seed'], s['probe_seed']) for s in job['samples']] != pairs:
        raise ValueError('Unexpected encoder/probe pairing')
    tag = 'seed_' if group['scheme'] in ('four_ssl', 'ten_ssl') else 'ssl_seed_'
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    args.data_root = args.data_root.resolve()
    train_set, test_set = RawDataset(args.data_root, cell['target'], True), RawDataset(args.data_root, cell['target'], False)
    summary_path = inside(job['summary_file'])
    expected_directory = BENCH / 'partial_probe_results' / group['id'] / f'{cell["target"]}_lbl{cell["labels"]}'
    if summary_path != expected_directory / 'summary.json':
        raise ValueError('Unexpected completion summary path')
    samples, fingerprints = [], set()
    with exclusive_run(expected_directory), contextlib.ExitStack() as locks:
        encoders = {}
        for sample in job['samples']:
            seed = sample['ssl_seed']
            folder = inside(sample['checkpoint_dir'])
            expected = inside(group['checkpoint_group']) / f'{tag}{seed:03d}'
            if folder != expected:
                raise ValueError('Unexpected checkpoint path')
            if seed not in encoders:
                output = inside(group['output_group']) / f'{tag}{seed:03d}'
                for directory in sorted({folder, output}):
                    directory.mkdir(parents=True, exist_ok=True)
                    lock = locks.enter_context((directory / '.lock').open('a'))
                    fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
                config, marker = read(folder / 'config.json'), read(folder / 'ssl_complete.json')
                signature = config['signature']
                if ((sample['config_sha256'] is not None and file_hash(folder / 'config.json') != sample['config_sha256'])
                        or digest_json(signature) != config['identity']['signature']
                        or signature['code_hashes'] != code_hashes()
                        or not marker['complete'] or marker['identity'] != config['identity']
                        or config['identity']['seed'] != seed or signature['protocol']['ssl']['subset_epochs'] != group['epochs']
                        or marker['epochs'] != group['epochs']
                        or (sample['encoder_sha256'] is not None and marker['encoder_sha256'] != sample['encoder_sha256'])
                        or (signature['source'], signature['model'], signature['selection_method'], signature['subset'], signature['ssl'])
                        != (group['source'], 'ResNet18', group['kind'], group['subset'], group['ssl'])
                        or signature['protocol']['evaluation']['epochs'] != 100):
                    raise ValueError('Completed encoder provenance changed')
                checkpoint = saved_encoder(folder, config, marker)
                if group['kind'] == 'ours' and output != folder:
                    reference = {'checkpoint': str(checkpoint), 'checkpoint_file_sha256': file_hash(checkpoint),
                        'driver_sha256': file_hash(Path(__file__)),
                        'targets': [t for t in evaluation_targets(group['source'], 'ResNet18') if t != group['source']],
                        'scope': 'downstream'}
                    with (output / '.metadata.lock').open('a') as metadata_lock:
                        fcntl.flock(metadata_lock, fcntl.LOCK_EX)
                        reference_path = output / 'checkpoint_reference.json'
                        if reference_path.exists():
                            previous = read(reference_path)
                            if any(previous.get(k) != value for k, value in reference.items() if k != 'driver_sha256'):
                                raise ValueError('Downstream checkpoint reference changed')
                            reference = previous
                        for name, value in {'config.json': config, 'ssl_complete.json': {**marker, 'checkpoint': str(checkpoint)},
                                            'checkpoint_reference.json': reference}.items():
                            path = output / name
                            if path.exists() and read(path) != value:
                                raise ValueError(f'Downstream encoder reference changed: {path}')
                            if not path.exists():
                                atomic_json(path, value)
                encoders[seed] = (checkpoint, config, marker, file_hash(checkpoint))
            checkpoint, config, marker, checksum = encoders[seed]
            destination = inside(sample['result_file'])
            output = inside(group['output_group']) / f'{tag}{seed:03d}'
            suffix = '' if group['scheme'] in ('four_ssl', 'ten_ssl') and sample['probe_seed'] == seed else f'_probe{sample["probe_seed"]}'
            allowed = {output / 'evaluations' / f'{cell["target"]}_lbl{cell["labels"]}{suffix}.json',
                       expected_directory / f'ssl_seed_{seed}_probe_{sample["probe_seed"]}.json'}
            if destination not in allowed:
                raise ValueError('Unexpected probe output path')
            lock_name = f'{cell["target"]}_lbl{cell["labels"]}_ssl{seed}_probe{sample["probe_seed"]}'
            with exclusive_run(output / '.probe_locks' / lock_name):
                result = linear_probe(checkpoint, config['identity'], config['signature']['protocol'],
                    group['source'], cell['target'], train_set, test_set, cell['labels'], sample['probe_seed'],
                    destination, workers=0, device='cuda')
            score = result['accuracy_percent']
            if (not result['complete'] or not result['encoder_unchanged']
                    or result['encoder_sha256'] != marker['encoder_sha256'] or not math.isfinite(score)):
                raise ValueError('Invalid frozen probe result')
            fingerprints.add((result['expected']['train_fingerprint'], result['expected']['test_fingerprint']))
            if len(fingerprints) != 1:
                raise ValueError('Cannot combine different target datasets')
            samples.append({'ssl_seed': seed, 'probe_seed': sample['probe_seed'],
                            'accuracy_percent': score, 'result_file': str(destination),
                            'encoder_sha256': marker['encoder_sha256']})
        for checkpoint, _, _, checksum in encoders.values():
            if file_hash(checkpoint) != checksum:
                raise ValueError('Frozen encoder file changed during probing')
        scores = [s['accuracy_percent'] for s in samples]
        mean, std = statistics.mean(scores), statistics.stdev(scores)
        summary = {**cell, 'cell_key': job['cell_key'], 'complete': True, 'samples': samples,
            'ssl_epochs': group['epochs'], 'ssl_seeds': sorted(encoders), 'linear_probe_runs': len(samples),
            'scheme': group['scheme'], 'repeated_encoders': job['repeated_encoders'],
            'mean_percent': mean, 'std_percent': std, 'std_ddof': 1,
            'tex_value': '\\dsacc{%.2f}{%.2f}' % (mean, std), 'ssl_training_performed': False}
        atomic_json(summary_path, summary)
        with (BENCH.parent / 'CVPR_tables/.probe_completion.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            subprocess.run(['node', str(HERE / 'tables.js'), str(summary_path)], check=True)
    print('COMPLETE:', job['name'], summary['tex_value'], 'SSL seeds:', summary['ssl_seeds'], flush=True)


if __name__ == '__main__':
    main()
