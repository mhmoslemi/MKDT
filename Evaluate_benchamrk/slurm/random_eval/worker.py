"""Evaluate retained random encoders only. Never call SSL training or cleanup."""
import csv
import functools
import gc
import json
import os
from pathlib import Path
import sys
import traceback

SCRIPT = Path(__file__).resolve().parent
BENCH = SCRIPT.parent.parent
sys.path.insert(0, str(BENCH))

import torch
import torchvision
from common import (atomic_json, code_hashes, digest_json, exclusive_run,
                    file_hash, load_encoder)
from data import RawDataset
from evaluate import linear_probe
from report import report


def read(path):
    return json.loads(Path(path).read_text())


def missing_checkpoint_files(unit, plan):
    folder = Path(plan['checkpoint_root']) / unit['relative']
    return [name for name in ('encoder.pt', 'config.json', 'ssl_complete.json')
            if not (folder / name).is_file() or (folder / name).stat().st_size == 0]


def build_plan(checkpoint_root, data_root):
    """Read the git-tracked task list and the existing table checklist in Slurm."""
    with (BENCH / 'table_coverage.tsv').open() as stream:
        cells = [r for r in csv.DictReader(stream, delimiter='\t')
                 if r['method'] == 'random' and r['disabled'] == 'False']
    groups = {}
    for cell in cells:
        key = f"{cell['source']}/{cell['model']}/random/pt{cell['subset']}/{cell['ssl']}"
        group = groups.setdefault(key, {'group': key, 'source': cell['source'],
            'model': cell['model'], 'subset': float(cell['subset']), 'ssl': cell['ssl'], 'cells': []})
        group['cells'].append(cell)
    jobs = {f'{i:02d}': {'task': f'{i:02d}', 'units': [], 'probes': 0} for i in range(1, 41)}
    seen = set()
    with (SCRIPT / 'tasks.tsv').open() as stream:
        for row in csv.DictReader(stream, delimiter='\t'):
            key = f"{row['source']}/{row['model']}/random/pt{row['subset']}/{row['ssl']}"
            seed = int(row['seed'])
            relative = f'{key}/seed_{seed:03d}'
            if seed not in range(10) or relative in seen:
                raise ValueError('Invalid or duplicate seed in tasks.tsv')
            seen.add(relative)
            unit = {**groups[key], 'seed': seed, 'relative': relative}
            expected_task = ('39' if row['subset'] == '1' else '40') if (
                row['source'] == 'TinyImageNet' and row['model'] == 'ResNet18'
                and row['subset'] in ('1', '2')) else None
            if (expected_task and row['task'] != expected_task) or (
                    not expected_task and row['task'] in ('39', '40')):
                raise ValueError('Task does not match its training dependency')
            job = jobs[row['task']]
            job['units'].append(unit)
            job['probes'] += len(unit['cells'])
    if (len(cells) != 324 or len(groups) != 54 or len(seen) != 540
            or sum(j['probes'] for j in jobs.values()) != 3240
            or any(not j['units'] for j in jobs.values())):
        raise ValueError('Task list does not cover the random benchmark exactly')
    return {'checkpoint_root': checkpoint_root, 'data_root': data_root,
            'jobs': list(jobs.values()), 'groups': list(groups.values())}


def validate_checkpoint(folder, unit, protocol, hashes):
    config = read(folder / 'config.json')
    signature, identity = config['signature'], config['identity']
    if digest_json(signature) != identity['signature']:
        raise ValueError(f'Configuration signature mismatch: {folder}')
    if (signature['protocol'] != protocol or signature['code_hashes'] != hashes
            or signature['selection_method'] != 'random'
            or identity['selection_method'] != 'random' or identity['seed'] != unit['seed']
            or any(signature[key] != unit[key] for key in ('source', 'model', 'subset', 'ssl'))):
        raise ValueError(f'Checkpoint does not match the planned recipe/code/seed: {folder}')
    checkpoint = folder / 'encoder.pt'
    marker = read(folder / 'ssl_complete.json')
    checksum = file_hash(checkpoint)
    if (not marker.get('complete') or marker['identity'] != identity
            or marker['epochs'] != protocol['ssl']['subset_epochs']
            or checksum != marker['checkpoint_file_sha256']):
        raise ValueError(f'Incomplete or changed SSL checkpoint: {checkpoint}')
    encoder, _, metadata = load_encoder(checkpoint, identity, 'cpu')
    del encoder
    if (not metadata['complete'] or metadata['epochs'] != protocol['ssl']['subset_epochs']
            or metadata['encoder_sha256'] != marker['encoder_sha256']
            or metadata['source'] != unit['source'] or metadata['model'] != unit['model']
            or metadata['method'] != unit['ssl'] or metadata['seed'] != unit['seed']
            or metadata['input_size'] != protocol['input_size'][unit['source']]
            or metadata['normalization'] != protocol['normalization'][unit['source']]):
        raise ValueError(f'Encoder metadata mismatch: {checkpoint}')
    return config, metadata, checksum


def evaluate_seed(unit, plan, result_root, datasets, protocol, hashes, device='cuda'):
    folder = Path(plan['checkpoint_root']) / unit['relative']
    output = result_root / unit['relative']
    # Lock the output and source seed so a restarted training job cannot replace it.
    with exclusive_run(output), exclusive_run(folder):
        config, metadata, checksum = validate_checkpoint(folder, unit, protocol, hashes)
        identity = config['identity']
        if datasets(unit['source'])[0].fingerprint != config['signature']['source_fingerprint']:
            raise ValueError('Source train split differs from the SSL training split')
        config_path = output / 'config.json'
        if config_path.exists() and read(config_path) != config:
            raise ValueError('Refusing to mix checkpoint identities in an evaluation directory')
        atomic_json(config_path, config)
        marker_path = output / 'complete.json'
        prior = read(marker_path) if marker_path.exists() else None
        requested = [[c['target'], int(c['labels'])] for c in unit['cells']]
        if prior and (not prior.get('complete') or prior['identity'] != identity
                      or prior['checkpoint_file_sha256'] != checksum
                      or prior['evaluation_code_hashes'] != hashes
                      or prior['evaluations'] != requested):
            raise ValueError('Incompatible completed evaluation marker')
        result_hashes = {}
        for target, label in sorted(requested):
            train_set, test_set = datasets(target)
            path = output / 'evaluations' / f'{target}_lbl{label}.json'
            if prior and (not path.exists() or file_hash(path) != prior['result_hashes'][path.name]):
                raise ValueError(f'Completed evaluation was modified or removed: {path}')
            result = linear_probe(folder / 'encoder.pt', identity, protocol, unit['source'],
                                  target, train_set, test_set, label, unit['seed'], path, 0, device)
            if (not result.get('complete') or not result['encoder_unchanged']
                    or result['encoder_sha256'] != metadata['encoder_sha256']):
                raise ValueError(f'Evaluation used a different or changed encoder: {path}')
            result_hashes[path.name] = file_hash(path)
        # Do not remove the input encoder or optimizer checkpoint, even on success.
        atomic_json(marker_path, {'complete': True, 'identity': identity,
            'encoder_sha256': metadata['encoder_sha256'], 'checkpoint_file_sha256': checksum,
            'checkpoint_retained': True, 'checkpoint': str(folder / 'encoder.pt'),
            'evaluations': requested, 'result_hashes': result_hashes,
            'evaluation_code_hashes': hashes, 'ssl_epochs': metadata['epochs']})
    print('ENCODER EVALUATED', unit['relative'], flush=True)


def main():
    if 'SLURM_JOB_ID' not in os.environ:
        raise RuntimeError('Run this worker through Slurm')
    task, data_root, checkpoint_root, output_root = sys.argv[1:]
    root = Path(output_root).resolve()
    plan = build_plan(checkpoint_root, data_root)
    jobs = [j for j in plan['jobs'] if j['task'] == task]
    if len(jobs) != 1:
        raise ValueError('Unknown or duplicate task')
    job = jobs[0]
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no CPU fallback')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    protocol = read(BENCH / 'protocol.json')
    hashes = code_hashes()

    # Seven bounded raw splits: CIFAR pixels, mmap'd Tiny pixels, and paths for
    # the four file datasets. Reuse fingerprints instead of restatting tens of
    # thousands of image files for every seed. Images from paths stay lazy.
    @functools.lru_cache(maxsize=7)
    def datasets(target):
        return (RawDataset(plan['data_root'], target, True),
                RawDataset(plan['data_root'], target, False))

    with exclusive_run(root / 'jobs' / task):
        atomic_json(root / 'jobs' / f'{task}_environment.json', {
            'job_id': os.environ['SLURM_JOB_ID'], 'cluster': os.environ.get('SLURM_CLUSTER_NAME'),
            'account': os.environ.get('SLURM_JOB_ACCOUNT'), 'python': sys.version, 'torch': str(torch.__version__),
            'torchvision': str(torchvision.__version__), 'gpu': torch.cuda.get_device_name(),
            'workers': 0, 'threads': 1, 'evaluation_code_hashes': hashes})
        ready, missing = [], []
        for unit in job['units']:
            absent = missing_checkpoint_files(unit, plan)
            if absent:
                missing.append({'encoder': unit['relative'], 'missing_files': absent})
                print('SKIP MISSING CHECKPOINT', unit['relative'], ','.join(absent), flush=True)
            else:
                ready.append(unit)
        # Check only datasets needed for the encoders available in this job.
        for target in sorted({c['target'] for u in ready for c in u['cells']}):
            for dataset in datasets(target):
                dataset[0]
        failures = []
        for unit in ready:
            try:
                evaluate_seed(unit, plan, root, datasets, protocol, hashes)
            except Exception as error:
                failures.append({'encoder': unit['relative'], 'error': str(error)})
                traceback.print_exc()
            gc.collect()
            torch.cuda.empty_cache()
        if failures:
            atomic_json(root / 'jobs' / f'{task}_failed.json', {'complete': False, 'failures': failures})
        if missing:
            atomic_json(root / 'jobs' / f'{task}_partial.json', {'complete': False, 'task': task,
                'evaluated_encoders': len(ready) - len(failures), 'missing_checkpoints': missing})
        # Each worker refreshes reports under one lock. The last successful worker
        # produces final summaries without requiring a 41st aggregation job.
        status = report(root, plan)
        print('COVERAGE', status, flush=True)
        if failures:
            raise RuntimeError(f'{len(failures)} encoders failed; see {root}/jobs/{task}_failed.json')
        if status['invalid_encoders']:
            raise RuntimeError('Report found invalid completed results; see progress.json')
        if not missing:
            atomic_json(root / 'jobs' / f'{task}.json', {'complete': True, 'task': task,
                'job_id': os.environ['SLURM_JOB_ID'], 'encoders': len(job['units']), 'probes': job['probes']})
            (root / 'jobs' / f'{task}_partial.json').unlink(missing_ok=True)
        else:
            print(f'PARTIAL task={task}: evaluated {len(ready)}, missing {len(missing)}; rerun after SSL resumes.', flush=True)
        (root / 'jobs' / f'{task}_failed.json').unlink(missing_ok=True)


if __name__ == '__main__':
    main()
