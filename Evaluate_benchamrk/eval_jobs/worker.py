"""Evaluate current retained SSL encoders; never train SSL or remove checkpoints.

Kept below eval_jobs/ so adding this driver does not change the top-level Python
code inventory embedded in the identities of the already-running SSL jobs.
"""
import csv
import fcntl
import functools
import gc
import hashlib
import itertools
import json
import os
from pathlib import Path
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent
sys.path.insert(0, str(BENCH))

import torch
import torchvision
from common import (atomic_json, code_hashes, digest_json, exclusive_run,
                    file_hash, load_encoder)
from data import RawDataset, SOURCE_COUNTS
from evaluate import linear_probe
from run import evaluation_targets, validate_protocol
from report import report


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    with Path(path).open() as stream:
        return list(csv.DictReader(stream, delimiter='\t'))


def expected_protocol(method, source, subset):
    protocol = read(BENCH / 'protocol.json')
    budgets = {0.2: 6000, 1.0: 5000, 2.0: 4000} if source == 'TinyImageNet' else {
        0.2: 10000, 1.0: 8000, 2.0: 6000}
    protocol['ssl']['subset_epochs'] = budgets[float(subset)]
    if method == 'kmeans':
        protocol['ssl']['checkpoint_every_epochs'] = 500
        protocol['subset_seeds'] = list(range(6))
    validate_protocol(protocol)
    return protocol


def build_plan(method, checkpoint_root, data_root):
    if method not in ('kmeans', 'random'):
        raise ValueError('Only current k-means and random checkpoints are in scope')
    seeds = list(range(6 if method == 'kmeans' else 10))
    cells = [r for r in rows(BENCH / 'table_coverage.tsv')
             if r['method'] == method and r['disabled'] == 'False']
    groups = {}
    for cell in cells:
        key = f"{cell['source']}/{cell['model']}/{method}/pt{cell['subset']}/{cell['ssl']}"
        group = groups.setdefault(key, {'group': key, 'source': cell['source'],
            'model': cell['model'], 'subset': float(cell['subset']), 'ssl': cell['ssl'], 'cells': []})
        group['cells'].append(cell)
    expected_groups = {f'{s}/{m}/{method}/pt{p}/{a}' for s, m, p, a in itertools.product(
        ('CIFAR10', 'CIFAR100', 'TinyImageNet'), ('ConvNet', 'VGG11', 'ResNet18'),
        ('0.2', '1', '2'), ('simclr', 'barlowtwins'))}
    if set(groups) != expected_groups or len(cells) != 324:
        raise ValueError('Table coverage no longer matches the planned 54 configurations/324 cells')
    for group in groups.values():
        actual = [(c['target'], int(c['labels'])) for c in group['cells']]
        wanted = set(itertools.product(evaluation_targets(group['source'], group['model']), (1, 5)))
        if len(actual) != len(set(actual)) or set(actual) != wanted:
            raise ValueError('Missing or duplicate main/downstream table cell')
        group['cells'].sort(key=lambda c: (c['target'], int(c['labels'])))
    jobs = {f'{i:02d}': {'task': f'{i:02d}', 'units': [], 'probes': 0} for i in range(1, 121)}
    seen = set()
    for row in rows(HERE / f'{method}_tasks.tsv'):
        key = f"{row['source']}/{row['model']}/{method}/pt{row['subset']}/{row['ssl']}"
        seed = int(row['seed'])
        relative = f'{key}/seed_{seed:03d}'
        if seed not in seeds or relative in seen:
            raise ValueError('Duplicate or unexpected seed in task manifest')
        seen.add(relative)
        unit = {**groups[key], 'seed': seed, 'relative': relative}
        job = jobs[row['task']]
        job['units'].append(unit)
        job['probes'] += len(unit['cells'])
    expected = {f'{g}/seed_{s:03d}' for g in groups for s in seeds}
    if seen != expected or any(not j['units'] for j in jobs.values()):
        raise ValueError('Task manifest does not cover every encoder exactly once')
    return {'method': method, 'seeds': seeds, 'checkpoint_root': str(Path(checkpoint_root).resolve()),
            'data_root': str(Path(data_root).resolve()), 'jobs': list(jobs.values()),
            'groups': list(groups.values())}


def validate_checkpoint(folder, unit, method, protocol, hashes, worker_hash):
    config = read(folder / 'config.json')
    signature, identity = config['signature'], config['identity']
    if (digest_json(signature) != identity['signature']
            or signature['protocol'] != protocol or signature['code_hashes'] != hashes
            or signature['ssl_worker_sha256'] != worker_hash
            or signature['selection_method'] != method or identity['selection_method'] != method
            or identity['seed'] != unit['seed']
            or any(signature[k] != unit[k] for k in ('source', 'model', 'subset', 'ssl'))):
        raise ValueError(f'Wrong checkpoint recipe, code, selection, or seed: {folder}')
    checkpoint = folder / 'encoder.pt'
    marker = read(folder / 'ssl_complete.json')
    checksum = file_hash(checkpoint)
    if (not marker.get('complete') or marker['identity'] != identity
            or marker['epochs'] != protocol['ssl']['subset_epochs']
            or marker['checkpoint_file_sha256'] != checksum):
        raise ValueError(f'Incomplete or changed SSL checkpoint: {folder}')
    encoder, _, saved = load_encoder(checkpoint, identity, 'cpu')
    del encoder
    if (saved['epochs'] != protocol['ssl']['subset_epochs']
            or saved['encoder_sha256'] != marker['encoder_sha256']
            or saved['source'] != unit['source'] or saved['model'] != unit['model']
            or saved['method'] != unit['ssl'] or saved['seed'] != unit['seed']
            or saved['input_size'] != protocol['input_size'][unit['source']]
            or saved['normalization'] != protocol['normalization'][unit['source']]):
        raise ValueError(f'Encoder metadata mismatch: {folder}')
    return config, saved, checksum


def evaluate_seed(unit, plan, root, datasets, hashes, driver_hashes, worker_hash):
    source = Path(plan['checkpoint_root']) / unit['relative']
    output = root / unit['relative']
    protocol = expected_protocol(plan['method'], unit['source'], unit['subset'])
    with exclusive_run(output), exclusive_run(source):
        config, saved, checksum = validate_checkpoint(source, unit, plan['method'], protocol, hashes, worker_hash)
        if datasets(unit['source'])[0].fingerprint != config['signature']['source_fingerprint']:
            raise ValueError('Source training split differs from the SSL training split')
        identity = config['identity']
        if (output / 'config.json').exists() and read(output / 'config.json') != config:
            raise ValueError('Refusing to mix checkpoint configurations in one output directory')
        atomic_json(output / 'config.json', config)
        requested = [[c['target'], int(c['labels'])] for c in unit['cells']]
        marker = {'complete': True, 'identity': identity, 'encoder_sha256': saved['encoder_sha256'],
            'checkpoint_file_sha256': checksum, 'checkpoint_retained': True,
            'checkpoint': str(source / 'encoder.pt'), 'evaluations': requested,
            'evaluation_code_hashes': hashes, 'driver_hashes': driver_hashes,
            'ssl_epochs': saved['epochs']}
        prior = read(output / 'complete.json') if (output / 'complete.json').exists() else None
        if prior and any(prior.get(k) != v for k, v in marker.items()):
            raise ValueError('Completed evaluation provenance differs from this run')
        result_hashes = {}
        for target, label in requested:
            path = output / 'evaluations' / f'{target}_lbl{label}.json'
            if prior and (not path.exists() or file_hash(path) != prior['result_hashes'][path.name]):
                raise ValueError(f'Completed result was changed or removed: {path}')
            train_set, test_set = datasets(target)
            result = linear_probe(source / 'encoder.pt', identity, protocol, unit['source'], target,
                                  train_set, test_set, label, unit['seed'], path, 0, 'cuda')
            if (not result.get('complete') or not result['encoder_unchanged']
                    or result['encoder_sha256'] != saved['encoder_sha256']):
                raise ValueError(f'Probe changed or used a different encoder: {path}')
            result_hashes[path.name] = file_hash(path)
        if file_hash(source / 'encoder.pt') != checksum:
            raise ValueError('SSL checkpoint changed during evaluation')
        atomic_json(output / 'complete.json', {**marker, 'result_hashes': result_hashes})
    print('ENCODER EVALUATED', unit['relative'], flush=True)


def main():
    if 'SLURM_JOB_ID' not in os.environ:
        raise RuntimeError('Run through the supplied Slurm launcher, never on a login node')
    method, task, data_root, checkpoint_root, output_root = sys.argv[1:]
    cluster = 'vulcan' if method == 'kmeans' else 'killarney'
    if os.environ.get('SLURM_CLUSTER_NAME', os.environ.get('CC_CLUSTER')) != cluster:
        raise RuntimeError(f'{method} evaluation belongs on {cluster}')
    root = Path(output_root).resolve()
    if not root.is_relative_to(BENCH) or root == BENCH:
        raise ValueError('Keep results in a dedicated directory inside Evaluate_benchamrk')
    plan = build_plan(method, checkpoint_root, data_root)
    if root == Path(plan['checkpoint_root']) or root.is_relative_to(Path(plan['checkpoint_root'])):
        raise ValueError('Do not place evaluation results inside the checkpoint root')
    job = next(j for j in plan['jobs'] if j['task'] == task)
    hashes = code_hashes()
    driver_hashes = {str(p.relative_to(BENCH)): file_hash(p) for p in (
        HERE / 'worker.py', HERE / 'report.py', HERE / f'{method}_tasks.tsv',
        BENCH / 'table_coverage.tsv', BENCH / 'protocol.json')}
    worker_text = (BENCH / f'{method}_ssl.sh').read_text().split("<<'PY'\n", 1)[1].split('\nPY\n', 1)[0]
    worker_hash = hashlib.sha256(worker_text.encode()).hexdigest()
    contract = {'plan': plan, 'code_hashes': hashes, 'driver_hashes': driver_hashes,
                'ssl_worker_sha256': worker_hash}
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.plan.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = root / 'plan.json'
        if path.exists() and read(path) != contract:
            raise ValueError('Evaluation plan/code changed; use a fresh results directory')
        atomic_json(path, contract)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no CPU fallback')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    @functools.lru_cache(maxsize=7)
    def datasets(target):
        return RawDataset(data_root, target, True), RawDataset(data_root, target, False)

    with exclusive_run(root / 'jobs' / task):
        started = time.monotonic()
        atomic_json(root / 'jobs' / f'{task}_environment.json', {
            'job_id': os.environ['SLURM_JOB_ID'], 'cluster': cluster,
            'account': os.environ.get('SLURM_JOB_ACCOUNT'), 'python': sys.version,
            'torch': str(torch.__version__), 'torchvision': str(torchvision.__version__),
            'gpu': torch.cuda.get_device_name(), 'workers': 0, 'threads': 1,
            'evaluation_code_hashes': hashes, 'driver_hashes': driver_hashes})
        # Missing/partial training is an error, never a silently omitted table seed.
        for unit in job['units']:
            for name in ('config.json', 'encoder.pt', 'ssl_complete.json'):
                path = Path(checkpoint_root) / unit['relative'] / name
                if not path.is_file() or not path.stat().st_size:
                    raise FileNotFoundError(f'Finish SSL first: {path}')
        for target in sorted({c['target'] for u in job['units'] for c in u['cells']}):
            for dataset in datasets(target):
                dataset[0]
            if target in SOURCE_COUNTS and len(datasets(target)[0]) != SOURCE_COUNTS[target]:
                raise ValueError(f'Unexpected source training split size: {target}')
        failures = []
        for unit in job['units']:
            try:
                evaluate_seed(unit, plan, root, datasets, hashes, driver_hashes, worker_hash)
            except Exception as error:
                failures.append({'encoder': unit['relative'], 'error': str(error)})
                traceback.print_exc()
            gc.collect()
            torch.cuda.empty_cache()
        atomic_json(root / 'jobs' / f'{task}_attempt.json', {'complete': not failures,
            'failures': failures, 'seconds': time.monotonic() - started})
        status = report(root, plan, hashes, driver_hashes)
        print('COVERAGE', status, flush=True)
        if failures or status['invalid_encoders']:
            raise RuntimeError('Evaluation failed validation; see job attempt/progress.json')
        atomic_json(root / 'jobs' / f'{task}.json', {'complete': True, 'task': task,
            'job_id': os.environ['SLURM_JOB_ID'], 'encoders': len(job['units']), 'probes': job['probes'],
            'contract_sha256': digest_json(contract), 'seconds': time.monotonic() - started})


if __name__ == '__main__':
    main()
