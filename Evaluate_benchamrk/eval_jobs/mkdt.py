"""Validated evaluation plan and worker for current six-seed MKDT encoders."""
import csv
import itertools
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent
sys.path.insert(0, str(BENCH))

from common import atomic_json, exclusive_run, file_hash
from evaluate import linear_probe
from run import evaluation_targets, validate_protocol
from worker import validate_checkpoint

SOURCES = ('CIFAR10', 'CIFAR100', 'TinyImageNet')
MODELS = ('ConvNet', 'VGG11', 'ResNet18')
SUBSETS = ('1', '2')
SSL_METHODS = ('simclr', 'barlowtwins')
SEEDS = tuple(range(6))


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    with Path(path).open() as stream:
        return list(csv.DictReader(stream, delimiter='\t'))


def expected_protocol(source, subset):
    protocol = read(BENCH / 'protocol.json')
    budgets = {1.0: 5000, 2.0: 4000} if source == 'TinyImageNet' else {
        1.0: 8000, 2.0: 6000}
    protocol['ssl']['subset_epochs'] = budgets[float(subset)]
    protocol['ssl']['checkpoint_every_epochs'] = 500
    protocol['subset_seeds'] = list(SEEDS)
    validate_protocol(protocol)
    return protocol


def build_plan(checkpoint_root, data_root):
    cells = [row for row in rows(BENCH / 'table_coverage.tsv')
             if row['method'] == 'mkdt' and row['disabled'] == 'False'
             and row['subset'] in SUBSETS]
    groups = {}
    for cell in cells:
        key = (f"{cell['source']}/{cell['model']}/mkdt/"
               f"pt{cell['subset']}/{cell['ssl']}")
        group = groups.setdefault(key, {
            'group': key, 'source': cell['source'], 'model': cell['model'],
            'subset': float(cell['subset']), 'ssl': cell['ssl'], 'cells': []})
        group['cells'].append(cell)

    expected_groups = {
        f'{source}/{model}/mkdt/pt{subset}/{ssl}'
        for source, model, subset, ssl in itertools.product(
            SOURCES, MODELS, SUBSETS, SSL_METHODS)}
    if set(groups) != expected_groups or len(cells) != 216:
        raise ValueError('MKDT table coverage must contain 36 configurations/216 cells at 1% and 2%')

    for group in groups.values():
        actual = [(cell['target'], int(cell['labels'])) for cell in group['cells']]
        wanted = set(itertools.product(
            evaluation_targets(group['source'], group['model']), (1, 5)))
        if len(actual) != len(set(actual)) or set(actual) != wanted:
            raise ValueError(f'Missing or duplicate MKDT table cell: {group["group"]}')
        group['cells'].sort(key=lambda cell: (
            cell['target'] != group['source'], cell['target'], int(cell['labels'])))

    jobs = {source: {'task': source, 'units': [], 'probes': 0} for source in SOURCES}
    seen = set()
    for key in sorted(groups):
        group = groups[key]
        for seed in SEEDS:
            relative = f'{key}/seed_{seed:03d}'
            if relative in seen:
                raise ValueError('Duplicate MKDT encoder in evaluation plan')
            seen.add(relative)
            unit = {**group, 'seed': seed, 'relative': relative}
            jobs[group['source']]['units'].append(unit)
            jobs[group['source']]['probes'] += len(unit['cells'])
    if len(seen) != 216 or any(len(job['units']) != 72 for job in jobs.values()):
        raise ValueError('MKDT evaluation plan must contain 216 encoders, 72 per source')

    return {
        'method': 'mkdt', 'seeds': list(SEEDS),
        'checkpoint_root': str(Path(checkpoint_root).resolve()),
        'data_root': str(Path(data_root).resolve()),
        'jobs': [jobs[source] for source in SOURCES],
        'groups': [groups[key] for key in sorted(groups)]}


def evaluate_seed(unit, plan, root, datasets, hashes, driver_hashes, worker_hash):
    source = Path(plan['checkpoint_root']) / unit['relative']
    output = root / unit['relative']
    protocol = expected_protocol(unit['source'], unit['subset'])
    with exclusive_run(output), exclusive_run(source):
        config, saved, checksum = validate_checkpoint(
            source, unit, 'mkdt', protocol, hashes, worker_hash)
        if datasets(unit['source'])[0].fingerprint != config['signature']['source_fingerprint']:
            raise ValueError('Source training split differs from the MKDT training split')
        if config['signature'].get('input_scope') != 'saved_synthetic_images_only':
            raise ValueError('Checkpoint is not from the saved MKDT synthetic images')
        identity = config['identity']
        if (output / 'config.json').exists() and read(output / 'config.json') != config:
            raise ValueError('Refusing to mix checkpoint configurations in one output directory')
        atomic_json(output / 'config.json', config)
        requested = [[cell['target'], int(cell['labels'])] for cell in unit['cells']]
        marker = {
            'complete': True, 'identity': identity,
            'encoder_sha256': saved['encoder_sha256'],
            'checkpoint_file_sha256': checksum, 'checkpoint_retained': True,
            'checkpoint': str(source / 'encoder.pt'), 'evaluations': requested,
            'evaluation_code_hashes': hashes, 'driver_hashes': driver_hashes,
            'ssl_epochs': saved['epochs']}
        prior = read(output / 'complete.json') if (output / 'complete.json').exists() else None
        if prior and any(prior.get(key) != value for key, value in marker.items()):
            raise ValueError('Completed evaluation provenance differs from this run')

        result_hashes = {}
        for target, label in requested:
            path = output / 'evaluations' / f'{target}_lbl{label}.json'
            if prior and (not path.exists()
                          or file_hash(path) != prior['result_hashes'][path.name]):
                raise ValueError(f'Completed result was changed or removed: {path}')
            train_set, test_set = datasets(target)
            result = linear_probe(
                source / 'encoder.pt', identity, protocol, unit['source'], target,
                train_set, test_set, label, unit['seed'], path, 0, 'cuda')
            if (not result.get('complete') or not result['encoder_unchanged']
                    or result['encoder_sha256'] != saved['encoder_sha256']):
                raise ValueError(f'Probe changed or used a different encoder: {path}')
            result_hashes[path.name] = file_hash(path)
        if file_hash(source / 'encoder.pt') != checksum:
            raise ValueError('MKDT SSL checkpoint changed during evaluation')
        atomic_json(output / 'complete.json', {**marker, 'result_hashes': result_hashes})
    print('ENCODER EVALUATED', unit['relative'], flush=True)
