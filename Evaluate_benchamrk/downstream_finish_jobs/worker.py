"""Finish only the approved empty downstream cells with the existing benchmark trainer/probe."""
import argparse
import contextlib
import fcntl
import functools
import importlib.util
import itertools
import math
import os
from pathlib import Path
import statistics
import sys

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent
sys.path[:0] = [str(BENCH / 'eval_jobs'), str(BENCH)]
spec = importlib.util.spec_from_file_location('downstream_finish_base', BENCH / 'cifar_finish_jobs/worker.py')
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
import interactive_ours_downstream as ours_downstream


def validate_unit(unit):
    kind, source, subset, method, seed = (unit[k] for k in ('kind', 'source', 'subset', 'ssl', 'seed'))
    epochs = 10000 if method == 'barlowtwins' else 1000 if kind == 'ours' and subset == 2 else 2000
    if (kind not in ('ours', 'mkdt') or source not in ('CIFAR10', 'CIFAR100', 'TinyImageNet')
            or subset not in (0.2, 1, 2) or method not in ('simclr', 'barlowtwins')
            or (subset == 2 and method == 'barlowtwins') or seed not in (0, 1)
            or unit['model'] != 'ResNet18' or unit['epochs'] != epochs):
        raise ValueError('Unit is outside the requested downstream tables/protocol')
    for name in ('checkpoint_dir', 'output_dir'):
        base.inside_bench(unit[name])


def check_config(config, unit):
    s, identity = config['signature'], config['identity']
    protocol = base.ours.expected_protocol('random', unit['source'], unit['subset'])
    protocol['ssl'].update(subset_epochs=unit['epochs'], checkpoint_every_epochs=500)
    protocol['subset_seeds'] = [0, 1]
    expected = {k: unit[k] for k in ('source', 'model', 'subset', 'ssl')}
    expected.update(selection_method=unit['kind'], protocol=protocol,
                    code_hashes=base.code_hashes(), ssl_seeds=[0, 1], probe_seeds=[0, 1, 2])
    if (any(s.get(k) != v for k, v in expected.items())
            or identity != {'signature': base.digest_json(s), 'seed': unit['seed'], 'selection_method': unit['kind']}):
        raise ValueError('Saved checkpoint identity or reference protocol differs')


def train(unit, args, datasets):
    validate_unit(unit)
    folder = base.inside_bench(unit['checkpoint_dir'])
    with base.group_reader(folder.parent), base.exclusive_run(folder):
        task = {**unit, 'parent_dir': None}
        if unit['kind'] == 'mkdt' and unit['subset'] == 0.2:
            protocol = base.ours.expected_protocol('random', unit['source'], unit['subset'])
            protocol['ssl'].update(subset_epochs=unit['epochs'], checkpoint_every_epochs=500)
            protocol['subset_seeds'] = [0, 1]
            entry = base.mkdt.synthetic_entry(args.synthetic_root, unit['source'], 'ResNet18', 0.2)
            synthetic = base.SyntheticDataset(entry, unit['source'], 0.2, protocol['input_size'][unit['source']])
            signature = {'protocol': protocol, 'source': unit['source'], 'model': 'ResNet18',
                'subset': 0.2, 'ssl': unit['ssl'], 'selection_method': 'mkdt',
                'source_fingerprint': datasets(unit['source'])[0].fingerprint,
                'synthetic': synthetic.metadata, 'ssl_seeds': [0, 1], 'probe_seeds': [0, 1, 2],
                'code_hashes': base.code_hashes(), 'driver_sha256': base.file_hash(base.mkdt.__file__),
                'input_scope': 'saved_synthetic_images_only'}
            config = {'signature': signature, 'identity': {'signature': base.digest_json(signature),
                'seed': unit['seed'], 'selection_method': 'mkdt'}}
        else:
            synthetic, config = base.synthetic_config(task, args, datasets)
        check_config(config, unit)
        path = folder / 'config.json'
        if path.exists() and base.read(path) != config:
            raise ValueError(f'Refusing to replace a different saved SSL configuration: {path}')
        base.atomic_json(path, config)
        checkpoint, saved = base.train(synthetic, config['signature']['protocol'], unit['source'],
            'ResNet18', unit['ssl'], unit['seed'], folder, config['identity'], 0, 'cuda')
        if not saved['complete'] or saved['epochs'] != unit['epochs']:
            raise ValueError('SSL did not finish the selected epoch budget')
        marker = {'complete': True, 'identity': config['identity'], 'epochs': saved['epochs'],
            'encoder_sha256': saved['encoder_sha256'], 'checkpoint_file_sha256': base.file_hash(checkpoint)}
        path = folder / 'ssl_complete.json'
        if path.exists() and base.read(path) != marker:
            raise ValueError('Final SSL marker changed')
        base.atomic_json(path, marker)
    print('SSL COMPLETE:', unit['id'], flush=True)


def summary(unit, cell):
    rows = []
    signatures, fingerprints = set(), set()
    for seed, probe in itertools.product((0, 1), (0, 1, 2)):
        folder = base.inside_bench(unit['output_dir']).parent / f'ssl_seed_{seed:03d}'
        path = folder / 'evaluations' / f'{cell["target"]}_lbl{cell["labels"]}_probe{probe}.json'
        if not path.exists():
            continue
        result, config, marker = base.read(path), base.read(folder / 'config.json'), base.read(folder / 'ssl_complete.json')
        check_config(config, {**unit, 'seed': seed})
        e, score = result['expected'], result['accuracy_percent']
        if (not result['complete'] or not result['encoder_unchanged']
                or e['identity'] != config['identity'] or e['target'] != cell['target']
                or e['source'] != unit['source'] or e['seed'] != probe
                or e['label_percentage'] != cell['labels']
                or e['evaluation'] != config['signature']['protocol']['evaluation']
                or e['evaluation_code_hashes'] != base.code_hashes()
                or result['encoder_sha256'] != marker['encoder_sha256'] or not marker['complete']
                or marker['identity'] != config['identity']
                or not math.isfinite(score) or not 0 <= score <= 100):
            raise ValueError(f'Invalid saved downstream result: {path}')
        signatures.add(config['identity']['signature'])
        fingerprints.add((e['train_fingerprint'], e['test_fingerprint']))
        if len(signatures) != 1 or len(fingerprints) != 1:
            raise ValueError('Cannot combine different synthetic inputs or downstream datasets')
        rows.append({'ssl_seed': seed, 'probe_seed': probe, 'accuracy_percent': score})
    scores = [r['accuracy_percent'] for r in rows]
    complete = len(scores) == 6
    directory = HERE / 'summaries' / unit['source'] / unit['kind'] / f'pt{unit["subset"]}' / unit['ssl']
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f'{cell["target"]}_lbl{cell["labels"]}.json'
    row = {**cell, 'method': unit['kind'], 'model': 'ResNet18BN' if unit['kind'] == 'ours' else 'ResNet18',
        'completed_probes': len(scores), 'expected_probes': 6,
        'complete': complete, 'mean_percent': statistics.mean(scores) if complete else None,
        'std_percent': statistics.stdev(scores) if complete else None, 'std_ddof': 1, 'samples': rows}
    base.atomic_json(path, row)
    group = base.inside_bench(unit['output_dir']).parent
    with (group / '.summary.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        destination = group / 'summary.json'
        old = base.read(destination) if destination.exists() else {}
        if old.get('signature', config['identity']['signature']) != config['identity']['signature']:
            raise ValueError('Existing group summary belongs to a different SSL configuration')
        merged = {(r['target'], r['labels']): r for r in old.get('cells', [])}
        for file in directory.glob('*.json'):
            value = base.read(file)
            merged[value['target'], value['labels']] = value
        cells = sorted(merged.values(), key=lambda r: (r['target'], r['labels']))
        base.atomic_json(destination, {'signature': config['identity']['signature'],
            'ssl_seeds': [0, 1], 'probe_seeds': [0, 1, 2], 'std_ddof': 1,
            'complete': all(r['complete'] for r in cells), 'cells': cells})


def probe(bundle, args, datasets):
    unit, cell = bundle['unit'], bundle['cell']
    validate_unit(unit)
    targets = [name for name in base.ours.evaluation_targets(unit['source'], 'ResNet18') if name != unit['source']]
    identifier = '|'.join(str(cell[k]) for k in ('source', 'kind', 'subset', 'ssl', 'target', 'labels'))
    if (identifier not in set(base.read(HERE / 'approved_cells.json'))
            or any(cell[k] != unit[k] for k in ('source', 'kind', 'subset', 'ssl'))
            or cell['target'] not in targets or cell['labels'] not in (1, 5)
            or not bundle['probes'] or not set(bundle['probes']).issubset({0, 1, 2})):
        raise ValueError('Probe was not approved as an empty, nonexcluded downstream cell')
    folder, output = (base.inside_bench(unit[k]) for k in ('checkpoint_dir', 'output_dir'))
    with contextlib.ExitStack() as locks:
        for directory in sorted({folder.parent, folder, output.parent, output}):
            locks.enter_context(base.group_reader(directory))
        config, marker = base.read(folder / 'config.json'), base.read(folder / 'ssl_complete.json')
        check_config(config, unit)
        checkpoint = folder / 'encoder.pt'
        checksum = base.file_hash(checkpoint)
        if (not marker['complete'] or marker['epochs'] != unit['epochs']
                or marker['identity'] != config['identity'] or checksum != marker['checkpoint_file_sha256']):
            raise ValueError('Missing, altered or unfinished SSL checkpoint')
        encoder, _, saved = base.load_encoder(checkpoint, config['identity'], 'cpu')
        del encoder
        expected = {'source': unit['source'], 'model': 'ResNet18', 'method': unit['ssl'],
            'seed': unit['seed'], 'epochs': unit['epochs'], 'encoder_sha256': marker['encoder_sha256'],
            'input_size': config['signature']['protocol']['input_size'][unit['source']],
            'normalization': config['signature']['protocol']['normalization'][unit['source']]}
        if any(saved.get(k) != v for k, v in expected.items()):
            raise ValueError('Encoder metadata differs from the requested model/SSL seed')
        if datasets(unit['source'])[0].fingerprint != config['signature']['source_fingerprint']:
            raise ValueError('Source train split changed')
        values = {'config.json': config}
        if unit['kind'] == 'ours':
            values.update({'ssl_complete.json': {**marker, 'checkpoint': str(checkpoint)},
                'checkpoint_reference.json': {'checkpoint': str(checkpoint), 'checkpoint_file_sha256': checksum,
                    'driver_sha256': base.file_hash(ours_downstream.__file__), 'targets': targets, 'scope': 'downstream'}})
        with base.exclusive_run(output / '.metadata'):
            for name, value in values.items():
                path = output / name
                if path.exists() and base.read(path) != value:
                    raise ValueError(f'Refusing to mix downstream provenance: {path}')
                base.atomic_json(path, value)
        for seed in bundle['probes']:
            name = f'{cell["target"]}_lbl{cell["labels"]}_probe{seed}'
            with base.exclusive_run(output / '.probe_locks' / name):
                train_set, test_set = datasets(cell['target'])
                result = base.linear_probe(checkpoint, config['identity'], config['signature']['protocol'],
                    unit['source'], cell['target'], train_set, test_set, cell['labels'], seed,
                    output / 'evaluations' / f'{name}.json', 0, 'cuda')
                if (not result['complete'] or not result['encoder_unchanged']
                        or result['encoder_sha256'] != marker['encoder_sha256']):
                    raise ValueError('Probe used or altered the wrong encoder')
            directory = HERE / 'summary_locks'
            directory.mkdir(exist_ok=True)
            with (directory / (identifier.replace('|', '_') + '.lock')).open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                summary(unit, cell)
        if base.file_hash(checkpoint) != checksum:
            raise ValueError('SSL encoder changed during downstream probing')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('data_root', type=Path)
    parser.add_argument('synthetic_root', type=Path)
    args = parser.parse_args()
    if (not os.environ.get('SLURM_JOB_ID') or not base.torch.cuda.is_available()
            or os.environ.get('SLURM_CLUSTER_NAME', os.environ.get('CC_CLUSTER')) != 'killarney'
            or os.environ.get('SLURM_JOB_ACCOUNT') != 'aip-boyuwang'):
        parser.error('Use a Wang-account GPU job on Killarney')
    if not args.manifest.resolve().is_relative_to(HERE / 'batches'):
        parser.error('Unexpected job manifest path')
    base.torch.set_num_threads(1)
    base.torch.set_num_interop_threads(1)
    base.torch.backends.cuda.matmul.allow_tf32 = False
    base.torch.backends.cudnn.allow_tf32 = False
    args.data_root, args.synthetic_root = args.data_root.resolve(), args.synthetic_root.resolve()

    @functools.lru_cache(maxsize=1)
    def datasets(name):
        return base.RawDataset(args.data_root, name, True), base.RawDataset(args.data_root, name, False)

    job = base.read(args.manifest)
    if job['stage'] == 'ssl':
        train(job['unit'], args, datasets)
    elif job['stage'] == 'probe' and 1 <= len(job['bundles']) <= 5:
        for bundle in job['bundles']:
            probe(bundle, args, datasets)
    else:
        raise ValueError('Invalid downstream stage/bundle count')
    print('COMPLETE:', job['name'], flush=True)


if __name__ == '__main__':
    main()
