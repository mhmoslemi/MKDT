"""ResNet18-generated Ours -> BN ResNet34/50; benchmark SSL and frozen probes."""
import fcntl
import functools
import itertools
import json
import math
import os
from pathlib import Path
import statistics
import sys

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent
ROOT = BENCH.parent
sys.path[:0] = [str(BENCH / 'eval_jobs'), str(BENCH)]
import torch
import common
import data
import train_ssl
from evaluate import linear_probe
from interactive_ours import load_synthetic
from models import install


def read(path):
    return json.loads(Path(path).read_text())


def folder_for(task, seed=None):
    return HERE / 'results' / task['source'] / task['model'] / task['kind'] / ('pt%g' % task['size']) / 'simclr' / (
        'ssl_seed_%03d' % (task['seed'] if seed is None else seed))


def summarize(task):
    group = folder_for(task).parent
    targets = (task['source'], 'CIFAR100' if task['source'] == 'CIFAR10' else 'CIFAR10', 'TinyImageNet')
    seeds = (0,) if task['kind'] == 'full' else (0, 1)
    probes = range(5 if task['kind'] == 'full' else 2)
    rows = []
    for target, labels in itertools.product(targets, (1, 5)):
        samples, signature, fingerprints = [], None, None
        for seed, probe in itertools.product(seeds, probes):
            folder = folder_for(task, seed)
            path = folder / 'evaluations' / ('%s_lbl%d_probe%d.json' % (target, labels, probe))
            if not path.is_file():
                continue
            result, config, marker = read(path), read(folder / 'config.json'), read(folder / 'ssl_complete.json')
            expected = result['expected']
            score = result['accuracy_percent']
            if (not result.get('complete') or not result.get('encoder_unchanged') or not marker['complete']
                    or expected['identity'] != config['identity'] or marker['identity'] != config['identity']
                    or result['encoder_sha256'] != marker['encoder_sha256']
                    or [expected['source'], expected['target'], expected['label_percentage'], expected['seed']] != [task['source'], target, labels, probe]
                    or expected['evaluation'] != config['signature']['protocol']['evaluation']
                    or expected['evaluation_code_hashes'] != common.code_hashes()
                    or config['signature']['ablation_code_hashes'] != task['code_hashes']
                    or not math.isfinite(score) or not 0 <= score <= 100):
                raise ValueError('Invalid cross-architecture probe: %s' % path)
            current = (expected['train_fingerprint'], expected['test_fingerprint'])
            if signature is not None and (signature != config['signature'] or fingerprints != current):
                raise ValueError('Different SSL recipe/data across seeds')
            signature, fingerprints = config['signature'], current
            samples.append({'ssl_seed': seed, 'probe_seed': probe, 'accuracy_percent': score})
        complete = len(samples) == len(seeds) * len(probes)
        scores = [s['accuracy_percent'] for s in samples]
        rows.append({'target': target, 'labels': labels, 'complete': complete,
                     'completed_probes': len(samples), 'expected_probes': len(seeds) * len(probes),
                     'mean_percent': statistics.mean(scores) if complete else None,
                     'std_percent': statistics.stdev(scores) if complete else None, 'samples': samples})
    common.atomic_json(group / 'summary.json', {'source': task['source'], 'model': task['model'],
        'kind': task['kind'], 'subset': task['size'], 'std_ddof': 1, 'cells': rows,
        'complete': all(r['complete'] for r in rows)})
    table = ROOT / 'CVPR_tables/ablations' / ('cross_arch_%s.tex' % task['source'].lower())
    text = table.read_text()
    for kind, size, labels in itertools.product(('full', 'random', 'ours'), (0.2, 1, 100), (1, 5)):
        if (kind == 'full') != (size == 100):
            continue
        values = []
        for model in ('ResNet34', 'ResNet50'):
            summary = HERE / 'results' / task['source'] / model / kind / ('pt%g' % size) / 'simclr/summary.json'
            cells = {(r['target'], r['labels']): r for r in read(summary)['cells']} if summary.is_file() else {}
            for target in targets:
                cell = cells.get((target, labels), {})
                values.append('\\dsacc{%.2f}{%.2f}' % (cell['mean_percent'], cell['std_percent']) if cell.get('complete') else '')
        tag = 'crossarch kind=%s pt=%g lbl=%d' % (kind, size, labels)
        # Match the first three metadata columns independently of the six results.
        matching = [line for line in text.splitlines() if line.endswith('% ' + tag)]
        assert len(matching) == 1, 'Missing/duplicate table row: ' + tag
        old = matching[0]
        prefix = '&'.join(old.split('&')[:3]).rstrip()
        text = text.replace(old, prefix + ' & ' + ' & '.join(values) + r' \\ % ' + tag)
    temporary = table.with_name('.%s.%d.tmp' % (table.name, os.getpid()))
    temporary.write_text(text)
    temporary.replace(table)


class SegmentComplete(Exception):
    pass


def main():
    task, data_root = read(sys.argv[1]), Path(sys.argv[2]).resolve()
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise RuntimeError('Use the submitted single-GPU job')
    assert task['stage'] in ('ssl', 'probe') and task['model'] in ('ResNet34', 'ResNet50')
    assert task['source'] in ('CIFAR10', 'CIFAR100') and task['kind'] in ('full', 'random', 'ours')
    assert task['size'] in ((100,) if task['kind'] == 'full' else (0.2, 1))
    assert task['seed'] in ((0,) if task['kind'] == 'full' else (0, 1))
    for path, sha in task['code_hashes'].items():
        if common.file_hash(ROOT / path) != sha:
            raise ValueError('Code changed since submission: ' + path)
    calibration = read(HERE / 'calibration.json')
    assert calibration['complete'] and all(calibration['models'][m]['bn_only'] for m in ('ResNet34', 'ResNet50'))
    for path, sha in calibration['code_hashes'].items():
        assert common.file_hash(ROOT / path) == sha, 'Preflight code changed: ' + path
    install()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    protocol = read(BENCH / 'protocol.json')
    protocol['ssl'].update(full_epochs=1000, subset_epochs=2000,
                           checkpoint_every_epochs=50 if task['kind'] == 'full' else 500)
    protocol['full_seeds'], protocol['subset_seeds'] = [0], [0, 1]
    assert protocol['evaluation']['epochs'] == 100

    @functools.lru_cache(maxsize=3)
    def datasets(target):
        train, test = data.RawDataset(data_root, target, True), data.RawDataset(data_root, target, False)
        assert train.fingerprint == calibration['fingerprints'][target + '_train']
        assert test.fingerprint == calibration['fingerprints'][target + '_test']
        return train, test

    base, _ = datasets(task['source'])
    metadata = None
    if task['kind'] == 'ours':
        path = ROOT / task['synthetic_file']
        if common.file_hash(path) != task['synthetic_sha256'] or calibration['synthetics'][task['synthetic_file']] != task['synthetic_sha256']:
            raise ValueError('ResNet18 synthetic data changed')
        loaded = load_synthetic(path, task['source'], 'ResNet18BN', task['size'])
        if loaded is None:
            raise ValueError('Ours synthetic generation is unfinished')
        selected, metadata = loaded
        selection = {'synthetic_images_sha256': metadata['images_sha256']}
    elif task['kind'] == 'random':
        selected, selection = data.select_subset(base, 'random', task['size'], task['seed'], protocol,
                                                HERE / 'selections', workers=0, device='cuda')
    else:
        selected, selection = base, {'scope': 'full'}
    signature = {'ablation': 'resnet18_synthetic_cross_architecture', 'source': task['source'],
        'source_fingerprint': base.fingerprint, 'model': task['model'], 'selection_method': task['kind'],
        'subset': task['size'], 'ssl': 'simclr', 'protocol': protocol, 'synthetic': metadata,
        'code_hashes': common.code_hashes(), 'ablation_code_hashes': task['code_hashes'],
        'probe_seeds': list(range(5 if task['kind'] == 'full' else 2))}
    identity = {'signature': common.digest_json(signature), 'seed': task['seed'],
                'selection': selection, 'selection_method': task['kind']}
    folder = folder_for(task)
    config = {'identity': identity, 'signature': signature}
    with common.exclusive_run(folder):
        config_file = folder / 'config.json'
        if config_file.exists() and read(config_file) != config:
            raise ValueError('Existing configuration differs; refusing to mix runs')
        common.atomic_json(config_file, config)
        checkpoint, marker_file = folder / 'encoder.pt', folder / 'ssl_complete.json'
        epochs = 1000 if task['kind'] == 'full' else 2000
        if task['stage'] == 'ssl':
            original_save = train_ssl.atomic_save

            def save_and_stop(path, value):
                original_save(path, value)
                if Path(path).name == 'training.pt':
                    common.atomic_json(folder / 'progress.json', {'identity': identity, 'epoch': value['epoch']})
                    if task['end_epoch'] <= value['epoch'] < epochs:
                        raise SegmentComplete

            train_ssl.atomic_save = save_and_stop
            try:
                checkpoint, saved = train_ssl.train(selected, protocol, task['source'], task['model'], 'simclr',
                    task['seed'], folder, identity, workers=0, device='cuda', full=task['kind'] == 'full')
            except SegmentComplete:
                print('SEGMENT COMPLETE: saved full optimizer/RNG state; total schedule remains 1000 epochs.', flush=True)
                return
            finally:
                train_ssl.atomic_save = original_save
            assert saved['epochs'] == epochs and saved['model'] == task['model']
            common.atomic_json(marker_file, {'complete': True, 'identity': identity, 'epochs': epochs,
                'encoder_sha256': saved['encoder_sha256'], 'checkpoint_file_sha256': common.file_hash(checkpoint)})
            return
        marker = read(marker_file)
        if (not marker['complete'] or marker['identity'] != identity or marker['epochs'] != epochs
                or common.file_hash(checkpoint) != marker['checkpoint_file_sha256']):
            raise ValueError('Invalid final SSL checkpoint')
        encoder, _, saved = common.load_encoder(checkpoint, identity, device='cpu')
        assert saved['encoder_sha256'] == marker['encoder_sha256'] and saved['epochs'] == epochs
        del encoder, selected
        targets = (task['source'], 'CIFAR100' if task['source'] == 'CIFAR10' else 'CIFAR10', 'TinyImageNet')
        try:
            for target, labels, seed in itertools.product(targets, (1, 5), signature['probe_seeds']):
                train, test = datasets(target)
                path = folder / 'evaluations' / ('%s_lbl%d_probe%d.json' % (target, labels, seed))
                result = linear_probe(checkpoint, identity, protocol, task['source'], target, train, test,
                                      labels, seed, path, workers=0, device='cuda')
                assert result['encoder_unchanged'] and result['encoder_sha256'] == marker['encoder_sha256']
        finally:
            with (HERE / '.report.lock').open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                summarize(task)


if __name__ == '__main__':
    main()
