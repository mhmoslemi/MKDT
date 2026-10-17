"""Train one encoder per seed, reuse it for every target, retain full-data only.

Run through scripts/*.sh or directly with --help. No scheduler integration.
"""

import argparse
import gc
import json
import os
from pathlib import Path
import statistics
import sys

import torch
import torchvision

from common import (HERE, ROOT, atomic_json, code_hashes, digest_json, exclusive_run,
                    file_hash)
from data import RawDataset, SyntheticDataset, TARGETS, SOURCE_COUNTS, select_subset
from evaluate import linear_probe
from train_ssl import train


def size_key(value):
    return format(float(value), 'g')


def evaluation_targets(source, model):
    return [source] + ([target for target in TARGETS if target != source] if model == 'ResNet18' else [])


def validate_protocol(protocol):
    # Reject unsupported switches rather than record a value that was ignored.
    ssl, evaluation = protocol['ssl'], protocol['evaluation']
    required = {
        'optimizer': 'lars', 'precision': 'float32', 'exclude_bias_and_norm': True,
        'schedule': 'warmup_cosine_per_step', 'small_batch_policy': 'min_dataset_size_no_replacement',
        'interpolation': 'bicubic'}
    for key, value in required.items():
        if ssl[key] != value:
            raise ValueError(f'Unsupported ssl.{key}: expected {value!r}')
    required = {'optimizer': 'sgd', 'schedule': 'cosine_per_step', 'warmup_epochs': 0,
                'label_policy': 'exact', 'normalize_using': 'source_statistics',
                'train_transform': 'random_resized_crop_and_flip',
                'test_transform': 'resize_short_side_and_center_crop',
                'checkpoint_selection': 'final_epoch_no_test_selection',
                'baseline': 'frozen_random_encoder'}
    for key, value in required.items():
        if evaluation[key] != value:
            raise ValueError(f'Unsupported evaluation.{key}: expected {value!r}')
    for value in (ssl['full_epochs'], ssl['subset_epochs'], ssl['batch_size'],
                  ssl['checkpoint_every_epochs'], evaluation['epochs'], evaluation['batch_size']):
        if not isinstance(value, int) or value < 1:
            raise ValueError('Epoch and batch counts must be positive integers')
    if ssl['batch_size'] < 2:
        raise ValueError('SSL batch size must be at least two')
    if evaluation['label_percentages'] != [1, 5]:
        raise ValueError('The table contract requires label budgets [1, 5]')
    if evaluation['full_data_sanity_label_percentage'] != 100:
        raise ValueError('Full-data sanity evaluation must use all training labels')
    expected_selection = {'method': 'kmeans', 'feature_model': 'random_ConvNetGAP_BN',
                          'feature_normalization': 'l2', 'unique_exemplars': True}
    for key, value in expected_selection.items():
        if protocol['selection'][key] != value:
            raise ValueError(f'Unsupported selection.{key}: expected {value!r}')
    for recipe in protocol['methods'].values():
        if recipe['lr'] <= 0 or len(recipe['projector']) < 2 or min(recipe['projector']) < 2:
            raise ValueError('Invalid learning rate or projector dimensions')


def synthetic_entry(path, method, source, percentage):
    entries = json.loads(Path(path).read_text())
    try:
        entry = entries[method][source][size_key(percentage)]
    except KeyError as error:
        raise ValueError(f'No synthetic input registered for {method}/{source}/{percentage}% in {path}') from error
    if not entry.get('path'):
        raise FileNotFoundError(f'Missing synthetic images: {method}/{source}/{percentage}%. '
                                f'Set its path, key and image space in {path}; no substitute data is allowed.')
    return entry


def expected_evaluations(source, model, full, protocol):
    entries = [(target, label) for target in evaluation_targets(source, model)
               for label in protocol['evaluation']['label_percentages']]
    if full:
        entries.append((source, protocol['evaluation']['full_data_sanity_label_percentage']))
    return entries


def validate_completed(folder, identity, datasets, requested):
    marker = folder / 'complete.json'
    if not marker.exists():
        return False
    saved = json.loads(marker.read_text())
    if saved.get('identity') != identity or not saved.get('complete'):
        raise ValueError(f'Run identity mismatch in {marker}; use a fresh output root')
    if sorted(map(tuple, saved['evaluations'])) != sorted(requested):
        raise ValueError('Completed run has a different evaluation coverage')
    for target, percentage in requested:
        path = folder / 'evaluations' / f'{target}_lbl{size_key(percentage)}.json'
        row = json.loads(path.read_text())
        train_set, test_set = datasets[target]
        if (not row.get('complete') or row['expected']['identity'] != identity
                or row['encoder_sha256'] != saved['encoder_sha256']
                or row['expected']['train_fingerprint'] != train_set.fingerprint
                or row['expected']['test_fingerprint'] != test_set.fingerprint
                or file_hash(path) != saved['result_hashes'][path.name]):
            raise ValueError(f'Changed or invalid completed result: {path}')
    if identity['selection_method'] == 'full':
        checkpoint = folder / 'encoder.pt'
        if not checkpoint.exists() or file_hash(checkpoint) != saved['checkpoint_file_sha256']:
            raise ValueError(f'Full-data checkpoint missing or changed: {checkpoint}')
    return True


def clean_temporary_checkpoints(folder, full):
    # Only exact files generated by this run, only after verified completion.
    (folder / 'training.pt').unlink(missing_ok=True)
    if not full:
        (folder / 'encoder.pt').unlink(missing_ok=True)


def aggregate(group, seeds, requested, signature):
    with exclusive_run(group / '_aggregate'):
        markers = []
        for seed in seeds:
            path = group / f'seed_{seed:03d}' / 'complete.json'
            if not path.exists():
                return
            marker = json.loads(path.read_text())
            if not marker.get('complete') or marker['identity']['signature'] != signature:
                raise ValueError('Refusing to aggregate incompatible seed results')
            markers.append(marker)
        output = []
        for target, label in requested:
            rows = []
            for seed, marker in zip(seeds, markers):
                path = group / f'seed_{seed:03d}' / 'evaluations' / f'{target}_lbl{size_key(label)}.json'
                if file_hash(path) != marker['result_hashes'][path.name]:
                    raise ValueError(f'Result hash mismatch: {path}')
                result = json.loads(path.read_text())
                if not result['encoder_unchanged']:
                    raise ValueError('Probe modified encoder')
                rows.append(result)
            values = [row['accuracy_percent'] for row in rows]
            output.append({'target': target, 'label_percentage': label, 'seeds': seeds,
                           'mean_percent': statistics.mean(values),
                           'std_percent': statistics.stdev(values) if len(values) > 1 else 0.,
                           'accuracies_percent': values, 'expected_runs': len(seeds),
                           'completed_runs': len(rows), 'complete': True})
        atomic_json(group / 'summary.json', {'signature': signature, 'complete': True,
                                           'results': output, 'std_ddof': 1})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', choices=('CIFAR10', 'CIFAR100', 'TinyImageNet'), required=True)
    parser.add_argument('--model', choices=('ConvNet', 'VGG11', 'ResNet18'), required=True)
    parser.add_argument('--method', choices=('full', 'random', 'kmeans', 'mkdt', 'ours', 'none'), required=True)
    parser.add_argument('--ssl', choices=('simclr', 'barlowtwins', 'none'), required=True)
    parser.add_argument('--subset', type=float, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', required=True)
    parser.add_argument('--protocol', type=Path, default=HERE / 'protocol.json')
    parser.add_argument('--inputs', type=Path, default=HERE / 'inputs.json')
    parser.add_argument('--data-root', type=Path, default=ROOT.parent / 'data')
    parser.add_argument('--output-root', type=Path, default=HERE / 'results')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--preflight', action='store_true', help='Read and validate inputs; do not train')
    args = parser.parse_args()
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        parser.error('Use one process per runner; distributed loss/BN is not configured')
    protocol = json.loads(args.protocol.read_text())
    validate_protocol(protocol)
    full = args.method == 'full'
    expected_seeds = protocol['full_seeds'] if full else protocol['subset_seeds']
    if not args.seeds or len(set(args.seeds)) != len(args.seeds) or not set(args.seeds) <= set(expected_seeds):
        parser.error(f'Seeds must be a distinct subset of {expected_seeds}')
    if ((args.method == 'none') != (args.ssl == 'none') or
            (full and args.subset != 100) or (args.method == 'none' and args.subset != 0) or
            (args.method not in ('none', 'full') and args.subset not in protocol['subset_percentages'])):
        parser.error('Invalid method/SSL/subset combination')
    if args.workers < 0 or args.threads < 1:
        parser.error('Invalid worker/thread count')
    if args.device.startswith('cuda') and not torch.cuda.is_available() and not args.preflight:
        parser.error('CUDA is unavailable; run on a GPU machine, or explicitly use --device cpu for tests')
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    requested = expected_evaluations(args.source, args.model, full, protocol)
    synthetic = None
    if args.method in ('mkdt', 'ours'):
        synthetic = SyntheticDataset(synthetic_entry(args.inputs, args.method, args.source, args.subset),
                                     args.source, args.subset, protocol['input_size'][args.source])
    # Fail on missing target data BEFORE an expensive SSL run.
    datasets = {target: (RawDataset(args.data_root, target, True),
                         RawDataset(args.data_root, target, False))
                for target in evaluation_targets(args.source, args.model)}
    source_set = datasets[args.source][0]
    if len(source_set) != SOURCE_COUNTS[args.source]:
        raise ValueError('Unexpected source training split size')
    if args.preflight:
        print(json.dumps({'status': 'ready', 'source': args.source, 'targets': list(datasets),
                          'synthetic': synthetic.metadata if synthetic is not None else None,
                          'evaluations_per_seed': requested}, indent=2))
        return
    signature = {'protocol': protocol, 'code_hashes': code_hashes(),
                 'source': args.source, 'source_fingerprint': source_set.fingerprint,
                 'target_fingerprints': {name: [train_data.fingerprint, test_data.fingerprint]
                                         for name, (train_data, test_data) in datasets.items()},
                 'model': args.model, 'selection_method': args.method, 'ssl': args.ssl,
                 'subset': args.subset, 'synthetic': synthetic.metadata if synthetic is not None else None}
    signature_hash = digest_json(signature)
    group = args.output_root / args.source / args.model / args.method / f'pt{size_key(args.subset)}' / args.ssl
    for seed in args.seeds:
        folder = group / f'seed_{seed:03d}'
        with exclusive_run(folder):
            if args.method in ('random', 'kmeans'):
                selected, selection = select_subset(source_set, args.method, args.subset, seed, protocol,
                                                    args.output_root / 'selections', args.workers, args.device)
            elif synthetic is not None:
                selected, selection = synthetic, synthetic.metadata
            else:
                selected, selection = source_set, {'scope': args.method}
            identity = {'signature': signature_hash, 'seed': seed, 'selection': selection,
                        'selection_method': args.method}
            if validate_completed(folder, identity, datasets, requested):
                clean_temporary_checkpoints(folder, full)
                print(f'SKIP verified complete run: {folder}', flush=True)
                continue
            config_path = folder / 'config.json'
            config = {'identity': identity, 'signature': signature}
            if config_path.exists() and json.loads(config_path.read_text()) != config:
                raise ValueError(f'Output config mismatch: {folder}; choose a fresh output root')
            atomic_json(config_path, config)
            atomic_json(folder / 'environment.json', {'python': sys.version, 'torch': str(torch.__version__),
                        'torchvision': str(torchvision.__version__), 'device': args.device,
                        'gpu': torch.cuda.get_device_name() if args.device.startswith('cuda') else None,
                        'workers': args.workers, 'threads': args.threads})
            checkpoint, metadata = train(selected, protocol, args.source, args.model, args.ssl,
                                          seed, folder, identity, args.workers, args.device, full=full)
            checkpoint_file_hash = file_hash(checkpoint)
            hashes = {}
            for target, percentage in requested:
                destination = folder / 'evaluations' / f'{target}_lbl{size_key(percentage)}.json'
                result = linear_probe(checkpoint, identity, protocol, args.source, target,
                                      *datasets[target], percentage, seed, destination, args.workers, args.device)
                if result['encoder_sha256'] != metadata['encoder_sha256']:
                    raise RuntimeError('Evaluations did not use the same encoder')
                hashes[destination.name] = file_hash(destination)
                gc.collect()
            atomic_json(folder / 'complete.json', {'complete': True, 'identity': identity,
                        'encoder_sha256': metadata['encoder_sha256'],
                        'checkpoint_file_sha256': checkpoint_file_hash,
                        'checkpoint_retained': full, 'evaluations': requested,
                        'result_hashes': hashes,
                        'ssl': {k: metadata[k] for k in ('epochs', 'steps', 'effective_batch')}})
            clean_temporary_checkpoints(folder, full)
            gc.collect()
        aggregate(group, expected_seeds, requested, signature_hash)
    aggregate(group, expected_seeds, requested, signature_hash)


if __name__ == '__main__':
    main()
