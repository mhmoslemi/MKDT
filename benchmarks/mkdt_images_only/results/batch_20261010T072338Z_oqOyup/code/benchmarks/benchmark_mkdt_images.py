"""Images-only MKDT benchmark for the CIFAR-10 main and downstream tables.

This intentionally ignores MKDT's learned target representations and synthetic
learning rate.  A fresh ConvNet is trained with SimCLR or Barlow Twins using
only ``images_<step>.pt`` and is then evaluated with frozen linear probes.
"""

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


BENCHMARKS = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCHMARKS))
import benchmark_cifar10 as shared
from downstream_data import SPECS, load_target, select_labels


PROTOCOL = 'mkdt_cifar10_images_only_convnet_v1_random_labels_barlow2x'
MODEL = 'ConvNet'
RUNS = 10
SIZES = (1, 2, 5)
SSL_METHODS = ('simclr', 'barlowtwins')
TARGETS = ('CIFAR10', 'CIFAR100', 'Aircraft', 'CUB2011', 'Dogs', 'Flowers', 'TinyImageNet')
DOWNSTREAM_TARGETS = TARGETS[1:]
PROBE_EPOCHS = {1: 500, 5: 300}


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_torch_save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def load_mkdt_images(path, size, device):
    """Load only MKDT's image tensor; auxiliary files are never consulted."""
    path = Path(path)
    if path.name.startswith(('labels_', 'synlr_')):
        raise ValueError(f'Expected images_<step>.pt, not an auxiliary MKDT file: {path}')
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(saved, torch.Tensor):
        raise TypeError(f'{path}: expected a bare tensor saved by distill.py, found {type(saved).__name__}')
    expected = (50000 * size // 100, 3, 32, 32)
    if tuple(saved.shape) != expected or not saved.is_floating_point():
        raise ValueError(f'{path}: expected floating tensor {expected}, found {tuple(saved.shape)} {saved.dtype}')
    if not torch.isfinite(saved).all():
        raise ValueError(f'{path}: synthetic images contain non-finite values')
    return saved.contiguous().to(device)


def compatible_config(path, expected):
    path = Path(path)
    if not path.exists():
        shared.write_json(path, expected)
        return
    actual = json.loads(path.read_text())
    if actual != expected:
        raise ValueError(f'Existing output has an incompatible config: {path}')


def completed_seeds(folder, suffix):
    return sorted(int(path.stem.split('_')[-1]) for path in Path(folder).glob(f'seed_*{suffix}'))


def pretrain(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    digest = file_sha256(args.images)
    settings = {
        'protocol': PROTOCOL,
        'source': 'CIFAR10',
        'model': MODEL,
        'ssl_method': args.ssl_method,
        'subset_percentage': args.subset_percentage,
        'ssl_epochs': args.ssl_epochs,
        'runs': args.runs,
        'synthetic_path': str(Path(args.images).resolve()),
        'synthetic_sha256': digest,
        'image_space': 'CIFAR10_normalized',
        'input_scope': 'images_only_no_labels_pt_no_synlr_pt',
        'temperature': 0.2,
        'batch_size': 256,
        'ssl_aug_strategy': shared.STRATEGY,
        'ssl_aug_mode': 'S',
    }
    compatible_config(output / 'config.json', settings)
    images = load_mkdt_images(args.images, args.subset_percentage, args.device)
    metadata = {
        'shape': list(images.shape),
        'dtype': str(images.dtype),
        'minimum': float(images.min().item()),
        'maximum': float(images.max().item()),
        'mean': float(images.mean().item()),
        'std': float(images.std().item()),
        'sha256': digest,
    }
    shared.write_json(output / 'images.json', metadata)

    for seed in range(args.runs):
        destination = output / f'seed_{seed:03d}.pt'
        if destination.exists():
            checkpoint = torch.load(destination, map_location='cpu', weights_only=True)
            if (checkpoint.get('protocol'), checkpoint.get('seed'), checkpoint.get('synthetic_sha256')) != (PROTOCOL, seed, digest):
                raise ValueError(f'Incompatible checkpoint: {destination}')
            print(f'SKIP pretrain seed={seed}: checkpoint exists', flush=True)
            continue
        started = time.perf_counter()
        network = shared.get_network(MODEL, 3, 10, (32, 32), seed=seed + 3000)
        steps = shared.train_ssl(network, images, args.ssl_method, args.ssl_epochs,
                                 seed, log_every=args.ssl_log_every)
        checkpoint = {
            **settings,
            'seed': seed,
            'ssl_steps': steps,
            'state_dict': {key: value.detach().cpu() for key, value in network.state_dict().items()},
        }
        atomic_torch_save(destination, checkpoint)
        print(f'PRETRAIN seed={seed} seconds={time.perf_counter() - started:.1f}', flush=True)
        del network, checkpoint
        gc.collect()

    seeds = completed_seeds(output, '.pt')
    summary = {
        'config': settings,
        'complete': seeds == list(range(args.runs)),
        'completed_runs': len(seeds),
        'expected_runs': args.runs,
        'seeds': seeds,
    }
    shared.write_json(output / 'summary.json', summary)
    if not summary['complete']:
        raise RuntimeError(f'Only {len(seeds)}/{args.runs} encoder checkpoints are complete')


def load_encoder(args, seed):
    path = Path(args.encoder_dir) / f'seed_{seed:03d}.pt'
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    expected = {
        'protocol': PROTOCOL,
        'source': 'CIFAR10',
        'model': MODEL,
        'ssl_method': args.ssl_method,
        'subset_percentage': args.subset_percentage,
        'ssl_epochs': args.ssl_epochs,
        'runs': args.runs,
        'input_scope': 'images_only_no_labels_pt_no_synlr_pt',
    }
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise ValueError(f'{path}: unexpected {key}={checkpoint.get(key)!r}; expected {value!r}')
    network = shared.get_network(MODEL, 3, 10, (32, 32), seed=seed + 3000)
    network.load_state_dict(checkpoint['state_dict'], strict=True)
    return network.eval().requires_grad_(False), checkpoint['synthetic_sha256']


def probe_features(train_features, train_labels, test_features, test_labels,
                   classes, epochs, seed):
    shared.seed_all(seed + 6000)
    head = nn.Linear(train_features.shape[1], classes).to(train_features.device)
    optimizer = torch.optim.SGD(head.parameters(), lr=0.1, momentum=0.9,
                                weight_decay=0.0005)
    for epoch in range(epochs):
        if epoch == epochs // 2:
            for group in optimizer.param_groups:
                group['lr'] *= 0.1
        for indices in torch.randperm(len(train_features), device=train_features.device).split(256):
            loss = F.cross_entropy(head(train_features[indices]), train_labels[indices])
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite linear-probe loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    with torch.no_grad():
        return (head(test_features).argmax(1) == test_labels).float().mean().item() * 100


def write_evaluation_summary(folder, settings, runs):
    folder = Path(folder)
    rows = []
    for seed in range(runs):
        path = folder / f'seed_{seed:03d}.json'
        if path.exists():
            rows.append(json.loads(path.read_text()))
    scores = [row['accuracy_percent'] for row in rows]
    summary = {
        'config': settings,
        'complete': len(rows) == runs and [row['seed'] for row in rows] == list(range(runs)),
        'completed_runs': len(rows),
        'expected_runs': runs,
        'mean_percent': statistics.mean(scores) if scores else None,
        'std_percent': statistics.stdev(scores) if len(scores) > 1 else None,
        'seeds': [row['seed'] for row in rows],
        'accuracies_percent': scores,
    }
    shared.write_json(folder / 'summary.json', summary)
    return summary


def target_data(args, target):
    if target == 'CIFAR10':
        train, labels, test, test_labels = shared.load_data(args.data_path, args.device)
        return train, labels, test, test_labels, 10
    cache = Path(args.target_cache_root) / f'{target}.pt'
    if not cache.is_file():
        raise FileNotFoundError(cache)
    return load_target(cache, args.device, target)


def evaluate_target(args, target):
    train, labels, test, test_labels, classes = target_data(args, target)
    labels_cpu = labels.cpu().numpy()
    folders = {}
    settings_by_label = {}
    for percentage in (1, 5):
        folder = Path(args.output) / 'evaluations' / f'{target}_lbl{percentage}'
        folder.mkdir(parents=True, exist_ok=True)
        settings = {
            'protocol': PROTOCOL,
            'source': 'CIFAR10',
            'target': target,
            'model': MODEL,
            'ssl_method': args.ssl_method,
            'subset_percentage': args.subset_percentage,
            'ssl_epochs': args.ssl_epochs,
            'label_percentage': percentage,
            'probe_epochs': PROBE_EPOCHS[percentage],
            'runs': args.runs,
            'label_selection': 'uniform_random_without_replacement',
            'probe_protocol': 'frozen_encoder',
            'input_scope': 'images_only_no_labels_pt_no_synlr_pt',
            'normalization': 'target_dataset_repository_statistics',
            'std_ddof': 1,
        }
        compatible_config(folder / 'config.json', settings)
        folders[percentage] = folder
        settings_by_label[percentage] = settings

    for seed in range(args.runs):
        pending = [percentage for percentage in (1, 5)
                   if not (folders[percentage] / f'seed_{seed:03d}.json').exists()]
        if not pending:
            print(f'SKIP target={target} seed={seed}: both probes exist', flush=True)
            continue
        started = time.perf_counter()
        network, digest = load_encoder(args, seed)
        train_features = shared.extract_features(network, train)
        test_features = shared.extract_features(network, test)
        for percentage in pending:
            if target == 'CIFAR10':
                indices = shared.labeled_indices(labels_cpu, percentage, seed)
            else:
                indices = select_labels(labels_cpu, percentage, seed, 'random')
            chosen = torch.tensor(indices, device=args.device)
            accuracy = probe_features(
                train_features[chosen], labels[chosen], test_features, test_labels,
                classes, PROBE_EPOCHS[percentage], seed)
            row = {
                'seed': seed,
                'accuracy_percent': accuracy,
                'labeled_indices': indices.tolist(),
                'labeled_count': len(indices),
                'label_percentage': percentage,
                'probe_epochs': PROBE_EPOCHS[percentage],
                'synthetic_sha256': digest,
                'seconds_for_encoder_features_and_this_probe': time.perf_counter() - started,
            }
            shared.write_json(folders[percentage] / f'seed_{seed:03d}.json', row)
            summary = write_evaluation_summary(
                folders[percentage], settings_by_label[percentage], args.runs)
            print(f'RUN target={target} labels={percentage}% seed={seed} '
                  f'accuracy={accuracy:.4f}% completed={summary["completed_runs"]}/{args.runs}',
                  flush=True)
        del network, train_features, test_features
        gc.collect()

    incomplete = []
    for percentage in (1, 5):
        summary = write_evaluation_summary(
            folders[percentage], settings_by_label[percentage], args.runs)
        if not summary['complete']:
            incomplete.append(f'{target}:{percentage}%')
        else:
            print(f'COMPLETE target={target} labels={percentage}% '
                  f'mean={summary["mean_percent"]:.4f} std={summary["std_percent"]:.4f}',
                  flush=True)
    del train, labels, test, test_labels
    gc.collect()
    return incomplete


def evaluate_all(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    incomplete = []
    for target in TARGETS:
        incomplete.extend(evaluate_target(args, target))
    if incomplete:
        raise RuntimeError('Incomplete evaluations: ' + ', '.join(incomplete))


def check_images(args):
    images = load_mkdt_images(args.images, args.subset_percentage, 'cpu')
    print(json.dumps({
        'path': str(Path(args.images).resolve()),
        'shape': list(images.shape),
        'dtype': str(images.dtype),
        'minimum': float(images.min().item()),
        'maximum': float(images.max().item()),
        'sha256': file_sha256(args.images),
        'auxiliary_files_loaded': False,
    }, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('check-images', 'pretrain', 'evaluate-all'))
    parser.add_argument('--images')
    parser.add_argument('--encoder-dir')
    parser.add_argument('--output')
    parser.add_argument('--data-path', default=os.path.join(os.environ.get('SCRATCH', '.'), 'data'))
    parser.add_argument('--target-cache-root')
    parser.add_argument('--ssl-method', choices=SSL_METHODS)
    parser.add_argument('--subset-percentage', type=int, choices=SIZES)
    parser.add_argument('--ssl-epochs', type=int)
    parser.add_argument('--ssl-log-every', type=int, default=0)
    parser.add_argument('--runs', type=int, default=RUNS)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    args = parser.parse_args()
    if args.runs != RUNS:
        parser.error(f'the table protocol requires exactly {RUNS} runs')
    if args.subset_percentage is None:
        parser.error('--subset-percentage is required')
    if args.command in ('check-images', 'pretrain') and not args.images:
        parser.error(f'{args.command} requires --images')
    if args.command == 'pretrain':
        if not args.output or not args.ssl_method or not args.ssl_epochs or args.ssl_epochs < 1:
            parser.error('pretrain requires output, SSL method, and positive SSL epochs')
    if args.command == 'evaluate-all':
        if not all((args.output, args.encoder_dir, args.target_cache_root,
                    args.ssl_method, args.ssl_epochs)):
            parser.error('evaluate-all requires output, encoder dir, target caches, SSL method, and epochs')
    if args.ssl_log_every < 0:
        parser.error('--ssl-log-every must be nonnegative')
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', '1')))
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('A GPU allocation is required')
    shared.seed_all(0)
    {'check-images': check_images, 'pretrain': pretrain,
     'evaluate-all': evaluate_all}[args.command](args)


if __name__ == '__main__':
    main()
