"""Images-only MKDT evaluation for CIFAR-100 and Tiny ImageNet tables."""

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


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import benchmark_cifar10 as shared
from downstream_data import SPECS, load_target, select_labels


PROTOCOL = 'mkdt_dual_generator_v2_random_labels_barlow2x'
MODELS = ('ConvNet', 'ResNet18')
GENERATORS = ('ConvNet', 'ResNet18BN')
LOSSES = ('simclr', 'barlowtwins')
SIZES = (1, 2, 5)
LABELS = (1, 5)
RUNS = 10
PROBE_EPOCHS = {1: 500, 5: 300}
SOURCES = {
    'CIFAR10': {
        'count': 50000,
        'raw_side': 32,
        'targets': ('CIFAR100', 'TinyImageNet', 'Aircraft', 'CUB2011', 'Dogs', 'Flowers'),
    },
    'CIFAR100': {
        'count': 50000,
        'raw_side': 32,
        'targets': ('TinyImageNet', 'CIFAR10', 'Aircraft', 'CUB2011', 'Dogs', 'Flowers'),
    },
    'TinyImageNet': {
        'count': 100000,
        'raw_side': 64,
        'targets': ('CIFAR10', 'CIFAR100', 'Aircraft', 'CUB2011', 'Dogs', 'Flowers'),
    },
}


def sha256(path):
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


def compatible_config(path, expected):
    path = Path(path)
    if not path.exists():
        shared.write_json(path, expected)
        return
    actual = json.loads(path.read_text())
    if actual != expected:
        raise ValueError(f'Existing output has an incompatible config: {path}')


def load_images(path, source, size, device):
    path = Path(path)
    if path.name.startswith(('labels_', 'synlr_')):
        raise ValueError(f'Expected images_<step>.pt, not {path}')
    images = torch.load(path, map_location='cpu', weights_only=True)
    spec = SOURCES[source]
    expected = (spec['count'] * size // 100, 3, spec['raw_side'], spec['raw_side'])
    if not isinstance(images, torch.Tensor) or tuple(images.shape) != expected:
        raise ValueError(f'{path}: expected tensor {expected}, found {getattr(images, "shape", None)}')
    if not images.is_floating_point() or not torch.isfinite(images).all():
        raise ValueError(f'{path}: expected finite floating-point MKDT images')
    images = images.contiguous()
    if spec['raw_side'] != 32:
        images = F.interpolate(images, size=(32, 32), mode='bicubic',
                               align_corners=False, antialias=True)
    return images.to(device)


def pretrain_settings(args):
    digest = sha256(args.images)
    settings = {
        'protocol': PROTOCOL,
        'source': args.source,
        'generator': args.generator,
        'source_classes': SPECS[args.source][0],
        'model': args.model,
        'ssl_method': args.ssl_method,
        'subset_percentage': args.subset_percentage,
        'ssl_epochs': args.ssl_epochs,
        'runs': RUNS,
        'synthetic_path': str(Path(args.images).resolve()),
        'synthetic_sha256': digest,
        'input_scope': 'images_only_no_labels_pt_no_synlr_pt',
        'evaluation_image_size': 32,
        'tiny_synthetic_resize': 'bicubic_antialias_64_to_32' if args.source == 'TinyImageNet' else 'none',
        'temperature': 0.2,
        'batch_size': 256,
        'ssl_aug_strategy': shared.STRATEGY,
        'ssl_aug_mode': 'S',
    }
    return digest, settings


def prepare_pretrain(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    _, settings = pretrain_settings(args)
    compatible_config(output / 'config.json', settings)
    print(f'PREPARED pretrain config: {output}', flush=True)


def pretrain(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    digest, settings = pretrain_settings(args)
    compatible_config(output / 'config.json', settings)
    images = load_images(args.images, args.source, args.subset_percentage, args.device)
    requested = list(range(args.seed_start, args.seed_start + args.runs))
    for seed in requested:
        destination = output / f'seed_{seed:03d}.pt'
        if destination.exists():
            checkpoint = torch.load(destination, map_location='cpu', weights_only=True)
            expected = (PROTOCOL, args.source, args.generator, args.model, seed, digest)
            actual = tuple(checkpoint.get(key) for key in
                           ('protocol', 'source', 'generator', 'model', 'seed', 'synthetic_sha256'))
            if actual != expected:
                raise ValueError(f'Incompatible checkpoint: {destination}')
            print(f'SKIP pretrain seed={seed}: checkpoint exists', flush=True)
            continue
        started = time.perf_counter()
        network = shared.get_network(args.model, 3, SPECS[args.source][0],
                                     (32, 32), seed=seed + 3000)
        steps = shared.train_ssl(network, images, args.ssl_method, args.ssl_epochs,
                                 seed, log_every=args.ssl_log_every)
        checkpoint = {
            **settings,
            'seed': seed,
            'ssl_steps': steps,
            'state_dict': {key: value.detach().cpu()
                           for key, value in network.state_dict().items()},
        }
        atomic_torch_save(destination, checkpoint)
        print(f'PRETRAIN source={args.source} model={args.model} seed={seed} '
              f'seconds={time.perf_counter() - started:.1f}', flush=True)
        del network, checkpoint
        gc.collect()
    completed = [seed for seed in requested
                 if (output / f'seed_{seed:03d}.pt').is_file()]
    shard_summary = {
        'config': settings,
        'complete': completed == requested,
        'completed_runs': len(completed),
        'expected_runs': len(requested),
        'seeds': completed,
    }
    shared.write_json(
        output / f'shard_{requested[0]:03d}_{requested[-1]:03d}.json',
        shard_summary)
    if args.seed_start == 0 and args.runs == RUNS:
        shared.write_json(output / 'summary.json', shard_summary)
    if not shard_summary['complete']:
        raise RuntimeError(
            f'Only {len(completed)}/{len(requested)} requested encoders are complete')


def finalize_pretrain(args):
    output = Path(args.output)
    config_path = output / 'config.json'
    if not config_path.is_file():
        raise FileNotFoundError(f'Missing pretraining config: {config_path}')
    settings = json.loads(config_path.read_text())
    expected_settings = {
        'protocol': PROTOCOL,
        'source': args.source,
        'generator': args.generator,
        'model': args.model,
        'ssl_method': args.ssl_method,
        'subset_percentage': args.subset_percentage,
        'ssl_epochs': args.ssl_epochs,
        'runs': RUNS,
        'input_scope': 'images_only_no_labels_pt_no_synlr_pt',
    }
    for key, value in expected_settings.items():
        if settings.get(key) != value:
            raise ValueError(
                f'{config_path}: unexpected {key}={settings.get(key)!r}; expected {value!r}')
    completed = []
    for seed in range(RUNS):
        path = output / f'seed_{seed:03d}.pt'
        if not path.is_file():
            continue
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        expected = {
            **expected_settings,
            'seed': seed,
            'synthetic_sha256': settings['synthetic_sha256'],
        }
        for key, value in expected.items():
            if checkpoint.get(key) != value:
                raise ValueError(f'{path}: unexpected {key}')
        completed.append(seed)
    summary = {
        'config': settings,
        'complete': completed == list(range(RUNS)),
        'completed_runs': len(completed),
        'expected_runs': RUNS,
        'seeds': completed,
    }
    shared.write_json(output / 'summary.json', summary)
    if not summary['complete']:
        raise RuntimeError(f'Only {len(completed)}/{RUNS} encoders are complete')
    print(f'FINALIZED {len(completed)}/{RUNS} encoders: {output}', flush=True)


def load_encoder(args, seed):
    path = Path(args.encoder_dir) / f'seed_{seed:03d}.pt'
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    expected = {
        'protocol': PROTOCOL,
        'source': args.source,
        'generator': args.generator,
        'model': args.model,
        'ssl_method': args.ssl_method,
        'subset_percentage': args.subset_percentage,
        'ssl_epochs': args.ssl_epochs,
        'seed': seed,
        'input_scope': 'images_only_no_labels_pt_no_synlr_pt',
    }
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise ValueError(f'{path}: unexpected {key}')
    network = shared.get_network(args.model, 3, checkpoint['source_classes'],
                                 (32, 32), seed=seed + 3000)
    network.load_state_dict(checkpoint['state_dict'], strict=True)
    return network.eval().requires_grad_(False), checkpoint['synthetic_sha256']


def probe(train_features, train_labels, test_features, test_labels,
          classes, epochs, seed):
    shared.seed_all(seed + 6000)
    head = nn.Linear(train_features.shape[1], classes).to(train_features.device)
    optimizer = torch.optim.SGD(head.parameters(), lr=0.1, momentum=0.9,
                                weight_decay=0.0005)
    for epoch in range(epochs):
        if epoch == epochs // 2:
            for group in optimizer.param_groups:
                group['lr'] *= 0.1
        for indices in torch.randperm(len(train_features),
                                      device=train_features.device).split(256):
            loss = F.cross_entropy(head(train_features[indices]), train_labels[indices])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    with torch.no_grad():
        return (head(test_features).argmax(1) == test_labels).float().mean().item() * 100


def write_summary(folder, settings):
    rows = []
    for seed in range(RUNS):
        path = Path(folder) / f'seed_{seed:03d}.json'
        if path.exists():
            rows.append(json.loads(path.read_text()))
    values = [row['accuracy_percent'] for row in rows]
    summary = {
        'config': settings,
        'complete': len(rows) == RUNS and [row['seed'] for row in rows] == list(range(RUNS)),
        'completed_runs': len(rows),
        'expected_runs': RUNS,
        'mean_percent': statistics.mean(values) if values else None,
        'std_percent': statistics.stdev(values) if len(values) > 1 else None,
        'seeds': [row['seed'] for row in rows],
        'accuracies_percent': values,
    }
    shared.write_json(Path(folder) / 'summary.json', summary)
    return summary


def evaluate_target(args, target):
    cache = Path(args.target_cache_root) / f'{target}.pt'
    train, labels, test, test_labels, classes = load_target(cache, args.device, target)
    labels_cpu = labels.cpu().numpy()
    folders = {}
    settings_by_label = {}
    for percentage in LABELS:
        folder = Path(args.output) / 'evaluations' / f'{target}_lbl{percentage}'
        folder.mkdir(parents=True, exist_ok=True)
        settings = {
            'protocol': PROTOCOL,
            'source': args.source,
            'generator': args.generator,
            'target': target,
            'model': args.model,
            'ssl_method': args.ssl_method,
            'subset_percentage': args.subset_percentage,
            'ssl_epochs': args.ssl_epochs,
            'label_percentage': percentage,
            'probe_epochs': PROBE_EPOCHS[percentage],
            'runs': RUNS,
            'label_selection': 'uniform_random_without_replacement',
            'probe_protocol': 'frozen_encoder',
            'input_scope': 'images_only_no_labels_pt_no_synlr_pt',
            'normalization': 'target_dataset_repository_statistics_32x32',
            'std_ddof': 1,
        }
        compatible_config(folder / 'config.json', settings)
        folders[percentage] = folder
        settings_by_label[percentage] = settings

    for seed in range(RUNS):
        pending = [percentage for percentage in LABELS
                   if not (folders[percentage] / f'seed_{seed:03d}.json').exists()]
        if not pending:
            continue
        started = time.perf_counter()
        network, digest = load_encoder(args, seed)
        train_features = shared.extract_features(network, train)
        test_features = shared.extract_features(network, test)
        for percentage in pending:
            indices = select_labels(labels_cpu, percentage, seed, 'random')
            chosen = torch.tensor(indices, device=args.device)
            accuracy = probe(train_features[chosen], labels[chosen], test_features,
                             test_labels, classes, PROBE_EPOCHS[percentage], seed)
            row = {
                'seed': seed,
                'accuracy_percent': accuracy,
                'labeled_indices': indices.tolist(),
                'labeled_count': len(indices),
                'synthetic_sha256': digest,
                'seconds': time.perf_counter() - started,
            }
            shared.write_json(folders[percentage] / f'seed_{seed:03d}.json', row)
            print(f'RUN source={args.source} target={target} model={args.model} '
                  f'labels={percentage}% seed={seed} accuracy={accuracy:.4f}%', flush=True)
        del network, train_features, test_features
        gc.collect()

    incomplete = []
    for percentage in LABELS:
        summary = write_summary(folders[percentage], settings_by_label[percentage])
        if not summary['complete']:
            incomplete.append(f'{target}:{percentage}%')
    del train, labels, test, test_labels
    gc.collect()
    return incomplete


def evaluate(args):
    targets = (args.source,)
    if args.model == 'ResNet18':
        targets += SOURCES[args.source]['targets']
    incomplete = []
    for target in targets:
        incomplete.extend(evaluate_target(args, target))
    if incomplete:
        raise RuntimeError('Incomplete evaluations: ' + ', '.join(incomplete))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'command', choices=('prepare-pretrain', 'pretrain',
                            'finalize-pretrain', 'evaluate'))
    parser.add_argument('--source', choices=tuple(SOURCES), required=True)
    parser.add_argument('--generator', choices=GENERATORS, required=True)
    parser.add_argument('--images')
    parser.add_argument('--encoder-dir')
    parser.add_argument('--target-cache-root')
    parser.add_argument('--output', required=True)
    parser.add_argument('--model', choices=MODELS, required=True)
    parser.add_argument('--ssl-method', choices=LOSSES, required=True)
    parser.add_argument('--subset-percentage', type=int, choices=SIZES, required=True)
    parser.add_argument('--ssl-epochs', type=int, required=True)
    parser.add_argument('--ssl-log-every', type=int, default=0)
    parser.add_argument('--seed-start', type=int, default=0)
    parser.add_argument('--runs', type=int, default=RUNS)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    args = parser.parse_args()
    if args.ssl_epochs < 1 or args.ssl_log_every < 0:
        parser.error('epoch counts must be valid')
    if args.seed_start < 0 or args.runs < 1 or args.seed_start + args.runs > RUNS:
        parser.error(f'seed range must be within 0..{RUNS - 1}')
    if args.command in ('prepare-pretrain', 'pretrain') and not args.images:
        parser.error(f'{args.command} requires --images')
    if args.command == 'evaluate' and not args.encoder_dir:
        parser.error('evaluate requires --encoder-dir')
    if args.command == 'evaluate' and not args.target_cache_root:
        parser.error('evaluate requires --target-cache-root')
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', '1')))
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('A GPU allocation is required')
    shared.seed_all(0)
    {
        'prepare-pretrain': prepare_pretrain,
        'pretrain': pretrain,
        'finalize-pretrain': finalize_pretrain,
        'evaluate': evaluate,
    }[args.command](args)


if __name__ == '__main__':
    main()
