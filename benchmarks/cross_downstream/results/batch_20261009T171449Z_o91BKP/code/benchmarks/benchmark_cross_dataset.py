"""ResNet18 transfer from CIFAR-100 or Tiny ImageNet to downstream datasets."""

import argparse
import gc
import json
import os
from pathlib import Path
import statistics
import time

import numpy as np
import torch

import benchmark_cifar10 as shared
import benchmark_dataset as main_dataset
from downstream_data import SPECS, load_target, prepare_target, select_labels


PROTOCOL = 'cross_dataset_resnet18_v1_random_labels_barlow2x'
SOURCES = ('CIFAR100', 'TinyImageNet')
TARGETS = tuple(SPECS)


def atomic_save(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def source_indices(args, count, seed):
    if args.method == 'full':
        return []
    expected = count * args.subset_percentage // 100
    if args.method == 'random':
        return np.random.default_rng(seed + 1000).permutation(count)[:expected].tolist()
    path = Path(args.selection_dir) / f'seed_{seed:03d}.json'
    selection = json.loads(path.read_text())
    expected_metadata = {
        'protocol': main_dataset.PROTOCOL,
        'dataset': args.source,
        'seed': seed,
    }
    for key, value in expected_metadata.items():
        if selection.get(key) != value:
            raise ValueError(f'{path}: unexpected {key}={selection.get(key)!r}, expected {value!r}')
    indices = selection['kmeans'][str(args.subset_percentage)]
    if (len(indices) != expected or len(set(indices)) != expected
            or min(indices) < 0 or max(indices) >= count):
        raise ValueError(f'{path}: invalid {args.subset_percentage}% K-means selection')
    return indices


def pretrain(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    settings = dict(vars(args), protocol=PROTOCOL,
                    source_protocol=main_dataset.PROTOCOL,
                    model='ResNet18', temperature=0.2, batch_size=256,
                    ssl_aug_strategy=shared.STRATEGY, ssl_aug_mode='S')
    shared.write_json(output / 'config.json', settings)
    train, labels, test, test_labels, classes = load_target(
        args.source_cache, args.device, args.source)
    del labels, test, test_labels
    for seed in range(args.seed_start, args.seed_start + args.runs):
        started = time.perf_counter()
        indices = source_indices(args, len(train), seed)
        images = train if args.method == 'full' else train[
            torch.tensor(indices, device=args.device)]
        network = shared.get_network('ResNet18', 3, classes, (32, 32),
                                     seed=seed + 3000)
        steps = shared.train_ssl(network, images, args.ssl_method,
                                 args.ssl_epochs, seed,
                                 log_every=args.ssl_log_every)
        checkpoint = {
            'protocol': PROTOCOL,
            'source_protocol': main_dataset.PROTOCOL,
            'source_dataset': args.source,
            'source_classes': classes,
            'seed': seed,
            'model': 'ResNet18',
            'method': args.method,
            'ssl_method': args.ssl_method,
            'subset_percentage': args.subset_percentage,
            'ssl_epochs': args.ssl_epochs,
            'temperature': 0.2,
            'ssl_aug_strategy': shared.STRATEGY,
            'ssl_aug_mode': 'S',
            'source_indices': indices,
            'ssl_steps': steps,
            'state_dict': {
                key: value.detach().cpu()
                for key, value in network.state_dict().items()
            },
        }
        atomic_save(output / f'seed_{seed:03d}.pt', checkpoint)
        print(f'PRETRAIN source={args.source} seed={seed} steps={steps} '
              f'seconds={time.perf_counter() - started:.1f}', flush=True)
        del network, checkpoint, images
        gc.collect()


def load_encoder(args, seed):
    path = Path(args.encoder_dir) / f'seed_{seed:03d}.pt'
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    expected = {
        'protocol': PROTOCOL,
        'source_protocol': main_dataset.PROTOCOL,
        'source_dataset': args.source,
        'source_classes': SPECS[args.source][0],
        'seed': seed,
        'model': 'ResNet18',
        'method': args.method,
        'ssl_method': args.ssl_method,
        'subset_percentage': args.subset_percentage,
        'ssl_epochs': args.ssl_epochs,
        'temperature': 0.2,
        'ssl_aug_strategy': shared.STRATEGY,
        'ssl_aug_mode': 'S',
    }
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise ValueError(
                f'{path}: unexpected {key}={checkpoint.get(key)!r}, expected {value!r}')
    network = shared.get_network('ResNet18', 3, checkpoint['source_classes'],
                                 (32, 32), seed=seed + 3000)
    network.load_state_dict(checkpoint['state_dict'], strict=True)
    network.eval().requires_grad_(False)
    return network


def evaluate(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    settings = dict(vars(args), protocol=PROTOCOL,
                    source_protocol=main_dataset.PROTOCOL,
                    model='ResNet18', std_ddof=1,
                    label_selection='uniform_random_without_replacement',
                    normalization='target_dataset_repository_statistics',
                    image_size=32, temperature=0.2,
                    ssl_aug_strategy=shared.STRATEGY, ssl_aug_mode='S',
                    no_pretrain_protocol='frozen_random_encoder')
    shared.write_json(output / 'config.json', settings)
    train, labels, test, test_labels, classes = load_target(
        args.target_cache, args.device, args.target)
    labels_cpu = labels.cpu().numpy()
    rows = []
    for seed in range(args.seed_start, args.seed_start + args.runs):
        started = time.perf_counter()
        indices = select_labels(labels_cpu, args.label_percentage, seed, 'random')
        chosen = torch.tensor(indices, device=args.device)
        if args.method == 'no_pretrain':
            network = shared.get_network('ResNet18', 3, classes, (32, 32),
                                         seed=seed + 3000)
        else:
            network = load_encoder(args, seed)
        accuracy = main_dataset.linear_probe(
            network, train[chosen], labels[chosen], test, test_labels,
            classes, args.probe_epochs, seed)
        row = {
            'seed': seed,
            'accuracy_percent': accuracy,
            'labeled_indices': indices.tolist(),
            'labeled_count': len(indices),
            'requested_label_percentage': args.label_percentage,
            'actual_label_percentage': 100 * len(indices) / len(train),
            'labeled_classes': len(np.unique(labels_cpu[indices])),
            'total_classes': classes,
            'probe_epochs': args.probe_epochs,
            'seconds': time.perf_counter() - started,
        }
        shared.write_json(output / f'seed_{seed:03d}.json', row)
        rows.append(row)
        scores = [value['accuracy_percent'] for value in rows]
        summary = {
            'config': settings,
            'complete': len(rows) == args.runs,
            'completed_runs': len(rows),
            'expected_runs': args.runs,
            'mean_percent': statistics.mean(scores),
            'std_percent': statistics.stdev(scores) if len(scores) > 1 else None,
            'seeds': [value['seed'] for value in rows],
            'accuracies_percent': scores,
        }
        shared.write_json(output / 'summary.json', summary)
        print(f'RUN source={args.source} target={args.target} seed={seed} '
              f'labels={len(indices)} accuracy={accuracy:.4f}% '
              f'seconds={row["seconds"]:.1f}', flush=True)
        del network
        gc.collect()
    print(f'COMPLETE source={args.source} target={args.target} '
          f'n={len(rows)} mean={summary["mean_percent"]:.4f} '
          f'std={summary["std_percent"]}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare-target', 'pretrain', 'evaluate'))
    parser.add_argument('--data-path', default=os.path.join(os.environ.get('SCRATCH', '.'), 'data'))
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--output', required=True)
    parser.add_argument('--source', choices=SOURCES)
    parser.add_argument('--source-cache')
    parser.add_argument('--selection-dir')
    parser.add_argument('--encoder-dir')
    parser.add_argument('--target', choices=TARGETS)
    parser.add_argument('--target-cache')
    parser.add_argument('--method', choices=('random', 'kmeans', 'full', 'no_pretrain'))
    parser.add_argument('--ssl-method', choices=('none', 'simclr', 'barlowtwins'))
    parser.add_argument('--subset-percentage', type=int, choices=(0, 1, 2, 5, 100), default=1)
    parser.add_argument('--label-percentage', type=int, choices=(1, 5), default=1)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--seed-start', type=int, default=0)
    parser.add_argument('--ssl-epochs', type=int, default=0)
    parser.add_argument('--probe-epochs', type=int, default=0)
    parser.add_argument('--ssl-log-every', type=int, default=0)
    args = parser.parse_args()
    if args.runs < 1 or args.seed_start < 0:
        parser.error('runs must be positive and seed-start nonnegative')
    if args.command == 'prepare-target':
        if not args.target:
            parser.error('prepare-target requires --target')
        shared.write_json(Path(args.output).with_suffix('.json'),
                          prepare_target(args.data_path, args.target, args.output))
        return
    if not args.source or not args.method:
        parser.error('pretrain/evaluate require --source and --method')
    if args.command == 'pretrain':
        if not args.source_cache or not args.selection_dir:
            parser.error('pretrain requires --source-cache and --selection-dir')
        if args.method not in ('random', 'kmeans', 'full'):
            parser.error('pretrain method must be random, kmeans, or full')
        if args.ssl_method not in ('simclr', 'barlowtwins') or args.ssl_epochs < 1:
            parser.error('pretrain requires an SSL method and positive epochs')
    else:
        if not args.target or not args.target_cache or args.probe_epochs < 1:
            parser.error('evaluate requires target/cache and positive probe epochs')
        if args.method == 'no_pretrain':
            if args.ssl_method != 'none' or args.subset_percentage != 0:
                parser.error('no_pretrain requires ssl-method none and subset-percentage 0')
        elif not args.encoder_dir:
            parser.error('pretrained evaluation requires --encoder-dir')
    if args.method == 'full' and args.subset_percentage != 100:
        parser.error('full requires subset-percentage 100')
    if args.method in ('random', 'kmeans') and args.subset_percentage not in (1, 2, 5):
        parser.error('subset methods require 1, 2, or 5 percent')
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', '1')))
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('A GPU allocation is required')
    shared.seed_all(0)
    {'pretrain': pretrain, 'evaluate': evaluate}[args.command](args)


if __name__ == '__main__':
    main()
