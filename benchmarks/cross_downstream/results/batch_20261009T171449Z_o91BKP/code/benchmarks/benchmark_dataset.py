"""Same-dataset SSL baselines for CIFAR-100 and Tiny ImageNet."""

import argparse
import gc
import json
import math
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import benchmark_cifar10 as shared
from downstream_data import SPECS, load_target


PROTOCOL = 'main_datasets_v1_random_labels_barlow2x'
DATASETS = ('CIFAR100', 'TinyImageNet')


def labeled_indices(labels, percentage, seed):
    generator = np.random.default_rng(seed + 5000)
    count = max(1, int(len(labels) * percentage / 100))
    return generator.permutation(len(labels))[:count].astype(np.int64, copy=False)


def linear_probe(network, train_images, train_labels, test_images, test_labels, classes, epochs, seed):
    network.eval().requires_grad_(False)
    train_features = shared.extract_features(network, train_images)
    test_features = shared.extract_features(network, test_images)
    shared.seed_all(seed + 6000)
    head = nn.Linear(train_features.shape[1], classes).to(train_images.device)
    optimizer = torch.optim.SGD(head.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0005)
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


def prepare(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    train, labels, test, test_labels, classes = load_target(args.target_cache, args.device, args.dataset)
    del labels, test, test_labels
    for seed in range(args.seed_start, args.seed_start + args.runs):
        started = time.perf_counter()
        network = shared.get_network('ConvNet', 3, classes, (32, 32), seed=seed + 2000)
        features = shared.extract_features(network, train)
        random_order = np.random.default_rng(seed + 1000).permutation(len(train))
        selections = {
            'protocol': PROTOCOL,
            'dataset': args.dataset,
            'seed': seed,
            'feature_model': 'random_ConvNet',
            'kmeans_iterations': 50,
            'random': {},
            'kmeans': {},
        }
        for percentage in (1, 2, 5):
            count = len(train) * percentage // 100
            selections['random'][str(percentage)] = random_order[:count].tolist()
            selections['kmeans'][str(percentage)] = shared.kmeans_indices(features, count, seed)
        shared.write_json(output / f'seed_{seed:03d}.json', selections)
        del network, features
        gc.collect()
        print(f'PREPARE dataset={args.dataset} seed={seed} seconds={time.perf_counter() - started:.1f}', flush=True)


def selection_indices(args, count, seed):
    expected = count * args.subset_percentage // 100
    if args.method == 'random':
        return np.random.default_rng(seed + 1000).permutation(count)[:expected].tolist()
    selection = json.loads((Path(args.selection_dir) / f'seed_{seed:03d}.json').read_text())
    if selection.get('protocol') != PROTOCOL or selection.get('dataset') != args.dataset or selection.get('seed') != seed:
        raise ValueError('Incompatible selection cache')
    indices = selection['kmeans'][str(args.subset_percentage)]
    if len(indices) != expected or len(set(indices)) != expected or min(indices) < 0 or max(indices) >= count:
        raise ValueError('Invalid K-means selection')
    return indices


def run(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    settings = dict(vars(args), protocol=PROTOCOL, label_selection='uniform_random_without_replacement',
                    temperature=0.2, batch_size=256, std_ddof=1,
                    no_pretrain_protocol='frozen_random_encoder',
                    ssl_aug_strategy=shared.STRATEGY, ssl_aug_mode='S')
    shared.write_json(output / 'config.json', settings)
    train, labels, test, test_labels, classes = load_target(args.target_cache, args.device, args.dataset)
    labels_cpu = labels.cpu().numpy()
    rows = []
    for seed in range(args.seed_start, args.seed_start + args.runs):
        started = time.perf_counter()
        chosen = torch.tensor(labeled_indices(labels_cpu, args.label_percentage, seed), device=args.device)
        network = shared.get_network(args.model, 3, classes, (32, 32), seed=seed + 3000)
        ssl_steps = 0
        subset_indices = []
        if args.method == 'no_pretrain':
            accuracy = linear_probe(network, train[chosen], labels[chosen], test, test_labels,
                                    classes, args.probe_epochs, seed)
        else:
            if args.method == 'full':
                source_images = train
            else:
                subset_indices = selection_indices(args, len(train), seed)
                source_images = train[torch.tensor(subset_indices, device=args.device)]
            ssl_steps = shared.train_ssl(network, source_images, args.ssl_method, args.ssl_epochs,
                                         seed, log_every=args.ssl_log_every)
            accuracy = linear_probe(network, train[chosen], labels[chosen], test, test_labels,
                                    classes, args.probe_epochs, seed)
            del source_images
        row = {
            'seed': seed,
            'accuracy_percent': accuracy,
            'seconds': time.perf_counter() - started,
            'ssl_steps': ssl_steps,
            'labeled_indices': chosen.cpu().tolist(),
            'subset_indices': subset_indices,
        }
        shared.write_json(output / f'seed_{seed:03d}.json', row)
        rows.append(row)
        scores = [value['accuracy_percent'] for value in rows]
        summary = {
            'config': settings,
            'completed_runs': len(scores),
            'expected_runs': args.runs,
            'complete': len(scores) == args.runs,
            'mean_percent': statistics.mean(scores),
            'std_percent': statistics.stdev(scores) if len(scores) > 1 else None,
            'seeds': [value['seed'] for value in rows],
            'accuracies_percent': scores,
        }
        shared.write_json(output / 'summary.json', summary)
        print(f'RUN dataset={args.dataset} seed={seed} accuracy={accuracy:.4f}% '
              f'seconds={row["seconds"]:.1f}', flush=True)
        del network
        gc.collect()
    print(f'COMPLETE n={len(scores)} mean={summary["mean_percent"]:.4f} '
          f'std={summary["std_percent"]}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'run'))
    parser.add_argument('--dataset', required=True, choices=DATASETS)
    parser.add_argument('--target-cache', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--selection-dir')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--method', choices=('no_pretrain', 'random', 'kmeans', 'full'))
    parser.add_argument('--model', choices=('ConvNet', 'VGG11', 'ResNet18'))
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
    if args.command == 'prepare':
        prepare(args)
        return
    if not args.model or not args.method or args.probe_epochs < 1:
        parser.error('run requires model, method, and positive probe epochs')
    if args.method == 'no_pretrain':
        if args.ssl_method != 'none' or args.subset_percentage != 0:
            parser.error('no_pretrain requires ssl-method none and subset-percentage 0')
    else:
        if args.ssl_method not in ('simclr', 'barlowtwins') or args.ssl_epochs < 1:
            parser.error('pretraining requires an SSL method and positive SSL epochs')
        if args.method == 'full' and args.subset_percentage != 100:
            parser.error('full requires subset-percentage 100')
        if args.method in ('random', 'kmeans') and args.subset_percentage not in (1, 2, 5):
            parser.error('subset methods require 1, 2, or 5 percent')
        if args.method == 'kmeans' and not args.selection_dir:
            parser.error('kmeans requires selection-dir')
    torch.set_num_threads(int(__import__('os').environ.get('SLURM_CPUS_PER_TASK', '1')))
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('A GPU allocation is required')
    shared.seed_all(0)
    run(args)


if __name__ == '__main__':
    main()
