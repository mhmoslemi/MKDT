"""Fresh full-data SSL pretraining and main/downstream linear-probe evaluation."""

import argparse
import gc
import json
import os
from pathlib import Path
import statistics
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import benchmark_cifar10 as shared
import benchmark_dataset as main_dataset
from downstream_data import SPECS, load_target, select_labels


PROTOCOL = 'full_data_refresh_v1_random_labels_barlow2x'
SOURCES = ('CIFAR10', 'CIFAR100', 'TinyImageNet')
MODELS = ('ConvNet', 'VGG11', 'ResNet18')


def atomic_save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def pretrain(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    settings = dict(vars(args), protocol=PROTOCOL, method='full',
                    subset_percentage=100, temperature=0.2, batch_size=256,
                    ssl_aug_strategy=shared.STRATEGY, ssl_aug_mode='S')
    shared.write_json(output / 'config.json', settings)
    train, labels, test, test_labels, classes = load_target(
        args.source_cache, args.device, args.source)
    del labels, test, test_labels
    for seed in range(args.seed_start, args.seed_start + args.runs):
        started = time.perf_counter()
        network = shared.get_network(args.model, 3, classes, (32, 32),
                                     seed=seed + 3000)
        steps = shared.train_ssl(network, train, args.ssl_method,
                                 args.ssl_epochs, seed,
                                 log_every=args.ssl_log_every)
        checkpoint = {
            'protocol': PROTOCOL,
            'source_dataset': args.source,
            'source_classes': classes,
            'seed': seed,
            'model': args.model,
            'method': 'full',
            'ssl_method': args.ssl_method,
            'subset_percentage': 100,
            'ssl_epochs': args.ssl_epochs,
            'temperature': 0.2,
            'ssl_aug_strategy': shared.STRATEGY,
            'ssl_aug_mode': 'S',
            'ssl_steps': steps,
            'state_dict': {key: value.detach().cpu()
                           for key, value in network.state_dict().items()},
        }
        atomic_save(output / f'seed_{seed:03d}.pt', checkpoint)
        print(f'PRETRAIN source={args.source} model={args.model} '
              f'ssl={args.ssl_method} seed={seed} steps={steps} '
              f'seconds={time.perf_counter() - started:.1f}', flush=True)
        del network, checkpoint
        gc.collect()


def checkpoint_path(root, seed):
    matches = sorted(Path(root).glob(f'shard_*/seed_{seed:03d}.pt'))
    direct = Path(root) / f'seed_{seed:03d}.pt'
    if direct.is_file():
        matches.append(direct)
    if len(matches) != 1:
        raise FileNotFoundError(
            f'Expected exactly one checkpoint for seed {seed} under {root}; found {matches}')
    return matches[0]


def load_encoder(args, encoder_seed):
    path = checkpoint_path(args.encoder_dir, encoder_seed)
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    expected = {
        'protocol': PROTOCOL,
        'source_dataset': args.source,
        'source_classes': SPECS[args.source][0],
        'seed': encoder_seed,
        'model': args.model,
        'method': 'full',
        'ssl_method': args.ssl_method,
        'subset_percentage': 100,
        'ssl_epochs': args.ssl_epochs,
        'temperature': 0.2,
        'ssl_aug_strategy': shared.STRATEGY,
        'ssl_aug_mode': 'S',
    }
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise ValueError(
                f'{path}: unexpected {key}={checkpoint.get(key)!r}; expected {value!r}')
    network = shared.get_network(args.model, 3, checkpoint['source_classes'],
                                 (32, 32), seed=encoder_seed + 3000)
    network.load_state_dict(checkpoint['state_dict'], strict=True)
    return network.eval().requires_grad_(False)


def linear_probe_features(train_features, train_labels, test_features,
                          test_labels, classes, epochs, seed):
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
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite linear-probe loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    with torch.no_grad():
        return (head(test_features).argmax(1) == test_labels).float().mean().item() * 100


def evaluate(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    settings = dict(vars(args), protocol=PROTOCOL, method='full',
                    subset_percentage=100, std_ddof=1,
                    label_selection='uniform_random_without_replacement',
                    normalization='target_dataset_repository_statistics',
                    image_size=32, temperature=0.2,
                    ssl_aug_strategy=shared.STRATEGY, ssl_aug_mode='S')
    shared.write_json(output / 'config.json', settings)
    train, labels, test, test_labels, classes = load_target(
        args.target_cache, args.device, args.target)
    labels_cpu = labels.cpu().numpy()
    shared_network = None
    train_features = test_features = None
    if args.reuse_encoder_seed is not None:
        shared_network = load_encoder(args, args.reuse_encoder_seed)
        train_features = shared.extract_features(shared_network, train)
        test_features = shared.extract_features(shared_network, test)
    rows = []
    for seed in range(args.runs):
        started = time.perf_counter()
        indices = select_labels(labels_cpu, args.label_percentage, seed, 'random')
        chosen = torch.tensor(indices, device=args.device)
        if shared_network is None:
            network = load_encoder(args, seed)
            accuracy = main_dataset.linear_probe(
                network, train[chosen], labels[chosen], test, test_labels,
                classes, args.probe_epochs, seed)
        else:
            network = None
            accuracy = linear_probe_features(
                train_features[chosen], labels[chosen], test_features, test_labels,
                classes, args.probe_epochs, seed)
        row = {
            'seed': seed,
            'encoder_seed': seed if args.reuse_encoder_seed is None else args.reuse_encoder_seed,
            'accuracy_percent': accuracy,
            'labeled_indices': indices.tolist(),
            'labeled_count': len(indices),
            'requested_label_percentage': args.label_percentage,
            'actual_label_percentage': 100 * len(indices) / len(train),
            'labeled_classes': len(np.unique(labels_cpu[indices])),
            'total_classes': classes,
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
        print(f'EVAL source={args.source} target={args.target} model={args.model} '
              f'ssl={args.ssl_method} labels={args.label_percentage}% seed={seed} '
              f'accuracy={accuracy:.4f}% seconds={row["seconds"]:.1f}', flush=True)
        if network is not None:
            del network
        gc.collect()
    if shared_network is not None:
        del shared_network, train_features, test_features
        gc.collect()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('pretrain', 'evaluate'))
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--output', required=True)
    parser.add_argument('--source', choices=SOURCES, required=True)
    parser.add_argument('--source-cache')
    parser.add_argument('--encoder-dir')
    parser.add_argument('--target', choices=tuple(SPECS))
    parser.add_argument('--target-cache')
    parser.add_argument('--model', choices=MODELS, required=True)
    parser.add_argument('--ssl-method', choices=('simclr', 'barlowtwins'), required=True)
    parser.add_argument('--ssl-epochs', type=int, required=True)
    parser.add_argument('--probe-epochs', type=int, default=0)
    parser.add_argument('--label-percentage', type=int, choices=(1, 5), default=1)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--seed-start', type=int, default=0)
    parser.add_argument('--reuse-encoder-seed', type=int)
    parser.add_argument('--ssl-log-every', type=int, default=0)
    args = parser.parse_args()
    if (args.runs < 1 or args.seed_start < 0 or args.ssl_epochs < 1 or
            (args.reuse_encoder_seed is not None and args.reuse_encoder_seed < 0)):
        parser.error('run and epoch counts must be positive')
    if args.command == 'pretrain' and not args.source_cache:
        parser.error('pretrain requires --source-cache')
    if args.command == 'evaluate':
        if not args.encoder_dir or not args.target or not args.target_cache:
            parser.error('evaluate requires encoder, target, and target cache')
        if args.probe_epochs < 1:
            parser.error('evaluate requires positive --probe-epochs')
        if args.seed_start != 0:
            parser.error('evaluation always consumes seeds 0 through runs-1')
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', '1')))
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('A GPU allocation is required')
    shared.seed_all(0)
    {'pretrain': pretrain, 'evaluate': evaluate}[args.command](args)


if __name__ == '__main__':
    main()
