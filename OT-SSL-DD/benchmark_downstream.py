"""CIFAR-10 SSL encoders transferred to five target datasets; run under Slurm."""

import argparse
import gc
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import benchmark_cifar10 as source
from downstream_data import SPECS, load_target, locate, prepare_target, records, select_labels


PROTOCOL = 'cifar10_to_downstream_v1'


def atomic_save(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def preflight(args):
    root = Path(args.data_path)
    if not (root / 'cifar-10-batches-py/data_batch_1').is_file():
        raise FileNotFoundError(f'Stage CIFAR-10 at {root / "cifar-10-batches-py"}')
    for name in args.targets.split(','):
        folder = locate(root, name)
        if name == 'CIFAR100':
            for file in ('train', 'test', 'meta'):
                if not (folder / 'cifar-100-python' / file).is_file():
                    raise FileNotFoundError(folder / 'cifar-100-python' / file)
        else:
            train, test = records(root, name, True), records(root, name, False)
            if not train or not test:
                raise ValueError(f'{name}: empty split')
            if {str(path) for path, _ in train} & {str(path) for path, _ in test}:
                raise ValueError(f'{name}: train/test overlap')
            # The packing job subsequently opens every image and validates classes.
            for path, _ in (train[0], test[0]):
                if not path.is_file():
                    raise FileNotFoundError(path)
        print(f'PREFLIGHT {name}: {folder}', flush=True)


def prepare_selections(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    previous = Path(args.source_batch) / 'selections' if args.source_batch else None
    if previous and all((previous / f'seed_{seed:03d}.json').is_file() for seed in range(args.runs)):
        for seed in range(args.runs):
            path = previous / f'seed_{seed:03d}.json'
            selection = json.loads(path.read_text())
            if selection['protocol'] != source.PROTOCOL or selection['seed'] != seed:
                raise ValueError(f'Incompatible prior selection: {path}')
            for size in (1, 2, 5):
                indices = selection['kmeans'][str(size)]
                if len(indices) != size * 500 or len(set(indices)) != len(indices) or min(indices) < 0 or max(indices) >= 50000:
                    raise ValueError(f'Invalid K-means selection: {path}, {size}%')
            shutil.copyfile(path, output / path.name)
        print(f'Reused {args.runs} existing CIFAR-10 selections from {previous}', flush=True)
    else:
        # Reuse the established CIFAR-10 selection implementation if unavailable.
        source.prepare(argparse.Namespace(output=str(output), data_path=args.data_path,
                                         device=args.device, seed_start=0, runs=args.runs))


def source_indices(args, count, seed):
    if args.method == 'full':
        return []
    expected = count * args.subset_percentage // 100
    if args.method == 'random':
        return np.random.default_rng(seed + 1000).permutation(count)[:expected].tolist()
    selection = json.loads((Path(args.selection_dir) / f'seed_{seed:03d}.json').read_text())
    if selection['protocol'] != source.PROTOCOL or selection['seed'] != seed:
        raise ValueError('Incompatible K-means selection')
    indices = selection['kmeans'][str(args.subset_percentage)]
    if len(indices) != expected or len(set(indices)) != expected or min(indices) < 0 or max(indices) >= count:
        raise ValueError('Invalid K-means indices')
    return indices


def pretrain(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    source.write_json(output / 'config.json', dict(vars(args), protocol=PROTOCOL, source_protocol=source.PROTOCOL))
    images, labels, test, test_labels = source.load_data(args.data_path, args.device)
    del labels, test, test_labels
    for seed in range(args.runs):
        started = time.perf_counter()
        indices = source_indices(args, len(images), seed)
        train = images if args.method == 'full' else images[torch.tensor(indices, device=args.device)]
        network = source.get_network(args.model, 3, 10, (32, 32), seed=seed + 3000)
        steps = source.train_ssl(network, train, args.ssl_method, args.ssl_epochs, seed)
        checkpoint = dict(protocol=PROTOCOL, source_protocol=source.PROTOCOL, seed=seed,
                          model=args.model, method=args.method, ssl_method=args.ssl_method,
                          subset_percentage=args.subset_percentage, ssl_epochs=args.ssl_epochs,
                          temperature=0.2, ssl_aug_strategy=source.STRATEGY, ssl_aug_mode='S',
                          source_indices=indices, ssl_steps=steps,
                          state_dict={key: value.detach().cpu() for key, value in network.state_dict().items()})
        atomic_save(output / f'seed_{seed:03d}.pt', checkpoint)
        print(f'PRETRAIN seed={seed} seconds={time.perf_counter() - started:.1f}', flush=True)
        del network, checkpoint, train
        gc.collect()


def load_encoder(args, seed):
    path = Path(args.encoder_dir) / f'seed_{seed:03d}.pt'
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    expected = dict(protocol=PROTOCOL, source_protocol=source.PROTOCOL, seed=seed,
                    model=args.model, method=args.method, ssl_method=args.ssl_method,
                    subset_percentage=args.subset_percentage, ssl_epochs=args.ssl_epochs,
                    temperature=0.2, ssl_aug_strategy=source.STRATEGY, ssl_aug_mode='S')
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise ValueError(f'{path}: unexpected {key}={checkpoint.get(key)!r}, expected {value!r}')
    network = source.get_network(args.model, 3, 10, (32, 32), seed=seed + 3000)
    network.load_state_dict(checkpoint['state_dict'], strict=True)
    network.eval().requires_grad_(False)
    return network


def probe(network, train, labels, test, test_labels, classes, epochs, seed):
    network.eval().requires_grad_(False)
    features = source.extract_features(network, train)
    test_features = source.extract_features(network, test)
    source.seed_all(seed + 6000)
    head = nn.Linear(features.shape[1], classes).to(train.device)
    optimizer = torch.optim.SGD(head.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0005)
    for epoch in range(epochs):
        if epoch == epochs // 2:
            for group in optimizer.param_groups:
                group['lr'] *= 0.1
        for indices in torch.randperm(len(features), device=train.device).split(256):
            loss = F.cross_entropy(head(features[indices]), labels[indices])
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
    settings = dict(vars(args), protocol=PROTOCOL, std_ddof=1,
                    normalization='target_dataset_repository_statistics', image_size=32,
                    no_pretrain_protocol='supervised_end_to_end', temperature=0.2,
                    ssl_aug_strategy=source.STRATEGY, ssl_aug_mode='S')
    source.write_json(output / 'config.json', settings)
    train, labels, test, test_labels, classes = load_target(args.target_cache, args.device, args.target)
    labels_cpu = labels.cpu().numpy()
    rows = []
    for seed in range(args.runs):
        started = time.perf_counter()
        indices = select_labels(labels_cpu, args.label_percentage, seed, args.label_policy)
        chosen = torch.tensor(indices, device=args.device)
        batches = math.ceil(len(indices) / 256)
        epochs = math.ceil(args.probe_updates / batches)
        if args.method == 'no_pretrain':
            network = source.get_network(args.model, 3, classes, (32, 32), seed=seed + 3000)
            source.train_supervised(network, train[chosen], labels[chosen], epochs, seed)
            accuracy = source.supervised_accuracy(network, test, test_labels)
        else:
            network = load_encoder(args, seed)
            accuracy = probe(network, train[chosen], labels[chosen], test, test_labels, classes, epochs, seed)
        row = dict(seed=seed, accuracy_percent=accuracy, labeled_indices=indices.tolist(),
                   labeled_count=len(indices), requested_label_percentage=args.label_percentage,
                   actual_label_percentage=100 * len(indices) / len(train),
                   labeled_classes=len(np.unique(labels_cpu[indices])), total_classes=classes,
                   probe_epochs=epochs, probe_updates=epochs * batches,
                   seconds=time.perf_counter() - started)
        source.write_json(output / f'seed_{seed:03d}.json', row)
        rows.append(row)
        scores = [value['accuracy_percent'] for value in rows]
        summary = dict(config=settings, complete=len(rows) == args.runs,
                       completed_runs=len(rows), expected_runs=args.runs,
                       mean_percent=statistics.mean(scores),
                       std_percent=statistics.stdev(scores) if len(scores) > 1 else None,
                       seeds=list(range(len(rows))), accuracies_percent=scores)
        source.write_json(output / 'summary.json', summary)
        print(f'RUN seed={seed} target={args.target} labels={len(indices)} classes={row["labeled_classes"]}/{classes} '
              f'epochs={epochs} accuracy={accuracy:.4f}% seconds={row["seconds"]:.1f}', flush=True)
        del network
        gc.collect()
    print(f'COMPLETE mean={summary["mean_percent"]:.4f} std={summary["std_percent"]} n={len(rows)}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['preflight', 'prepare-target', 'prepare-selections', 'pretrain', 'evaluate'])
    parser.add_argument('--data-path', default=os.path.join(os.environ.get('SCRATCH', '.'), 'data'))
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--output')
    parser.add_argument('--targets', default=','.join(SPECS))
    parser.add_argument('--target', choices=list(SPECS))
    parser.add_argument('--target-cache')
    parser.add_argument('--source-batch')
    parser.add_argument('--selection-dir')
    parser.add_argument('--encoder-dir')
    parser.add_argument('--model', choices=['ConvNet', 'VGG11', 'ResNet18'])
    parser.add_argument('--method', choices=['random', 'kmeans', 'full', 'no_pretrain'])
    parser.add_argument('--ssl-method', choices=['none', 'simclr', 'barlowtwins'])
    parser.add_argument('--subset-percentage', type=int, choices=[0, 1, 2, 5, 100], default=1)
    parser.add_argument('--label-percentage', type=int, choices=[1, 5], default=1)
    parser.add_argument('--label-policy', choices=['exact', 'at_least_one_per_class'], default='exact')
    parser.add_argument('--probe-updates', type=int, default=400)
    parser.add_argument('--ssl-epochs', type=int)
    parser.add_argument('--runs', type=int, default=15)
    args = parser.parse_args()
    args.ssl_epochs = args.ssl_epochs if args.ssl_epochs is not None else source.SSL_EPOCHS.get(args.subset_percentage, 0)
    if args.runs < 1 or args.probe_updates < 1:
        parser.error('runs and probe-updates must be positive')
    if args.command != 'preflight' and args.output is None:
        parser.error('--output is required')
    if args.command in ('prepare-target', 'evaluate') and args.target is None:
        parser.error('--target is required')
    if args.command in ('pretrain', 'evaluate'):
        if not args.model or not args.method:
            parser.error('--model and --method are required')
        if args.method == 'no_pretrain':
            if args.command != 'evaluate' or args.ssl_method != 'none' or args.subset_percentage != 0:
                parser.error('no_pretrain requires evaluate, ssl-method none, and subset-percentage 0')
        elif args.ssl_method not in ('simclr', 'barlowtwins') or args.ssl_epochs < 1:
            parser.error('pretrained methods need an SSL method and positive epochs')
        if args.method == 'full' and args.subset_percentage != 100:
            parser.error('full requires subset-percentage 100')
        if args.method in ('random', 'kmeans') and args.subset_percentage not in (1, 2, 5):
            parser.error('subset methods require 1, 2, or 5 percent')
        if args.command == 'pretrain' and args.method == 'kmeans' and not args.selection_dir:
            parser.error('K-means pretraining needs --selection-dir')
        if args.command == 'evaluate' and (not args.target_cache or (args.method != 'no_pretrain' and not args.encoder_dir)):
            parser.error('evaluate needs --target-cache and pretrained methods need --encoder-dir')
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', '1')))
    if args.device == 'cuda' and args.command in ('pretrain', 'evaluate', 'prepare-selections') and not torch.cuda.is_available():
        raise RuntimeError('GPU allocation is required')
    source.seed_all(0)
    if args.command == 'prepare-target':
        source.write_json(Path(args.output).with_suffix('.json'),
                          prepare_target(args.data_path, args.target, args.output))
    else:
        {'preflight': preflight, 'prepare-selections': prepare_selections,
         'pretrain': pretrain, 'evaluate': evaluate}[args.command](args)


if __name__ == '__main__':
    main()
