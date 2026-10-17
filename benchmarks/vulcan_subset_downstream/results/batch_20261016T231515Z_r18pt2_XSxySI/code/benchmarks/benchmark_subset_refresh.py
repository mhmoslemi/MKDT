"""Checkpointed Random/K-means SSL baselines for CIFAR-10 and CIFAR-100."""

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
from downstream_data import SPECS, load_target, select_labels


PROTOCOL = 'subset_refresh_v1_pt0p2_long_probes'
SOURCES = ('CIFAR10', 'CIFAR100')
MODELS = ('ConvNet', 'VGG11', 'ResNet18')
METHODS = ('random', 'kmeans')
SSL_METHODS = ('simclr', 'barlowtwins')
SUBSETS = (0.2, 1.0, 2.0, 5.0)


def percentage_key(value):
    value = float(value)
    return str(int(value)) if value.is_integer() else format(value, 'g')


def subset_count(total, percentage):
    return max(1, int(total * float(percentage) / 100))


def atomic_torch_save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def canonical_config(args):
    return {
        'protocol': PROTOCOL,
        'source_dataset': args.source,
        'source_classes': SPECS[args.source][0],
        'model': args.model,
        'method': args.method,
        'ssl_method': args.ssl_method,
        'subset_percentage': float(args.subset_percentage),
        'ssl_epochs': args.ssl_epochs,
        'temperature': 0.2,
        'batch_size': 256,
        'ssl_aug_strategy': shared.STRATEGY,
        'ssl_aug_mode': 'S',
    }


def ensure_config(path, value):
    """Create a shared config once, or verify the one another array task wrote."""
    path = Path(path)
    encoded = json.dumps(value, indent=2, allow_nan=False, sort_keys=True) + '\n'
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temporary.write_text(encoded)
    try:
        os.link(temporary, path)
    except FileExistsError:
        if json.loads(path.read_text()) != value:
            raise ValueError(f'Incompatible existing configuration: {path}')
    finally:
        temporary.unlink(missing_ok=True)


def validate_selection(selection, source, seed, total):
    expected_header = {
        'protocol': PROTOCOL,
        'source_dataset': source,
        'seed': seed,
        'train_count': total,
        'feature_model': 'random_ConvNet',
        'kmeans_iterations': 50,
    }
    for key, expected in expected_header.items():
        if selection.get(key) != expected:
            raise ValueError(
                f'Incompatible selection field {key}={selection.get(key)!r}; '
                f'expected {expected!r}')
    for percentage in SUBSETS:
        key = percentage_key(percentage)
        expected = subset_count(total, percentage)
        for method in METHODS:
            indices = selection.get(method, {}).get(key)
            if (not isinstance(indices, list) or len(indices) != expected or
                    len(set(indices)) != expected or min(indices) < 0 or
                    max(indices) >= total):
                raise ValueError(
                    f'Invalid {source} {method} selection for seed={seed}, PT={key}%')


def prepare(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    train, labels, test, test_labels, _ = load_target(
        args.source_cache, args.device, args.source)
    del labels, test, test_labels
    total = len(train)
    for seed in range(args.seed_start, args.seed_start + args.runs):
        destination = output / f'seed_{seed:03d}.json'
        if destination.is_file():
            selection = json.loads(destination.read_text())
            validate_selection(selection, args.source, seed, total)
            print(f'SKIP selection source={args.source} seed={seed}: valid cache exists',
                  flush=True)
            continue
        started = time.perf_counter()
        network = shared.get_network(
            'ConvNet', 3, SPECS[args.source][0], (32, 32), seed=seed + 2000)
        features = shared.extract_features(network, train)
        random_order = np.random.default_rng(seed + 1000).permutation(total)
        selection = {
            'protocol': PROTOCOL,
            'source_dataset': args.source,
            'seed': seed,
            'train_count': total,
            'feature_model': 'random_ConvNet',
            'kmeans_iterations': 50,
            'random': {},
            'kmeans': {},
        }
        for percentage in SUBSETS:
            key = percentage_key(percentage)
            count = subset_count(total, percentage)
            selection['random'][key] = random_order[:count].tolist()
            selection['kmeans'][key] = shared.kmeans_indices(
                features, count, seed, iterations=50)
        validate_selection(selection, args.source, seed, total)
        shared.write_json(destination, selection)
        print(f'PREPARE source={args.source} seed={seed} '
              f'seconds={time.perf_counter() - started:.1f}', flush=True)
        del network, features
        gc.collect()


def load_selection(args, seed, total):
    if args.method == 'random':
        count = subset_count(total, args.subset_percentage)
        return np.random.default_rng(seed + 1000).permutation(total)[:count].tolist()
    path = Path(args.selection_dir) / f'seed_{seed:03d}.json'
    selection = json.loads(path.read_text())
    validate_selection(selection, args.source, seed, total)
    return selection['kmeans'][percentage_key(args.subset_percentage)]


def checkpoint_expected(args, seed):
    return dict(canonical_config(args), seed=seed)


def validate_checkpoint(path, args, seed):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    for key, expected in checkpoint_expected(args, seed).items():
        if checkpoint.get(key) != expected:
            raise ValueError(
                f'{path}: unexpected {key}={checkpoint.get(key)!r}; '
                f'expected {expected!r}')
    if not isinstance(checkpoint.get('state_dict'), dict):
        raise ValueError(f'{path}: missing encoder state_dict')
    expected_count = subset_count(checkpoint['source_train_count'],
                                  args.subset_percentage)
    indices = checkpoint.get('source_indices')
    if (not isinstance(indices, list) or len(indices) != expected_count or
            len(set(indices)) != expected_count or min(indices) < 0 or
            max(indices) >= checkpoint['source_train_count']):
        raise ValueError(f'{path}: invalid source indices')
    return checkpoint


def pretrain(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    ensure_config(output / 'config.json', canonical_config(args))
    train, labels, test, test_labels, classes = load_target(
        args.source_cache, args.device, args.source)
    del labels, test, test_labels
    for seed in range(args.seed_start, args.seed_start + args.runs):
        destination = output / f'seed_{seed:03d}.pt'
        if destination.is_file():
            validate_checkpoint(destination, args, seed)
            print(f'SKIP pretrain source={args.source} model={args.model} '
                  f'method={args.method} ssl={args.ssl_method} '
                  f'PT={percentage_key(args.subset_percentage)}% seed={seed}: '
                  'valid checkpoint exists', flush=True)
            continue
        started = time.perf_counter()
        indices = load_selection(args, seed, len(train))
        source_images = train[torch.tensor(indices, device=args.device)]
        network = shared.get_network(
            args.model, 3, classes, (32, 32), seed=seed + 3000)
        steps = shared.train_ssl(
            network, source_images, args.ssl_method, args.ssl_epochs, seed,
            log_every=args.ssl_log_every)
        checkpoint = dict(
            checkpoint_expected(args, seed),
            source_train_count=len(train),
            source_indices=indices,
            ssl_steps=steps,
            state_dict={key: value.detach().cpu()
                        for key, value in network.state_dict().items()},
        )
        atomic_torch_save(destination, checkpoint)
        validate_checkpoint(destination, args, seed)
        print(f'PRETRAIN source={args.source} model={args.model} '
              f'method={args.method} ssl={args.ssl_method} '
              f'PT={percentage_key(args.subset_percentage)}% seed={seed} '
              f'steps={steps} seconds={time.perf_counter() - started:.1f}',
              flush=True)
        del network, checkpoint, source_images
        gc.collect()


def checkpoint_path(root, seed):
    path = Path(root) / f'seed_{seed:03d}.pt'
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def load_encoder(args, seed):
    path = checkpoint_path(args.encoder_dir, seed)
    checkpoint = validate_checkpoint(path, args, seed)
    network = shared.get_network(
        args.model, 3, checkpoint['source_classes'], (32, 32),
        seed=seed + 3000)
    network.load_state_dict(checkpoint['state_dict'], strict=True)
    return network.eval().requires_grad_(False)


def evaluate(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    settings = dict(
        canonical_config(args),
        target_dataset=args.target,
        label_percentage=args.label_percentage,
        probe_epochs=args.probe_epochs,
        runs=args.runs,
        probe_protocol='frozen_encoder',
        label_selection='uniform_random_without_replacement',
        normalization='target_dataset_repository_statistics',
        image_size=32,
        std_ddof=1,
    )
    shared.write_json(output / 'config.json', settings)
    train, labels, test, test_labels, classes = load_target(
        args.target_cache, args.device, args.target)
    labels_cpu = labels.cpu().numpy()
    rows = []
    for seed in range(args.runs):
        started = time.perf_counter()
        indices = select_labels(labels_cpu, args.label_percentage, seed, 'random')
        chosen = torch.tensor(indices, device=args.device)
        network = load_encoder(args, seed)
        accuracy = main_dataset.linear_probe(
            network, train[chosen], labels[chosen], test, test_labels, classes,
            args.probe_epochs, seed)
        row = {
            'seed': seed,
            'encoder_seed': seed,
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
        print(f'EVAL source={args.source} target={args.target} '
              f'model={args.model} method={args.method} ssl={args.ssl_method} '
              f'PT={percentage_key(args.subset_percentage)}% '
              f'labels={args.label_percentage}% seed={seed} '
              f'accuracy={accuracy:.4f}% seconds={row["seconds"]:.1f}',
              flush=True)
        del network
        gc.collect()
    print(f'COMPLETE target={args.target} labels={args.label_percentage}% '
          f'n={len(rows)} mean={summary["mean_percent"]:.4f} '
          f'std={summary["std_percent"]}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'pretrain', 'evaluate'))
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--source', choices=SOURCES, required=True)
    parser.add_argument('--source-cache')
    parser.add_argument('--output', required=True)
    parser.add_argument('--selection-dir')
    parser.add_argument('--encoder-dir')
    parser.add_argument('--target', choices=tuple(SPECS))
    parser.add_argument('--target-cache')
    parser.add_argument('--model', choices=MODELS)
    parser.add_argument('--method', choices=METHODS)
    parser.add_argument('--ssl-method', choices=SSL_METHODS)
    parser.add_argument('--subset-percentage', type=float, choices=SUBSETS)
    parser.add_argument('--ssl-epochs', type=int)
    parser.add_argument('--probe-epochs', type=int, default=0)
    parser.add_argument('--label-percentage', type=int, choices=(1, 5), default=1)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--seed-start', type=int, default=0)
    parser.add_argument('--ssl-log-every', type=int, default=0)
    args = parser.parse_args()

    if args.runs < 1 or args.seed_start < 0 or args.ssl_log_every < 0:
        parser.error('run counts must be positive and seeds/log intervals nonnegative')
    if args.command in ('prepare', 'pretrain') and not args.source_cache:
        parser.error('prepare/pretrain require --source-cache')
    if args.command in ('pretrain', 'evaluate'):
        if not all((args.model, args.method, args.ssl_method,
                    args.subset_percentage is not None, args.ssl_epochs)):
            parser.error('pretrain/evaluate require model, method, SSL, PT, and SSL epochs')
        if args.method == 'kmeans' and args.command == 'pretrain' and not args.selection_dir:
            parser.error('K-means pretraining requires --selection-dir')
    if args.command == 'evaluate':
        if not all((args.encoder_dir, args.target, args.target_cache)):
            parser.error('evaluate requires encoder dir, target, and target cache')
        if args.probe_epochs < 1 or args.seed_start != 0:
            parser.error('evaluate requires positive probe epochs and seed-start 0')

    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', '1')))
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('A GPU allocation is required')
    shared.seed_all(0)
    {'prepare': prepare, 'pretrain': pretrain, 'evaluate': evaluate}[args.command](args)


if __name__ == '__main__':
    main()
