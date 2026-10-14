"""CIFAR-10 baselines; no pretraining uses a frozen random encoder and linear probe."""

import argparse
import gc
import json
import os
from pathlib import Path
import random
import statistics
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.datasets import CIFAR10

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'OT-SSL-DD'))
from utils import DiffAugment, ParamDiffAug, get_network


STRATEGY = 'color_crop_cutout_flip_scale_rotate'
SSL_EPOCHS = {1: 1200, 2: 800, 5: 500, 100: 300}
PROBE_EPOCHS = {1: 200, 5: 100}
PROTOCOL = 'cifar10_baselines_v3_random_labels_barlow2x'


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def load_data(root, device):
    """Load local CIFAR once; no concurrent download/extraction across jobs."""
    mean = torch.tensor([0.4914, 0.4822, 0.4465], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.2023, 0.1994, 0.2010], device=device).view(1, 3, 1, 1)
    output = []
    for training in (True, False):
        dataset = CIFAR10(root, train=training, download=False)
        images = torch.from_numpy(dataset.data).permute(0, 3, 1, 2).to(device, dtype=torch.float32).div_(255)
        images.sub_(mean).div_(std)
        labels = torch.tensor(dataset.targets, device=device, dtype=torch.long)
        output.extend((images.contiguous(), labels))
    return output


def labeled_indices(labels, percentage, seed):
    """Random nested label subsets, paired across methods for a run seed."""
    labels = np.asarray(labels)
    generator = np.random.default_rng(seed + 5000)
    count = max(1, int(len(labels) * percentage / 100))
    return generator.permutation(len(labels))[:count].astype(np.int64, copy=False)


@torch.no_grad()
def extract_features(network, images, batch_size=256):
    network.eval()
    return torch.cat([network.embed(batch) for batch in images.split(batch_size)])


@torch.no_grad()
def kmeans_indices(features, count, seed, iterations=50, chunk_size=1024):
    """Lloyd K-means with bounded distance matrices and distinct real exemplars."""
    generator = torch.Generator(device=features.device).manual_seed(seed + 1000)
    centers = features[torch.randperm(len(features), generator=generator, device=features.device)[:count]].clone()
    feature_norm = features.square().sum(1)
    for _ in range(iterations):
        sums = torch.zeros_like(centers)
        counts = torch.zeros(count, device=features.device)
        center_norm = centers.square().sum(1)
        for start in range(0, len(features), chunk_size):
            batch = features[start:start + chunk_size]
            distances = feature_norm[start:start + chunk_size, None] + center_norm[None] - 2 * batch @ centers.T
            assignments = distances.argmin(1)
            sums.index_add_(0, assignments, batch)
            counts.add_(torch.bincount(assignments, minlength=count))
        updated = sums / counts.clamp_min(1)[:, None]
        updated[counts == 0] = centers[counts == 0]
        converged = torch.allclose(updated, centers, atol=1e-4)
        centers = updated
        if converged:
            break
    # Select nearest real images without replacement, resolving shared nearest
    # neighbors in center order. This keeps the requested subset size exact.
    chosen = []
    used = np.zeros(len(features), dtype=bool)
    for batch in centers.split(64):
        distances = (batch.square().sum(1)[:, None] + feature_norm[None] - 2 * batch @ features.T).cpu().numpy()
        for row in distances:
            row[used] = np.inf
            index = int(row.argmin())
            chosen.append(index)
            used[index] = True
    return chosen


def augment(images, param):
    return DiffAugment(images.clone(), STRATEGY, seed=-1, param=param)


def ssl_loss(z1, z2, method):
    if method == 'simclr':
        z = torch.cat((F.normalize(z1, dim=1), F.normalize(z2, dim=1)))
        logits = z @ z.T / 0.2
        logits.fill_diagonal_(-1e9)
        targets = (torch.arange(len(z), device=z.device) + len(z1)) % len(z)
        return F.cross_entropy(logits, targets)
    if method != 'barlowtwins':
        raise ValueError(method)
    z1 = (z1 - z1.mean(0)) / (z1.std(0, unbiased=False) + 1e-5)
    z2 = (z2 - z2.mean(0)) / (z2.std(0, unbiased=False) + 1e-5)
    correlation = z1.T @ z2 / len(z1)
    diagonal = correlation.diagonal()
    return (diagonal - 1).square().sum() + 0.005 * (correlation - torch.diag(diagonal)).square().sum()


def train_ssl(network, images, method, epochs, seed, max_steps=None, log_every=None):
    seed_all(seed + 4000)
    dimension = network.embed(images[:1]).shape[1]
    projector = nn.Sequential(nn.Linear(dimension, 128), nn.ReLU(), nn.Linear(128, 128)).to(images.device)
    parameters = list(network.parameters()) + list(projector.parameters())
    optimizer = torch.optim.SGD(parameters, lr=0.01, momentum=0.9, weight_decay=0.0005)
    param = ParamDiffAug()
    param.aug_mode = 'S'
    steps = 0
    network.train()
    for epoch in range(epochs):
        if epoch == epochs // 2:
            for group in optimizer.param_groups:
                group['lr'] *= 0.1
        order = torch.randperm(len(images), device=images.device)
        for indices in order.split(256):
            if len(indices) < 2:
                continue
            batch = images[indices]
            loss = ssl_loss(projector(network.embed(augment(batch, param))),
                            projector(network.embed(augment(batch, param))), method)
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite SSL loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            steps += 1
            if max_steps is not None and steps >= max_steps:
                return steps
        log_interval = log_every if log_every else max(1, epochs // 4)
        if (epoch + 1) % log_interval == 0 or epoch + 1 == epochs:
            print(f'SSL epoch={epoch + 1}/{epochs} loss={loss.item():.6f}', flush=True)
    return steps


def train_supervised(network, images, labels, epochs, seed, max_steps=None):
    seed_all(seed + 6000)
    network.train()
    optimizer = torch.optim.SGD(network.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)
    param = ParamDiffAug()
    param.aug_mode = 'S'
    steps = 0
    for epoch in range(epochs):
        if epoch == epochs // 2:
            for group in optimizer.param_groups:
                group['lr'] *= 0.1
        for indices in torch.randperm(len(images), device=images.device).split(256):
            loss = F.cross_entropy(network(augment(images[indices], param)), labels[indices])
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite supervised loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            steps += 1
            if max_steps is not None and steps >= max_steps:
                return steps
    return steps


def linear_probe(network, train_images, train_labels, test_images, test_labels, epochs, seed):
    # Cache deterministic features once. Only the new linear layer is trained.
    network.eval().requires_grad_(False)
    train_features = extract_features(network, train_images)
    test_features = extract_features(network, test_images)
    seed_all(seed + 6000)
    linear = nn.Linear(train_features.shape[1], 10).to(train_images.device)
    optimizer = torch.optim.SGD(linear.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0005)
    for epoch in range(epochs):
        if epoch == epochs // 2:
            for group in optimizer.param_groups:
                group['lr'] *= 0.1
        for indices in torch.randperm(len(train_features), device=train_features.device).split(256):
            loss = F.cross_entropy(linear(train_features[indices]), train_labels[indices])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    with torch.no_grad():
        return (linear(test_features).argmax(1) == test_labels).float().mean().item() * 100


@torch.no_grad()
def supervised_accuracy(network, images, labels):
    network.eval()
    predictions = torch.cat([network(batch).argmax(1) for batch in images.split(256)])
    return (predictions == labels).float().mean().item() * 100


def prepare(args):
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=True)
    train, labels, test, test_labels = load_data(args.data_path, args.device)
    for seed in range(args.seed_start, args.seed_start + args.runs):
        path = destination / f'seed_{seed:03d}.json'
        if path.exists():
            cached = json.loads(path.read_text())
            if cached['protocol'] != PROTOCOL or cached['seed'] != seed:
                raise ValueError(f'Incompatible selection cache: {path}')
            continue
        start = time.perf_counter()
        network = get_network('ConvNet', 3, 10, (32, 32), seed=seed + 2000)
        features = extract_features(network, train)
        random_order = np.random.default_rng(seed + 1000).permutation(len(train))
        selections = {'protocol': PROTOCOL, 'seed': seed, 'feature_model': 'random_ConvNet',
                      'kmeans_iterations': 50, 'random': {}, 'kmeans': {}}
        for percentage in (1, 2, 5):
            count = len(train) * percentage // 100
            selections['random'][str(percentage)] = random_order[:count].tolist()
            selections['kmeans'][str(percentage)] = kmeans_indices(features, count, seed)
        write_json(path, selections)
        del network, features
        print(f'Prepared selection seed={seed} seconds={time.perf_counter() - start:.1f}', flush=True)


def benchmark(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    settings = vars(args).copy()
    settings.update(protocol=PROTOCOL, ssl_aug_strategy=STRATEGY, ssl_aug_mode='S',
                    label_selection='uniform_random_without_replacement',
                    temperature=0.2, batch_size=256, std_ddof=1,
                    no_pretrain_protocol='frozen_random_encoder',
                    kmeans_features='random_ConvNet')
    write_json(output / 'config.json', settings)
    train, labels, test, test_labels = load_data(args.data_path, args.device)
    labels_cpu = labels.cpu().numpy()
    records = []
    for seed in range(args.seed_start, args.seed_start + args.runs):
        started = time.perf_counter()
        labeled = labeled_indices(labels_cpu, args.label_percentage, seed)
        labeled = torch.tensor(labeled, device=args.device)
        network = get_network(args.model, 3, 10, (32, 32), seed=seed + 3000)
        ssl_steps = 0
        subset_indices = []
        if args.method == 'no_pretrain':
            accuracy = linear_probe(network, train[labeled], labels[labeled], test, test_labels, args.probe_epochs, seed)
        else:
            if args.method == 'full':
                source = train
            else:
                if args.method == 'random':
                    subset_indices = np.random.default_rng(seed + 1000).permutation(len(train))[:len(train) * args.subset_percentage // 100].tolist()
                else:
                    path = Path(args.selection_dir) / f'seed_{seed:03d}.json'
                    selection = json.loads(path.read_text())
                    if selection['protocol'] != PROTOCOL or selection['seed'] != seed:
                        raise ValueError(f'Incompatible selection cache: {path}')
                    subset_indices = selection['kmeans'][str(args.subset_percentage)]
                expected = len(train) * args.subset_percentage // 100
                if len(set(subset_indices)) != expected or min(subset_indices) < 0 or max(subset_indices) >= len(train):
                    raise ValueError('Invalid subset size or indices')
                source = train[torch.tensor(subset_indices, device=args.device)]
            ssl_steps = train_ssl(network, source, args.ssl_method, args.ssl_epochs, seed, log_every=args.ssl_log_every)
            accuracy = linear_probe(network, train[labeled], labels[labeled], test, test_labels, args.probe_epochs, seed)
            del source
        record = dict(seed=seed, accuracy_percent=accuracy, seconds=time.perf_counter() - started,
                      ssl_steps=ssl_steps, labeled_indices=labeled.cpu().tolist(), subset_indices=subset_indices)
        write_json(output / f'seed_{seed:03d}.json', record)
        records.append(record)
        scores = [row['accuracy_percent'] for row in records]
        summary = dict(config=settings, completed_runs=len(scores), expected_runs=args.runs,
                       complete=len(scores) == args.runs, mean_percent=statistics.mean(scores),
                       std_percent=statistics.stdev(scores) if len(scores) > 1 else None,
                       seeds=[row['seed'] for row in records], accuracies_percent=scores)
        write_json(output / 'summary.json', summary)
        print(f'RUN seed={seed} test_accuracy={accuracy:.4f}% seconds={record["seconds"]:.1f}', flush=True)
        del network
        gc.collect()
    print(f'COMPLETE n={len(scores)} mean={summary["mean_percent"]:.4f} std={summary["std_percent"]}', flush=True)


def timed(operation):
    torch.cuda.synchronize()
    started = time.perf_counter()
    value = operation()
    torch.cuda.synchronize()
    return time.perf_counter() - started, value


def profile(args):
    train, labels, test, test_labels = load_data(args.data_path, args.device)
    measurements = dict(gpu=torch.cuda.get_device_name(), models={}, kmeans={})
    for model in ('ConvNet', 'VGG11', 'ResNet18'):
        values = {}
        for method in ('simclr', 'barlowtwins'):
            network = get_network(model, 3, 10, (32, 32), seed=3000)
            # Warm kernels/allocator, then time actual training at the full batch size.
            train_ssl(network, train[:512], method, 20, 0, max_steps=4)
            seconds, steps = timed(lambda: train_ssl(network, train[:512], method, 20, 0, max_steps=20))
            values[f'{method}_step_seconds'] = seconds / steps
            del network
        network = get_network(model, 3, 10, (32, 32), seed=3000)
        seconds, steps = timed(lambda: train_supervised(network, train[:512], labels[:512], 20, 0, max_steps=20))
        values['supervised_step_seconds'] = seconds / steps
        seconds, _ = timed(lambda: linear_probe(network, train[:2500], labels[:2500], test, test_labels, 40, 0))
        values['probe_seconds'] = seconds
        measurements['models'][model] = values
        print(model, values, flush=True)
        del network
    network = get_network('ConvNet', 3, 10, (32, 32), seed=2000)
    seconds, features = timed(lambda: extract_features(network, train))
    measurements['selection_features_seconds'] = seconds
    for percentage in (1, 2, 5):
        seconds, indices = timed(lambda: kmeans_indices(features, len(train) * percentage // 100, 0))
        measurements['kmeans'][str(percentage)] = seconds
        assert len(set(indices)) == len(train) * percentage // 100
        print(f'Kmeans {percentage}%: {seconds:.2f}s', flush=True)
    write_json(args.output, measurements)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['run', 'prepare', 'profile'])
    parser.add_argument('--data-path', default=os.path.join(os.environ.get('SCRATCH', '.'), 'data'))
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--output', required=True)
    parser.add_argument('--selection-dir')
    parser.add_argument('--method', choices=['no_pretrain', 'random', 'kmeans', 'full'])
    parser.add_argument('--model', choices=['ConvNet', 'VGG11', 'ResNet18'])
    parser.add_argument('--ssl-method', choices=['none', 'simclr', 'barlowtwins'])
    parser.add_argument('--subset-percentage', type=int, choices=[0, 1, 2, 5, 100], default=1)
    parser.add_argument('--label-percentage', type=int, choices=[1, 5], default=1)
    parser.add_argument('--runs', type=int, default=15)
    parser.add_argument('--seed-start', type=int, default=0)
    parser.add_argument('--ssl-epochs', type=int)
    parser.add_argument('--ssl-log-every', type=int, default=0, help='Print SSL loss every N epochs; 0 prints four times per run')
    parser.add_argument('--probe-epochs', type=int)
    args = parser.parse_args()
    if args.runs < 1:
        parser.error('--runs must be positive')
    if args.ssl_log_every < 0:
        parser.error('--ssl-log-every must be nonnegative')
    args.ssl_epochs = args.ssl_epochs if args.ssl_epochs is not None else SSL_EPOCHS.get(args.subset_percentage, 0)
    args.probe_epochs = args.probe_epochs if args.probe_epochs is not None else PROBE_EPOCHS[args.label_percentage]
    if args.command == 'run':
        if args.model is None or args.method is None:
            parser.error('run requires --model and --method')
        if args.method != 'no_pretrain' and args.ssl_method not in ('simclr', 'barlowtwins'):
            parser.error('pretraining requires --ssl-method simclr or barlowtwins')
        if args.method == 'kmeans' and not args.selection_dir:
            parser.error('kmeans requires --selection-dir')
        if args.method in ('random', 'kmeans') and args.subset_percentage not in (1, 2, 5):
            parser.error('subset methods require --subset-percentage 1, 2, or 5')
        if args.method == 'full' and args.subset_percentage != 100:
            parser.error('full requires --subset-percentage 100')
        if args.method == 'no_pretrain' and (args.subset_percentage != 0 or args.ssl_method != 'none'):
            parser.error('no_pretrain requires --subset-percentage 0 and --ssl-method none')
        if args.probe_epochs < 1 or (args.method != 'no_pretrain' and args.ssl_epochs < 1):
            parser.error('training stages must have positive epoch counts')
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', '1')))
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('A GPU allocation is required')
    seed_all(0)
    {'run': benchmark, 'prepare': prepare, 'profile': profile}[args.command](args)


if __name__ == '__main__':
    main()
