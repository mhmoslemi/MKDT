"""Frozen-checkpoint diagnostic, NOT a new SSL run or a table result.

Compare the historical/current probe budgets with identical labels, initial
heads and optimizers. Report feature statistics and training fit, without
selecting hyperparameters or epochs using test accuracy.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import benchmark_cifar10 as shared
import benchmark_full_refresh as full
from downstream_data import load_target, select_labels


def state_digest(network):
    digest = hashlib.sha256()
    for name, value in network.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def feature_stats(features):
    if not torch.isfinite(features).all():
        raise RuntimeError('Non-finite encoder features')
    centered = features.double() - features.double().mean(0)
    variances = centered.square().mean(0)
    eigenvalues = torch.linalg.eigvalsh(centered.T @ centered / len(features)).clamp_min(0)
    total = eigenvalues.sum()
    probabilities = eigenvalues / total if total > 0 else eigenvalues
    positive = probabilities[probabilities > 0]
    effective_rank = (-(positive * positive.log()).sum()).exp().item() if total > 0 else 0.0
    return dict(samples=len(features), dimensions=features.shape[1],
                mean_norm=features.norm(dim=1).mean().item(),
                mean_variance=variances.mean().item(),
                dimensions_with_variance_below_1e_8=(variances < 1e-8).sum().item(),
                effective_rank=effective_rank)


def probe(train_features, labels, test_features, test_labels, classes, epochs, seed):
    # Keep this optimizer/schedule identical to benchmark_dataset.linear_probe.
    shared.seed_all(seed + 6000)
    head = nn.Linear(train_features.shape[1], classes).to(train_features.device)
    optimizer = torch.optim.SGD(head.parameters(), lr=0.1, momentum=0.9,
                                weight_decay=0.0005)
    for epoch in range(epochs):
        if epoch == epochs // 2:
            for group in optimizer.param_groups:
                group['lr'] *= 0.1
        for indices in torch.randperm(len(train_features), device=train_features.device).split(256):
            loss = F.cross_entropy(head(train_features[indices]), labels[indices])
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite probe loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    with torch.no_grad():
        train_logits = head(train_features)
        test_logits = head(test_features)
        return dict(epochs=epochs,
                    train_accuracy_percent=(train_logits.argmax(1) == labels).float().mean().item() * 100,
                    test_accuracy_percent=(test_logits.argmax(1) == test_labels).float().mean().item() * 100,
                    train_cross_entropy=F.cross_entropy(train_logits, labels).item())


def self_test():
    shared.seed_all(12)
    train = torch.randn(300, 16, device='cuda')
    labels = torch.arange(300, device='cuda') % 5
    test = torch.randn(60, 16, device='cuda')
    test_labels = torch.arange(60, device='cuda') % 5
    for epochs in (4, 7):
        actual = probe(train, labels, test, test_labels, 5, epochs, 2)
        expected = full.linear_probe_features(train, labels, test, test_labels, 5, epochs, 2)
        assert actual['test_accuracy_percent'] == expected, (actual, expected)
    assert feature_stats(torch.ones(64, 16, device='cuda'))['effective_rank'] == 0
    print('SELF_TEST_PASSED: probe matches the existing implementation', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind', choices=('full', 'random'), required=True)
    parser.add_argument('--encoder-dir', required=True)
    parser.add_argument('--target-cache', required=True)
    parser.add_argument('--epoch-config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--runs', type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.runs <= 10:
        parser.error('runs must be between 1 and 10')
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise RuntimeError('Run this diagnostic inside a Slurm GPU allocation')
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', '1')))
    self_test()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    config = json.loads(Path(args.epoch_config).read_text())
    budgets = config['evaluation_by_dataset']['CIFAR100']['linear_probe_epochs_by_label_percentage']
    settings = dict(vars(args), diagnostic_only=True, ssl_training=False,
                    probe_budgets=budgets, test_used_for_selection=False)
    shared.write_json(output / 'config.json', settings)
    train, labels, test, test_labels, classes = load_target(args.target_cache, 'cuda', 'CIFAR100')
    labels_cpu = labels.cpu().numpy()
    stat_indices = torch.tensor(np.random.default_rng(9821).permutation(len(train))[:4096], device='cuda')
    expected = dict(source_dataset='CIFAR100', source_classes=100, model='ResNet18',
                    method=args.kind, ssl_method='barlowtwins', temperature=0.2,
                    ssl_aug_strategy=shared.STRATEGY, ssl_aug_mode='S')
    expected.update(
        protocol='full_data_main_simclr1000_barlow1500_v1' if args.kind == 'full' else 'subset_refresh_v1_pt0p2_long_probes',
        ssl_epochs=1500 if args.kind == 'full' else 20000,
        subset_percentage=100 if args.kind == 'full' else 0.2)
    rows = []
    for seed in range(args.runs):
        path = full.checkpoint_path(args.encoder_dir, seed)
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        for key, value in dict(expected, seed=seed).items():
            if checkpoint.get(key) != value:
                raise ValueError(f'{path}: {key}={checkpoint.get(key)!r}, expected {value!r}')
        network = shared.get_network('ResNet18', 3, classes, (32, 32), seed=seed + 3000)
        network.load_state_dict(checkpoint['state_dict'], strict=True)
        network.eval().requires_grad_(False)
        before = state_digest(network)
        test_features = shared.extract_features(network, test)
        stat_features = shared.extract_features(network, train[stat_indices])
        row = dict(seed=seed, checkpoint=str(path), encoder_sha256=before,
                   ssl_steps=checkpoint.get('ssl_steps'), feature_statistics=feature_stats(stat_features), probes=[])
        del stat_features, checkpoint
        for percentage, old_epochs in ((1, 500), (5, 300)):
            indices = select_labels(labels_cpu, percentage, seed, 'random')
            chosen = torch.tensor(indices, device='cuda')
            features = shared.extract_features(network, train[chosen])
            label_row = dict(label_percentage=percentage, labeled_indices=indices.tolist(),
                             labeled_class_count=len(np.unique(labels_cpu[indices])), results=[])
            for epochs in sorted({old_epochs, budgets[str(percentage)]}):
                result = probe(features, labels[chosen], test_features, test_labels, classes, epochs, seed)
                label_row['results'].append(result)
                print(json.dumps(dict(kind=args.kind, seed=seed, labels=percentage, **result)), flush=True)
            row['probes'].append(label_row)
            del features
        if state_digest(network) != before:
            raise RuntimeError('Diagnostic changed the frozen encoder')
        row['encoder_unchanged'] = True
        shared.write_json(output / f'seed_{seed:03d}.json', row)
        rows.append(row)
        del network, test_features
    shared.write_json(output / 'diagnostic.json', dict(config=settings, complete=True, rows=rows))


if __name__ == '__main__':
    main()
