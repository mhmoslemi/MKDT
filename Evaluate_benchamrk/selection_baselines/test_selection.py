"""Small reference checks plus an optional real-data SSL/probe smoke test."""
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import torch
from scipy import sparse

from selection import (cluster_features, quotas, sas_subsets, estimate_radius,
                       radius_graph, probcover_order, lazy_greedy, SubsetSelectionObjective)


def check_selection():
    torch.set_num_threads(1)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    generator = torch.Generator().manual_seed(7)
    features = torch.nn.functional.normalize(torch.randn(37, 8, generator=generator), dim=1).to(device)
    groups = np.arange(37) % 4
    budgets = [4, 13, 30]
    results = sas_subsets(features, groups, budgets)
    for budget, actual in zip(budgets, results):
        expected = []
        allocation = quotas(groups, budget)
        for group in np.unique(groups):
            indices = np.flatnonzero(groups == group)
            vectors = torch.nn.functional.normalize(features[torch.as_tensor(indices, device=device)], dim=1)
            matrix = (vectors @ vectors.T).cpu().double().numpy()
            ranking = lazy_greedy(SubsetSelectionObjective(matrix), range(len(indices)), len(indices))
            expected.extend(indices[ranking[:allocation[group]]].tolist())
        assert actual == expected and len(actual) == len(set(actual)) == budget
    for budget in range(1, 38):
        assert quotas(groups, budget).sum() == budget

    radius, purity = estimate_radius(features, groups, alpha=0.8, chunk=8)
    distances = torch.cdist(features, features).cpu().numpy()
    cross = np.where(groups[:, None] != groups[None, :], distances, np.inf).min(1)
    assert np.isclose(radius, np.sort(cross)[int(np.floor(0.2 * len(groups)))], atol=1e-5)
    assert purity >= 0.8
    radius *= 0.9
    graph = radius_graph(features, radius, chunk=8)
    assert graph.shape == (37, 37) and graph.diagonal().all()
    assert np.array_equal(graph.toarray().astype(bool), distances < radius)
    dense = graph.toarray().astype(bool)
    covered = np.zeros(len(groups), dtype=bool)
    selected = []
    for _ in range(len(groups)):
        degrees = dense[:, ~covered].sum(1)
        degrees[selected] = -1
        current = int(degrees.argmax())
        selected.append(current)
        covered |= dense[current]
    assert probcover_order(graph, len(groups)) == selected
    assert probcover_order(sparse.csr_matrix(np.ones((5, 5), dtype=np.uint8)), 5) == list(range(5))
    clustered, _ = cluster_features(features, 4, iterations=30, chunk=8)
    assert clustered.shape == (37,) and 0 <= clustered.min() <= clustered.max() < 4
    print('PASS: SAS official-prefix equivalence, exact budgets, radius purity, partial batches, greedy coverage and no duplicates.')


def check_integration():
    from worker import BENCH, HERE, protocol
    from common import digest_json, load_encoder
    from data import RawDataset
    from evaluate import linear_probe
    from train_ssl import train
    assert torch.cuda.is_available()
    base = RawDataset('/scratch/mmoslem3/data', 'CIFAR10', True)
    test = torch.utils.data.Subset(RawDataset('/scratch/mmoslem3/data', 'CIFAR10', False), range(64))
    test.fingerprint = digest_json(dict(smoke_test_indices=list(range(64))))
    cfg = protocol()
    cfg['ssl'].update(subset_epochs=2, checkpoint_every_epochs=1)
    cfg['evaluation']['epochs'] = 2
    identity = dict(signature='selection_baselines_smoke_only', seed=0, selection_method='sas')
    (HERE / 'checks').mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=HERE / 'checks', prefix='smoke_') as temporary:
        output = Path(temporary)
        selected = torch.utils.data.Subset(base, range(100))
        checkpoint, saved = train(selected, cfg, 'CIFAR10', 'ResNet18', 'simclr', 0, output, identity, 0, 'cuda')
        assert saved['epochs'] == 2
        result = linear_probe(checkpoint, identity, cfg, 'CIFAR10', 'CIFAR10', base, test,
                              1, 0, output / 'probe.json', 0, 'cuda')
        assert result['complete'] and result['encoder_unchanged'] and result['test_count'] == 64
        _, _, reloaded = load_encoder(checkpoint, identity, device='cpu')
        assert saved['encoder_sha256'] == reloaded['encoder_sha256'] == result['encoder_sha256']
    print('PASS: fresh SSL training, checkpoint reload and frozen linear probing using the existing benchmark.')


if __name__ == '__main__':
    assert os.environ.get('SLURM_JOB_ID'), 'Run checks in a Slurm allocation'
    check_selection()
    check_integration()
