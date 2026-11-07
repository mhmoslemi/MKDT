"""Label-free SAS and ProbCover on fixed, normalized teacher features."""
from pathlib import Path
import sys

import numpy as np
from scipy import sparse
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent / 'upstream'))
from sas.submodular_maximization import lazy_greedy
from sas.subset_dataset import SubsetSelectionObjective


@torch.no_grad()
def cluster_features(features, count, seed=0, iterations=1000, chunk=4096):
    """Euclidean Lloyd clustering, as in SAS's label-free kmeans_approx."""
    generator = torch.Generator(device=features.device).manual_seed(seed)
    centers = features[torch.randperm(len(features), generator=generator, device=features.device)[:count]].clone()
    previous = None
    for iteration in range(iterations):
        assignments = torch.cat([torch.cdist(features[start:start + chunk], centers).argmin(1)
                                 for start in range(0, len(features), chunk)])
        if previous is not None and torch.equal(previous, assignments):
            break
        sums = torch.zeros_like(centers)
        sums.index_add_(0, assignments, features)
        counts = torch.bincount(assignments, minlength=count)
        occupied = counts > 0
        centers[occupied] = sums[occupied] / counts[occupied, None]
        empty = (~occupied).nonzero().flatten()
        if len(empty):
            distances = (features - centers[assignments]).square().sum(1)
            centers[empty] = features[distances.argsort(descending=True)[:len(empty)]]
        previous = assignments.clone()
    return assignments.cpu().numpy(), iteration + 1


def quotas(groups, budget):
    """Largest-remainder rounding preserves SAS's proportional group budgets."""
    counts = np.bincount(groups)
    exact = counts * (budget / len(groups))
    result = np.floor(exact).astype(np.int64)
    order = np.argsort(-(exact - result), kind='stable')
    result[order[:budget - int(result.sum())]] += 1
    assert result.sum() == budget and np.all(result <= counts)
    return result


@torch.no_grad()
def sas_subsets(features, groups, budgets):
    allocations = [quotas(groups, budget) for budget in budgets]
    subsets = [[] for _ in budgets]
    for group in np.unique(groups):
        indices = np.flatnonzero(groups == group)
        needed = max(allocation[group] for allocation in allocations)
        if not needed:
            continue
        vectors = F.normalize(features[torch.as_tensor(indices, device=features.device)], dim=1)
        similarity = (vectors @ vectors.T).cpu().double().numpy()
        objective = SubsetSelectionObjective(similarity, threshold=0.0)
        # A prefix is identical to the official implementation's full ordering.
        chosen = lazy_greedy(objective, range(len(indices)), int(needed), verbose=False)
        for subset, allocation in zip(subsets, allocations):
            subset.extend(indices[chosen[:allocation[group]]].tolist())
        print(f'SAS cluster {group}: {len(indices)} images, ranked {needed}', flush=True)
    return subsets


@torch.no_grad()
def estimate_radius(features, groups, alpha=0.95, chunk=256):
    """Largest strict radius with at least alpha cluster-pure balls."""
    labels = torch.as_tensor(groups, device=features.device)
    if len(np.unique(groups)) < 2:
        raise ValueError('ProbCover radius requires at least two occupied clusters')
    nearest_cross = []
    for start in range(0, len(features), chunk):
        distances = torch.cdist(features[start:start + chunk], features)
        distances.masked_fill_(labels[start:start + chunk, None] == labels[None, :], float('inf'))
        nearest_cross.append(distances.amin(1).cpu())
    nearest_cross = torch.cat(nearest_cross).numpy()
    radius = float(np.sort(nearest_cross)[int(np.floor((1 - alpha) * len(features)))])
    purity = float(np.mean(nearest_cross >= radius))
    if not np.isfinite(radius) or radius <= 0 or purity < alpha - 1e-12:
        raise ValueError(f'Invalid unsupervised ProbCover radius: {radius}, purity={purity}')
    return radius, purity


@torch.no_grad()
def radius_graph(features, radius, chunk=256):
    """Exact sparse strict-radius graph, including the final partial batch."""
    rows, columns = [], []
    for start in range(0, len(features), chunk):
        distances = torch.cdist(features[start:start + chunk], features)
        x, y = (distances < radius).nonzero(as_tuple=True)
        rows.append((x.cpu().numpy() + start).astype(np.int32))
        columns.append(y.cpu().numpy().astype(np.int32))
    rows, columns = np.concatenate(rows), np.concatenate(columns)
    graph = sparse.csr_matrix((np.ones(len(rows), dtype=np.uint8), (rows, columns)),
                              shape=(len(features), len(features)))
    graph.setdiag(1)
    graph.sort_indices()
    print(f'ProbCover graph: {graph.shape[0]} vertices, {graph.nnz} edges', flush=True)
    return graph


def probcover_order(graph, budget):
    """Same greedy uncovered-neighbor rule as official ProbCover, empty lSet."""
    inverse = graph.tocsc()
    degrees = np.diff(graph.indptr).astype(np.int64)
    covered = np.zeros(graph.shape[0], dtype=bool)
    selected = []
    for step in range(budget):
        current = int(degrees.argmax())
        neighbors = graph.indices[graph.indptr[current]:graph.indptr[current + 1]]
        newly_covered = neighbors[~covered[neighbors]]
        covered[newly_covered] = True
        for vertex in newly_covered:
            parents = inverse.indices[inverse.indptr[vertex]:inverse.indptr[vertex + 1]]
            degrees[parents] -= 1
        degrees[current] = -graph.shape[0] - 1
        selected.append(current)
        if (step + 1) % 100 == 0 or step + 1 == budget:
            print(f'ProbCover selected {step + 1}/{budget}; coverage={covered.mean():.4f}', flush=True)
    assert len(selected) == len(set(selected)) == budget
    return selected
