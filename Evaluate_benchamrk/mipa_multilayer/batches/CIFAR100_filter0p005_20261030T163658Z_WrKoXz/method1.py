"""Breakthrough 1: Bipartite View-Pair Mutual Information with Intra-Pair Directional Shift Alignment.
Prevents stationary texture collapse by augmenting S into paired views (u_syn, v_syn) and aligning both
the bipartite cluster mutual information and the intra-pair view transition vectors (u - v)."""

import argparse
import importlib.util
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torchvision import transforms
from torchvision.utils import save_image
from dm_dsa import make_diff_augmenter as dm_augmenter
from dd_teachers import load_teacher

from eval import ROOT, SYNTHETIC_NORMALIZATION, evaluate_synset_SSL, reference_modules
from utils import DiffAugment, ParamDiffAug, get_network, get_time


def nonnegative_float(value):
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise argparse.ArgumentTypeError('must be finite and nonnegative')
    return value


def nonnegative_int(value):
    value = int(value)
    if value < 0:
        raise argparse.ArgumentTypeError('must be nonnegative')
    return value


def set_random_seed(seed):
    """Seed initialization/evaluation while retaining explicit objective RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _images_from_batch(batch):
    return batch[0] if isinstance(batch, (tuple, list)) else batch


def digit_training_data(args):
    bench = ROOT / 'Evaluate_benchamrk'
    paths = {'datasets': bench / 'digits_simclr/datasets.py'}
    if args.dataset == 'FashionMNIST':
        paths['_method1_fashion'] = bench / 'fashionmnist/dataset.py'
    common = reference_modules()['common']
    previous = {name: sys.modules.get(name) for name in ('common', *paths)}
    try:
        sys.modules['common'] = common
        for name, path in paths.items():
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
        dataset_class = module.DigitDataset
    finally:
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
    raw = dataset_class(args.data_path, args.dataset, True)
    folder = 'fashionmnist/full' if args.dataset == 'FashionMNIST' else 'digits_simclr'
    prepared = json.loads((bench / folder / 'prepared.json').read_text())
    assert raw.fingerprint == prepared['datasets'][args.dataset]['train_fingerprint'], 'Digit training data differs from the saved teacher data'
    return raw, prepared['protocol']['normalization'][args.dataset]


def frozen_embedder(network, feature_layers='last', similarity='cosine'):
    """Return frozen final, multilayer cosine, or packed D3S Gaussian features."""
    module = network.module if isinstance(network, nn.DataParallel) else network
    network.eval()
    for parameter in network.parameters():
        parameter.requires_grad_(False)
    encoder = module.encoder if hasattr(module, 'encoder') else module.embed
    gaussian = similarity == 'd3s'
    if feature_layers == 'last' and not gaussian:
        return encoder
    backbone = module.network if hasattr(module, 'network') else module
    if feature_layers not in ('last', 'all') or not all(hasattr(backbone, name) for name in ('layer2', 'layer3', 'layer4')):
        raise ValueError('Intermediate features require a ResNet teacher with layer2/layer3/layer4')

    def embed_all(images):
        descriptors = []

        def moments(layer, inputs, features):
            spatial = features.flatten(2)
            mean = spatial.mean(dim=2)
            if gaussian:
                centered = spatial - mean.unsqueeze(2)
                covariance = centered @ centered.transpose(1, 2) / spatial.shape[2]
                descriptors.append(torch.cat((mean, covariance.flatten(1)), dim=1))
            else:
                descriptors.append(torch.cat((mean, spatial.std(dim=2, unbiased=False)), dim=1))

        # Run the original encoder so teacher normalization and final pooling stay unchanged.
        handles = [layer.register_forward_hook(moments) for layer in (backbone.layer2, backbone.layer3)]
        try:
            final = encoder(images)
        finally:
            for handle in handles:
                handle.remove()
        if gaussian:
            return torch.cat([*descriptors, final], dim=1)
        # Cosine on this concatenation equals the mean of the three normalized similarities.
        return torch.cat([F.normalize(value, dim=1) for value in (*descriptors, final)], dim=1) / math.sqrt(3)

    return embed_all


def unpack_d3s(features):
    """Unpack ResNet18 layer2/layer3 per-image Gaussian moments and final features."""
    if features.ndim != 2 or features.shape[1] != 128 + 128**2 + 256 + 256**2 + 512:
        raise ValueError('d3s needs ResNet18 full-covariance middle features and 512 final features')
    moments, position = [], 0
    for channels in (128, 256):
        mean = features[:, position:position + channels]
        position += channels
        covariance = features[:, position:position + channels**2].reshape(-1, channels, channels)
        position += channels**2
        moments.append((mean, covariance))
    return moments, features[:, position:]


def gaussian_kl(real_mean, real_covariance, syn_mean, syn_covariance, ridge):
    """Pairwise KL(N_syn || N_real), per channel; full covariance, not symmetric KL."""
    channels = real_mean.shape[1]
    real_mean, syn_mean = real_mean.double(), syn_mean.double()
    identity = torch.eye(channels, dtype=torch.float64, device=real_mean.device)
    real_covariance = real_covariance.double()
    syn_covariance = syn_covariance.double()
    real_covariance = 0.5 * (real_covariance + real_covariance.transpose(1, 2)) + ridge * identity
    syn_covariance = 0.5 * (syn_covariance + syn_covariance.transpose(1, 2)) + ridge * identity
    real_cholesky = torch.linalg.cholesky(real_covariance)
    syn_cholesky = torch.linalg.cholesky(syn_covariance)
    real_logdet = 2 * real_cholesky.diagonal(dim1=1, dim2=2).log().sum(1)
    syn_logdet = 2 * syn_cholesky.diagonal(dim1=1, dim2=2).log().sum(1)
    distances = []
    # Do not materialize an entire real-by-synthetic batch of covariance matrices.
    for mean, cholesky, logdet in zip(real_mean.split(32), real_cholesky.split(32), real_logdet.split(32)):
        precision = torch.cholesky_inverse(cholesky)
        difference = syn_mean.unsqueeze(0) - mean.unsqueeze(1)
        quadratic = ((difference @ precision) * difference).sum(2)
        trace = torch.einsum('bij,nji->bn', precision, syn_covariance)
        distance = 0.5 * (logdet[:, None] - syn_logdet[None, :] - channels + trace + quadratic)
        distances.append(distance.clamp_min(0) / channels)
    return torch.cat(distances)


@torch.no_grad()
def real_d3s_statistics(dataset, network, device, batch_size, regularization):
    """Fit fixed covariance ridges and KL scales on 256 unlabeled real training images."""
    count = min(256, len(dataset)) // 2 * 2
    if count < 4:
        raise ValueError('d3s calibration needs at least four real training images')
    indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(17000))[:count]
    subset = torch.utils.data.Subset(dataset, indices.tolist())
    loader = torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0,
                                        generator=torch.Generator().manual_seed(0))
    encoder = frozen_embedder(network, similarity='d3s')
    features = torch.cat([encoder(_images_from_batch(batch).to(device)).cpu() for batch in loader])
    moments, _ = unpack_d3s(features)
    ridges, scales = [], []
    for mean, covariance in moments:
        mean, covariance = mean.to(device), covariance.to(device)
        ridge = regularization * covariance.diagonal(dim1=1, dim2=2).mean().double().clamp_min(1e-6)
        half = count // 2
        distance = gaussian_kl(mean[:half], covariance[:half], mean[half:], covariance[half:], ridge)
        ridges.append(ridge.cpu())
        scales.append(distance.mean().clamp_min(1e-6).cpu())
    return {'ridge': torch.stack(ridges), 'scale': torch.stack(scales), 'indices': indices}


def make_diff_augmenter(strategy, param, dsa=False):
    """Build a seeded differentiable augmenter with independent per-image draws."""
    if dsa:
        return dm_augmenter(strategy, param)
    cuda_devices = list(range(torch.cuda.device_count()))

    def augment(images, seed):
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(seed)
            return DiffAugment(images, strategy, seed=-1, param=param)

    return augment


@torch.no_grad()
def deepcluster_initialize(dst_train, num_syn, device, seed, channel, num_classes, im_size, model_name='ConvNet', batch_size=128, init='random'):
    """Initialize from random real images or random-feature K-Means medoids."""
    if init == 'random':
        # Same indices and order as Evaluate_benchamrk/data.py's Random baseline.
        indices = np.random.default_rng(seed + 1000).permutation(len(dst_train))[:num_syn]
        assert len(indices) == num_syn, 'Not enough real images for initialization'
        print(f'Initializing {num_syn} synthetic images from the Random-baseline training images (seed={seed})...', flush=True)
        return torch.stack([_images_from_batch(dst_train[int(index)]) for index in indices]).to(device)

    assert init == 'kmeans'
    rng = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dst_train), generator=rng)[:num_syn]
    assert len(indices) == num_syn, 'Not enough real images for initialization'
    print(f'Initializing {num_syn} synthetic prototypes via DeepCluster K-Means on {model_name}...', flush=True)
    network = get_network(model_name, channel, num_classes, im_size, seed=seed).to(device)
    encoder = frozen_embedder(network)
    loader = torch.utils.data.DataLoader(dst_train, batch_size=batch_size, shuffle=False, num_workers=0)
    features = None
    position = 0
    for batch in loader:
        feats = encoder(_images_from_batch(batch).to(device)).cpu()
        if features is None:
            features = torch.empty(len(dst_train), feats.shape[1], dtype=feats.dtype)
        features[position:position + len(feats)] = feats
        position += len(feats)
    centers = features[indices].clone()
    for _ in range(5):
        assigns = torch.cat([torch.cdist(block, centers).argmin(dim=1) for block in features.split(1024)])
        counts = torch.bincount(assigns, minlength=num_syn)
        new_centers = torch.zeros_like(centers).index_add_(0, assigns, features)
        new_centers /= counts.clamp_min(1)[:, None]
        new_centers[counts == 0] = centers[counts == 0]
        if torch.allclose(centers, new_centers, atol=1e-4):
            break
        centers = new_centers
    closest_indices = torch.cat([torch.cdist(block, features).argmin(dim=1) for block in centers.split(32)])
    images = torch.stack([_images_from_batch(dst_train[int(index)]) for index in closest_indices])
    return images.to(device)


@torch.no_grad()
def real_feature_whitening(dataset, network, device, batch_size, regularization, feature_layers='last'):
    """Fit fixed whitening statistics on normalized real training features."""
    encoder = frozen_embedder(network, feature_layers)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0,
                                        generator=torch.Generator().manual_seed(0))
    count, feature_sum, second_moment = 0, None, None
    for batch in loader:
        features = F.normalize(encoder(_images_from_batch(batch).to(device)), dim=1)
        feature_dtype = features.dtype
        features = features.double()
        if feature_sum is None:
            feature_sum = torch.zeros(features.shape[1], dtype=features.dtype, device=device)
            second_moment = torch.zeros(features.shape[1], features.shape[1], dtype=features.dtype, device=device)
        feature_sum += features.sum(dim=0)
        second_moment += features.t() @ features
        count += len(features)
    assert count > 1, 'Whitening needs at least two real training images'
    mean = feature_sum / count
    covariance = (second_moment - count * mean[:, None] * mean[None, :]) / (count - 1)
    covariance = 0.5 * (covariance + covariance.t())
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    ridge = regularization * covariance.diagonal().mean().clamp_min(torch.finfo(covariance.dtype).eps)
    transform = (eigenvectors * (eigenvalues.clamp_min(0) + ridge).rsqrt()[None, :]) @ eigenvectors.t()
    return mean.to(feature_dtype), transform.to(feature_dtype)


def regularized_ot(cost, epsilon, iterations):
    """Log-domain Sinkhorn cost with KL regularization and uniform sample weights."""
    rows, columns = cost.shape[-2:]
    if rows == columns == 2:
        # Two equally weighted samples allow an exact entropic mixture of the two matchings.
        matching_costs = torch.stack(((cost[..., 0, 0] + cost[..., 1, 1]) / 2,
                                      (cost[..., 0, 1] + cost[..., 1, 0]) / 2), dim=-1)
        return -epsilon * (torch.logsumexp(-matching_costs / epsilon, dim=-1) - math.log(2))
    log_kernel = -cost / epsilon
    log_u = torch.zeros_like(cost[..., :, 0])
    log_v = torch.zeros_like(cost[..., 0, :])
    for _ in range(iterations):
        log_u = -math.log(rows) - torch.logsumexp(log_kernel + log_v.unsqueeze(-2), dim=-1)
        log_v = -math.log(columns) - torch.logsumexp(log_kernel + log_u.unsqueeze(-1), dim=-2)
    log_plan = log_kernel + log_u.unsqueeze(-1) + log_v.unsqueeze(-2)
    plan = log_plan.exp()
    return (plan * (cost + epsilon * (log_plan + math.log(rows * columns)))).sum(dim=(-2, -1))


def feature_similarity(real, synthetic, metric='cosine', mmd_sigma=1.0,
                       whitening=None, sinkhorn_epsilon=0.1, sinkhorn_iterations=50,
                       d3s_stats=None, d3s_weight=1.0):
    """Assignment scores; Gaussian moments stay unnormalized, larger means more similar."""
    if metric in ('dm', 'mmd-dist', 'energy', 'sinkhorn'):
        assert real.ndim == synthetic.ndim == 3, 'Distribution similarities need [images, augmented samples, features]'
    if metric == 'd3s':
        real_moments, real_final = unpack_d3s(real)
        syn_moments, syn_final = unpack_d3s(synthetic)
        scores = F.normalize(real_final, dim=1) @ F.normalize(syn_final, dim=1).t()
        if d3s_weight:
            if d3s_stats is None:
                raise ValueError('d3s needs fixed real-training covariance ridges and KL scales')
            distances = [gaussian_kl(*r, *s, d3s_stats['ridge'][index]) / d3s_stats['scale'][index]
                         for index, (r, s) in enumerate(zip(real_moments, syn_moments))]
            scores = scores - d3s_weight * torch.stack(distances).mean(0).to(scores.dtype)
    elif metric == 'cosine':
        assert real.ndim == synthetic.ndim and real.ndim in (2, 3)
        if real.ndim == 3:
            assert real.shape[1] == synthetic.shape[1]
            scores = torch.einsum('bkd,nkd->bn', real, synthetic) / real.shape[1]
        else:
            scores = torch.mm(real, synthetic.t())
    elif metric == 'whitened-cosine':
        assert whitening is not None, 'whitened-cosine needs fixed real-training statistics'
        mean, transform = whitening
        real_white = F.normalize((real - mean) @ transform, dim=1)
        syn_white = F.normalize((synthetic - mean) @ transform, dim=1)
        scores = torch.mm(real_white, syn_white.t())
    elif metric == 'l1':
        scores = -torch.cdist(real, synthetic, p=1)
    elif metric == 'l2':
        scores = -torch.cdist(real, synthetic, p=2)
    elif metric == 'dm':
        scores = -torch.cdist(real.mean(dim=1), synthetic.mean(dim=1), p=2).square()
    elif metric == 'energy':
        cross = torch.cdist(real.flatten(0, 1), synthetic.flatten(0, 1), p=2)
        cross = cross.reshape(len(real), real.shape[1], len(synthetic), synthetic.shape[1]).mean(dim=(1, 3))
        within_real = torch.cdist(real, real, p=2).mean(dim=(1, 2))
        within_syn = torch.cdist(synthetic, synthetic, p=2).mean(dim=(1, 2))
        scores = -(2 * cross - within_real[:, None] - within_syn[None, :]).clamp_min(0)
    elif metric == 'sinkhorn':
        cross_cost = torch.cdist(real.flatten(0, 1), synthetic.flatten(0, 1), p=2).square()
        cross_cost = cross_cost.reshape(len(real), real.shape[1], len(synthetic), synthetic.shape[1]).permute(0, 2, 1, 3)
        # Recompute transport iterations in backward instead of retaining every pair's solver graph.
        cross = (checkpoint(regularized_ot, cross_cost, sinkhorn_epsilon, sinkhorn_iterations, use_reentrant=False)
                 if cross_cost.requires_grad else regularized_ot(cross_cost, sinkhorn_epsilon, sinkhorn_iterations))
        within_real = regularized_ot(torch.cdist(real, real, p=2).square(), sinkhorn_epsilon, sinkhorn_iterations)
        within_syn = regularized_ot(torch.cdist(synthetic, synthetic, p=2).square(), sinkhorn_epsilon, sinkhorn_iterations)
        scores = -(cross - 0.5 * within_real[:, None] - 0.5 * within_syn[None, :]).clamp_min(0)
    elif metric == 'mmd':
        squared_distance = torch.cdist(real, synthetic, p=2).square()
        kernel = torch.exp(-squared_distance / (2 * mmd_sigma ** 2))
        # For singleton distributions, RBF-MMD squared is k(x,x)+k(y,y)-2k(x,y).
        scores = -2 * (1 - kernel)
    elif metric == 'mmd-dist':
        def kernel(first, second):
            return torch.exp(-torch.cdist(first, second, p=2).square() / (2 * mmd_sigma ** 2))

        within_real = kernel(real, real).mean(dim=(1, 2))
        within_syn = kernel(synthetic, synthetic).mean(dim=(1, 2))
        cross = kernel(real.flatten(0, 1), synthetic.flatten(0, 1))
        cross = cross.reshape(len(real), real.shape[1], len(synthetic), synthetic.shape[1]).mean(dim=(1, 3))
        scores = -(within_real[:, None] + within_syn[None, :] - 2 * cross).clamp_min(0)
    else:
        raise ValueError('Unknown feature similarity: ' + metric)
    assert scores.shape == (len(real), len(synthetic)), 'Similarity must remain a real-by-synthetic matrix'
    return scores


def bipartite_view_transition_loss(features_u_real, features_v_real, features_u_syn, features_v_syn, tau,
                                  similarity='cosine', mmd_sigma=1.0,
                                  whitening=None, sinkhorn_epsilon=0.1, sinkhorn_iterations=50,
                                  d3s_stats=None, d3s_weight=1.0,
                                  assignment_filter=False, assignment_threshold=0.05):
    """Compute the same MI objective with the selected soft-assignment similarity."""
    if similarity == 'd3s':
        u_r, v_r, u_s, v_s = features_u_real, features_v_real, features_u_syn, features_v_syn
    else:
        u_r = F.normalize(features_u_real, dim=-1)
        v_r = F.normalize(features_v_real, dim=-1)
        u_s = F.normalize(features_u_syn, dim=-1)
        v_s = F.normalize(features_v_syn, dim=-1)

    # Bipartite similarities: u_r -> u_s and v_r -> v_s [Batch, Num_Syn]
    sim_u = feature_similarity(u_r, u_s, similarity, mmd_sigma, whitening, sinkhorn_epsilon, sinkhorn_iterations, d3s_stats, d3s_weight)
    sim_v = feature_similarity(v_r, v_s, similarity, mmd_sigma, whitening, sinkhorn_epsilon, sinkhorn_iterations, d3s_stats, d3s_weight)

    # Soft assignments to paired synthetic prototypes
    p_u = F.softmax(sim_u / tau, dim=1)
    p_v = F.softmax(sim_v / tau, dim=1)

    if assignment_filter:
        with torch.no_grad():
            log_prototypes = math.log(p_u.shape[1])
            confidence = [((p * (p.clamp_min(torch.finfo(p.dtype).tiny).log() + log_prototypes)).sum(1)
                           / log_prototypes).clamp(0, 1) if log_prototypes else torch.zeros_like(p[:, 0])
                          for p in (p_u, p_v)]
            keep = (torch.minimum(*confidence) > assignment_threshold)
        if not keep.any():
            # No accepted pairs: detached zeros also prevent an Adam momentum-only update.
            zero = p_u.new_zeros(())
            return zero, zero, zero
        p_u, p_v = p_u[keep], p_v[keep]
        u_r, v_r = u_r[keep], v_r[keep]

    # Joint probability matrix P [Num_Syn, Num_Syn]
    B = u_r.shape[0]
    P = torch.mm(p_u.t(), p_v) / B
    P = 0.5 * (P + P.t())
    P = torch.clamp(P, min=1e-8)

    P_i = P.sum(dim=1, keepdim=True)
    P_j = P.sum(dim=0, keepdim=True)

    # 1. Negative Bipartite Mutual Information
    MI = (P * torch.log(P / (P_i * P_j))).sum()
    negative_mi = -MI

    if similarity == 'd3s':
        u_r, v_r, u_s, v_s = [F.normalize(unpack_d3s(features)[1], dim=1)
                              for features in (u_r, v_r, u_s, v_s)]
    if similarity in ('mmd-dist', 'dm', 'energy', 'sinkhorn') or (similarity == 'cosine' and u_r.ndim == 3):
        u_r, v_r, u_s, v_s = [F.normalize(features.mean(dim=1), dim=1)
                              for features in (u_r, v_r, u_s, v_s)]

    # 2. Intra-Pair View-Shift Alignment (Semantic Collage Constraint)
    # Responsibility of each real pair b for synthetic prototype k: w_bk [Batch, Num_Syn]
    resp = 0.5 * (p_u + p_v)
    resp_norm = resp / (resp.sum(dim=0, keepdim=True) + 1e-6)  # [Batch, Num_Syn]

    delta_real = u_r - v_r  # [Batch, D]
    delta_syn = u_s - v_s   # [Num_Syn, D]

    # Expected real directional shift for each prototype k: [Num_Syn, D]
    target_delta = torch.mm(resp_norm.t(), delta_real)
    shift_loss = F.mse_loss(delta_syn, target_delta)

    # 3. Conditional Intra-View Discrepancy Matching (Anti-Shortcut Repulsion)
    # Match the cosine similarity of synthetic view pairs (u_s, v_s) to the real view pairs assigned to k
    real_pair_cos = (u_r * v_r).sum(dim=1, keepdim=True)  # [Batch, 1]
    target_pair_cos = torch.mm(resp_norm.t(), real_pair_cos).squeeze(1)  # [Num_Syn]
    syn_pair_cos = (u_s * v_s).sum(dim=1)  # [Num_Syn]
    # Match pair similarity in either direction.
    repulsion_loss = F.smooth_l1_loss(syn_pair_cos, target_pair_cos)

    return negative_mi, shift_loss, repulsion_loss


def main():
    parser = argparse.ArgumentParser(description='Breakthrough 1: Bipartite View-Pair MI + Shift Alignment')


    # 'teacher_source': 'pretrained', 'teacher_seed': 0, 'teacher_ssl': 'simclr', 'teacher_path': '/scratch/mmoslem3/MKDT/Evaluate_benchamrk/full_ssl_checkpoints_1000ep_3seeds', 'encoder_ssl_protocol': None, 'method': 'Bipartite_ViewTransition_MI', 'dsa_param': <utils.ParamDiffAug object at 0x723263394550>}
    # Data.
    parser.add_argument('--dataset', choices=['CIFAR10', 'CIFAR100', 'TinyImageNet', 'FashionMNIST', 'SVHN'], default='FashionMNIST')
    parser.add_argument('--percentage', type=float, default=0.2)
    parser.add_argument('--data_path', default='/home/mmoslem3/scratch/data')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--init', choices=['random', 'kmeans'], default='random', help='Synthetic-image initialization from real training data')
    parser.add_argument('--single_channel', action='store_true', help='FashionMNIST: optimize one grayscale channel and repeat it to RGB for teachers and saved images')

    # Information-Theoretic Objective.
    parser.add_argument('--Iteration', type=int, default=5000)
    parser.add_argument('--lr_img', type=float, default=0.002, help='Higher LR, decays via CosineAnnealing')
    parser.add_argument('--batch_real', type=int, default=256)

    parser.add_argument('--temperature', type=float, default=0.2, help='Softmax temperature for prototype assignment')
    parser.add_argument('--assignment_filter', action='store_true', help='Exclude real images unless both branches pass the normalized KL-to-uniform threshold')
    parser.add_argument('--assignment_threshold', type=float, default=0.05, help='Minimum KL(assignment || uniform) / log(num_syn) in both branches, in [0,1]; ignored unless assignment_filter is enabled')
    parser.add_argument('--similarity', choices=['cosine', 'l1', 'l2', 'mmd', 'mmd-dist', 'dm', 'energy', 'whitened-cosine', 'sinkhorn', 'd3s'], default='cosine', help='Similarity inside softmax assignments; d3s always uses middle-layer full Gaussian KL plus final cosine; every option keeps MI')
    parser.add_argument('--feature_layers', choices=['last', 'all'], default='last', help='For non-d3s similarities: last uses existing final features; all adds separately normalized ResNet layer2/layer3 channel mean+std')
    parser.add_argument('--d3s_weight', type=nonnegative_float, default=1.0, help='Weight of real-calibrated middle-layer Gaussian KL inside d3s similarity; final cosine keeps weight 1')
    parser.add_argument('--d3s_reg', type=float, default=0.001, help='Covariance ridge as a fraction of real-training within-image mean variance, fixed per layer')
    parser.add_argument('--mmd_sigma', type=float, default=1.0, help='RBF bandwidth for mmd and mmd-dist')
    parser.add_argument('--cosine_views', type=int, default=1, help='Paired augmentation draws per MI branch; average cosine matrices before softmax')
    parser.add_argument('--mmd_views', type=int, default=2, help='Augmented feature samples per image in each independent MI branch for mmd-dist')
    parser.add_argument('--dm_views', type=int, default=2, help='Augmented feature samples per image to average in each independent MI branch for dm')
    parser.add_argument('--energy_views', type=int, default=2, help='Augmented samples per image in each independent MI branch for energy distance')
    parser.add_argument('--sinkhorn_views', type=int, default=2, help='Augmented samples per image in each independent MI branch for Sinkhorn divergence')
    parser.add_argument('--sinkhorn_epsilon', type=float, default=0.1, help='Entropic regularization for Sinkhorn divergence on normalized features')
    parser.add_argument('--sinkhorn_iterations', type=int, default=50, help='Log-domain Sinkhorn iterations')
    parser.add_argument('--whitening_reg', type=float, default=0.001, help='Whitening ridge as a fraction of mean real-feature variance')
    parser.add_argument('--disp_weight', type=nonnegative_float, default=0.0, help='Weight for feature displacement matching')
    parser.add_argument('--pair_weight', type=nonnegative_float, default=0.0, help='Weight for view-pair similarity matching')
    

    parser.add_argument('--num_random_networks', type=int, default=1)

    parser.add_argument('--random_models', default='ResNet18BN', help='Comma-separated P_net support')
    parser.add_argument('--distill_aug_strategy', '--dsa_strategy', '--dsa-strategy', default='color_crop_cutout_flip_scale_rotate')
    parser.add_argument('--dsa', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--distill_aug_mode', choices=['S', 'M'], default='M')
    parser.add_argument('--distill_aug_backend', choices=['dsa', 'simclr'], default='dsa', help='Legacy DiffAugment or differentiable views from the benchmark SimCLR recipe')
    parser.add_argument('--simclr_compose_probability', type=float, default=0.5, help='For SimCLR distillation views: probability of the full pipeline instead of one augmentation family')

    # Teacher.
    parser.add_argument('--teacher_source', choices=['fresh', 'pretrained'], default='pretrained')
    parser.add_argument('--teacher_seed', type=int, nargs='+', default=[0,1,2], help='Saved teacher seeds; their losses are summed')
    parser.add_argument('--teacher_ssl', choices=['simclr', 'barlowtwins'], default='simclr')
    parser.add_argument('--teacher_path', default=None, help='Automatically selects the saved teacher directory for the dataset')

    # Downstream SSL evaluation.
    parser.add_argument('--model', choices=['ConvNet', 'ConvNetBN', 'VGG11', 'VGG11BN', 'ResNet18', 'ResNet18BN'], default='ResNet18BN')
    parser.add_argument('--ssl_method', choices=['simclr', 'barlowtwins'], default='simclr')
    parser.add_argument('--eval_mode', choices=['S', 'SS', 'M'], default='S')
    parser.add_argument('--num_eval', type=nonnegative_int, default=0)
    parser.add_argument('--epoch_eval_train', type=int, default=10000)
    parser.add_argument('--label_percentage', type=float, choices=[1.0, 5.0], default=1.0)
    parser.add_argument('--lr_linear', type=float, default=0.1)
    parser.add_argument('--batch_linear', type=int, default=256)

    # Output.
    parser.add_argument('--save_path', default='newdata')
    parser.add_argument('--resume', action='store_true', help='Resume the checkpoint in save_path, or skip a completed run')
    parser.add_argument('--save_every', type=int, default=100, help='Replace the single checkpoint every N iterations')
    parser.add_argument('--feature_batch', type=int, default=128, help='Frozen feature extraction chunk size')

    args = parser.parse_args()
    if not math.isfinite(args.assignment_threshold) or not 0 <= args.assignment_threshold <= 1:
        parser.error('assignment_threshold must be finite and in [0, 1]')
    if args.feature_layers == 'all' or args.similarity == 'd3s':
        feature_models = [args.model] if args.teacher_source == 'pretrained' else (args.random_models or args.model).split(',')
        if any(name.strip() not in ('ResNet18', 'ResNet18BN') for name in feature_models):
            parser.error('Intermediate features support ResNet18/ResNet18BN teachers')
    if args.similarity == 'd3s':
        if args.teacher_source != 'pretrained':
            parser.error('d3s needs saved teachers to keep real-training calibration fixed')
        if not math.isfinite(args.d3s_reg) or args.d3s_reg <= 0:
            parser.error('d3s_reg must be finite and positive')
    if args.single_channel and args.dataset != 'FashionMNIST':
        parser.error('single_channel is only for FashionMNIST')
    if not math.isfinite(args.simclr_compose_probability) or not 0 <= args.simclr_compose_probability <= 1:
        parser.error('simclr_compose_probability must be in [0, 1]')
    if min(args.Iteration, args.save_every, args.feature_batch, args.batch_real, args.num_random_networks) < 1:
        parser.error('Iterations, saving interval, batches and augmentation draws must be positive')
    if not 0 < args.percentage <= 100:
        parser.error('percentage must be in (0, 100]')
    if not math.isfinite(args.mmd_sigma) or args.mmd_sigma <= 0:
        parser.error('mmd_sigma must be finite and positive')
    if args.similarity == 'mmd-dist' and args.mmd_views < 2:
        parser.error('mmd-dist needs at least two augmented samples per image; use mmd for singleton features')
    if args.similarity == 'dm' and args.dm_views < 1:
        parser.error('dm_views must be positive')
    if args.cosine_views < 1:
        parser.error('cosine_views must be positive')
    if min(args.energy_views, args.sinkhorn_views, args.sinkhorn_iterations) < 1:
        parser.error('Energy/Sinkhorn sample counts and Sinkhorn iterations must be positive')
    if any(not math.isfinite(value) or value <= 0 for value in (args.sinkhorn_epsilon, args.whitening_reg)):
        parser.error('sinkhorn_epsilon and whitening_reg must be finite and positive')
    if args.similarity == 'whitened-cosine' and args.teacher_source != 'pretrained':
        parser.error('whitened-cosine needs saved teachers to keep real-feature statistics fixed')
    if args.teacher_path is None:
        teacher_folder = {'FashionMNIST': 'fashionmnist/full/checkpoints',
                          'SVHN': 'digits_simclr/checkpoints'}.get(args.dataset, 'full_ssl_checkpoints_1000ep_3seeds')
        args.teacher_path = str(ROOT / 'Evaluate_benchamrk' / teacher_folder)

    output = Path(args.save_path)
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / ('res_IIC_%s_%s_%gpercent_final.pt' % (args.dataset, args.model, args.percentage))
    png_name = 'vis_IIC_%s_%s_%gpercent_iter%%d.png' % (args.dataset, args.model, args.percentage)
    run_args = {key: getattr(args, key) for key in (
        'dataset', 'percentage', 'model', 'seed', 'Iteration', 'lr_img', 'batch_real',
        'temperature', 'disp_weight', 'pair_weight', 'num_random_networks', 'random_models',
        'dsa', 'distill_aug_strategy', 'distill_aug_mode', 'teacher_source', 'teacher_seed',
        'teacher_ssl', 'teacher_path', 'feature_batch')}
    if args.single_channel:
        run_args['single_channel'] = True
    if args.feature_layers == 'all':
        run_args['feature_layers'] = 'all'
    if args.similarity == 'd3s':
        run_args.update(d3s_weight=args.d3s_weight, d3s_reg=args.d3s_reg)
    if args.similarity == 'cosine' and args.cosine_views > 1:
        run_args['cosine_views'] = args.cosine_views
    if args.assignment_filter:
        run_args.update(assignment_filter=True, assignment_threshold=args.assignment_threshold)
    if args.distill_aug_backend == 'simclr':
        run_args.update(distill_aug_backend='simclr', simclr_compose_probability=args.simclr_compose_probability)
    saved = None
    if args.resume and result_path.exists():
        saved = torch.load(result_path, map_location='cpu', weights_only=True)
        previous_similarity = saved.get('iic_params', {})
        if (previous_similarity.get('assignment_filter', False) != args.assignment_filter
                or (args.assignment_filter and previous_similarity.get('assignment_threshold') != args.assignment_threshold)):
            raise ValueError('Checkpoint assignment filtering settings differ from this run')
        if previous_similarity.get('feature_layers', 'last') != args.feature_layers:
            raise ValueError('Checkpoint feature layers differ from this run')
        if (previous_similarity.get('similarity', 'cosine') != args.similarity
                or (args.similarity in ('mmd', 'mmd-dist') and previous_similarity.get('mmd_sigma', 1.0) != args.mmd_sigma)):
            raise ValueError('Checkpoint similarity or MMD bandwidth differs from this run')
        if args.similarity == 'mmd-dist' and previous_similarity.get('mmd_views') != args.mmd_views:
            raise ValueError('Checkpoint uses the old direct-MMD objective or a different number of MMD views')
        if args.similarity == 'dm' and previous_similarity.get('dm_views') != args.dm_views:
            raise ValueError('Checkpoint uses a different number of DM views')
        if args.similarity == 'cosine' and previous_similarity.get('cosine_views', 1) != args.cosine_views:
            raise ValueError('Checkpoint uses a different number of cosine views')
        metric_parameters = {'energy': ('energy_views',),
                             'sinkhorn': ('sinkhorn_views', 'sinkhorn_epsilon', 'sinkhorn_iterations'),
                             'whitened-cosine': ('whitening_reg',),
                             'd3s': ('d3s_weight', 'd3s_reg')}.get(args.similarity, ())
        if any(previous_similarity.get(name) != getattr(args, name) for name in metric_parameters):
            raise ValueError('Checkpoint similarity settings differ from this run')
        if 'init' in saved and saved['init'] != args.init:
            raise ValueError('Checkpoint initialization differs from --init')
        if saved.get('run_args') != run_args:
            raise ValueError('Checkpoint settings differ from this run; do not resume with changed parameters')
        if not 0 <= saved['iteration'] <= args.Iteration:
            raise ValueError('Invalid saved iteration')
        if saved.get('complete'):
            if saved['iteration'] != args.Iteration or not all((output / (png_name % i)).is_file() for i in (1, args.Iteration)):
                raise ValueError('Completed checkpoint is missing its first/final PNG or iteration')
            print('SKIP completed: %s (iteration %d)' % (result_path, saved['iteration']), flush=True)
            return

    random_models = [name.strip() for name in (args.random_models or args.model).split(',') if name.strip()]

    method = 'Bipartite_ViewTransition_MI'

    distill_dsa_param = ParamDiffAug()
    distill_dsa_param.aug_mode = 'S' if args.dsa else args.distill_aug_mode
    set_random_seed(args.seed)

    os.makedirs(args.data_path, exist_ok=True)
    data = reference_modules()['data']
    if args.dataset in ('FashionMNIST', 'SVHN'):
        raw_train, (mean, std) = digit_training_data(args)
    else:
        raw_train = data.RawDataset(args.data_path, args.dataset, True)
        mean, std = SYNTHETIC_NORMALIZATION[args.dataset]
    channel, im_size, num_classes = 3, ((64, 64) if args.dataset == 'TinyImageNet' else (32, 32)), raw_train.classes
    # Keep the same normalized training images without materializing all of TinyImageNet as float tensors.
    dst_train = data.ImageView(raw_train, transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)]))
    num_syn = int(len(dst_train) * args.percentage / 100)
    teachers = [load_teacher(argparse.Namespace(**dict(vars(args), teacher_seed=seed)), mean, std, im_size)
                for seed in args.teacher_seed] if args.teacher_source == 'pretrained' else [None]

    if not 2 <= num_syn <= len(dst_train) or args.batch_real > len(dst_train):
        raise ValueError('Invalid synthetic count or real batch size')
    whitening_stats = None
    if args.similarity == 'whitened-cosine':
        if saved is None:
            whitening_stats = []
            for seed, teacher in zip(args.teacher_seed, teachers):
                print('Computing fixed real-feature whitening for teacher seed %d...' % seed, flush=True)
                whitening_stats.append(real_feature_whitening(dst_train, teacher, args.device, args.feature_batch, args.whitening_reg, args.feature_layers))
        else:
            whitening_stats = saved.get('whitening_stats')
            if whitening_stats is None or len(whitening_stats) != len(teachers):
                raise ValueError('Checkpoint is missing per-teacher whitening statistics')
            whitening_stats = [tuple(value.to(args.device) for value in stats) for stats in whitening_stats]
        for feature_mean, transform in whitening_stats:
            if (feature_mean.ndim != 1 or transform.shape != (len(feature_mean), len(feature_mean))
                    or not torch.isfinite(feature_mean).all() or not torch.isfinite(transform).all()):
                raise ValueError('Invalid saved whitening statistics')
    d3s_stats = None
    if args.similarity == 'd3s':
        if saved is None:
            d3s_stats = []
            for seed, teacher in zip(args.teacher_seed, teachers):
                print('Calibrating middle-layer Gaussian KL on real training images for teacher seed %d...' % seed, flush=True)
                d3s_stats.append(real_d3s_statistics(dst_train, teacher, args.device, args.feature_batch, args.d3s_reg))
        else:
            d3s_stats = saved.get('d3s_stats')
            if d3s_stats is None or len(d3s_stats) != len(teachers):
                raise ValueError('Checkpoint is missing per-teacher d3s calibration')
        for stats in d3s_stats:
            for name in ('ridge', 'scale'):
                if stats[name].shape != (2,) or not torch.isfinite(stats[name]).all() or not (stats[name] > 0).all():
                    raise ValueError('Invalid d3s covariance ridge or similarity scale')
                stats[name] = stats[name].to(args.device)
            print('d3s calibration: ridge=%s, KL scale=%s' % (stats['ridge'].tolist(), stats['scale'].tolist()), flush=True)
    if saved is None:
        init_options = {'init': 'kmeans'} if args.init == 'kmeans' else {}
        image_syn_init = deepcluster_initialize(
            dst_train, num_syn, args.device, args.seed, channel, num_classes, im_size,
            model_name=random_models[0], batch_size=args.feature_batch, **init_options
        )
        image_syn = image_syn_init.detach().requires_grad_(True)
        image_syn_initial = image_syn.detach().clone()
    else:
        if saved['data'].shape != (num_syn, channel, *im_size) or saved['normalization'] != [mean, std]:
            raise ValueError('Checkpoint image shape or normalization differs from this dataset')
        image_syn = saved['data'].to(args.device).detach().requires_grad_(True)
        image_syn_initial = saved['initial_data'].to(args.device)

    if args.single_channel:
        assert mean == [mean[0]] * channel and std == [std[0]] * channel
        assert torch.equal(image_syn, image_syn[:, :1].expand_as(image_syn))
        assert torch.equal(image_syn_initial, image_syn_initial[:, :1].expand_as(image_syn_initial))
        image_syn = image_syn[:, :1].detach().clone().requires_grad_(True)
        image_syn_initial = image_syn_initial[:, :1].contiguous()
    diag_mean = torch.tensor(mean[:1] if args.single_channel else mean, device=args.device).view(1, -1, 1, 1)
    diag_std = torch.tensor(std[:1] if args.single_channel else std, device=args.device).view(1, -1, 1, 1)
    pixel_min = -diag_mean / diag_std
    pixel_max = (1 - diag_mean) / diag_std

    def to_uint8(images):
        return ((images * diag_std + diag_mean) * 255 + 0.5).clamp(0, 255).to(torch.uint8)

    initial_uint8 = to_uint8(image_syn_initial)

    data_rng = torch.Generator().manual_seed(args.seed + 20_000)
    real_order = torch.randperm(len(dst_train), generator=data_rng)
    real_position = 0

    def real_batches():
        return iter(torch.utils.data.DataLoader(
            dst_train, batch_size=args.batch_real, sampler=real_order[real_position:].tolist(),
            num_workers=0, pin_memory=args.device == 'cuda', drop_last=True,
            generator=torch.Generator().manual_seed(args.seed)
        ))

    augmenter = make_diff_augmenter(args.distill_aug_strategy, distill_dsa_param, args.dsa)
    if args.distill_aug_backend == 'simclr':
        from simclr_dsa import SimCLRDSA
        if args.dataset in ('FashionMNIST', 'SVHN'):
            prepared_file = ROOT / 'Evaluate_benchamrk' / ('fashionmnist/full/prepared.json' if args.dataset == 'FashionMNIST' else 'digits_simclr/prepared.json')
            aug_protocol = json.loads(prepared_file.read_text())['protocol']
        else:
            aug_protocol = json.loads((ROOT / 'Evaluate_benchamrk/protocol.json').read_text())
        ssl_view = data.ssl_transform(aug_protocol, args.dataset, 'simclr').transforms[0]
        augmenter = SimCLRDSA(ssl_view, mean, std, args.simclr_compose_probability)
        print('Distillation augmentation: SimCLR operations, %.0f%% full composition, otherwise one family; Siamese=%s' %
              (100 * args.simclr_compose_probability, args.dsa), flush=True)
    objective_rng = np.random.default_rng(args.seed + 10_000)

    optimizer_img = torch.optim.Adam([image_syn], lr=args.lr_img)
    scheduler_img = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_img, T_max=args.Iteration, eta_min=args.lr_img)

    loss_history = []
    negative_mi_history = []
    shift_loss_history = []
    total_loss_history = []
    model_eval_pool = ([args.model] if args.eval_mode in ('S', 'SS') else ['ConvNet', 'VGG11', 'ResNet18']) if args.num_eval else []

    common = reference_modules()['common']
    start_iteration = 0
    if saved is not None:
        start_iteration = saved['iteration']
        optimizer_img.load_state_dict(saved['optimizer'])
        scheduler_img.load_state_dict(saved['scheduler'])
        objective_rng.bit_generator.state = saved['objective_rng']
        data_rng.set_state(saved['data_rng'])
        real_order, real_position = saved['real_order'], saved['real_position']
        common.restore_rng(saved['rng'])
        numpy_rng = saved['numpy_rng']
        np.random.set_state((numpy_rng[0], np.asarray(numpy_rng[1], dtype=np.uint32), *numpy_rng[2:]))
        loss_history = saved['loss_history'].tolist()
        negative_mi_history = saved['negative_mi_history'].tolist()
        shift_loss_history = saved['shift_loss_history'].tolist()
        total_loss_history = saved['total_loss_history'].tolist()
        assert len(loss_history) == start_iteration and 0 <= real_position <= len(real_order) == len(dst_train)
        print('RESUME iteration %d -> %d, including Adam, sampling and RNG state: %s' %
              (start_iteration, args.Iteration, result_path), flush=True)
    real_iter = real_batches()

    def save_checkpoint(iteration):
        numpy_rng = np.random.get_state()
        common.atomic_save(result_path, {
            'data': image_syn.detach().expand(-1, channel, -1, -1).contiguous().cpu(),
            'initial_data': image_syn_initial.detach().expand(-1, channel, -1, -1).contiguous().cpu(),
            'dataset': args.dataset, 'percentage': args.percentage, 'normalization': [mean, std],
            'method': method, 'iteration': iteration, 'complete': iteration == args.Iteration,
            'run_args': run_args, 'init': args.init, 'optimizer': optimizer_img.state_dict(), 'scheduler': scheduler_img.state_dict(),
            'rng': common.rng_state(), 'numpy_rng': (numpy_rng[0], numpy_rng[1].tolist(), *numpy_rng[2:]),
            'objective_rng': objective_rng.bit_generator.state, 'data_rng': data_rng.get_state(),
            'real_order': real_order, 'real_position': real_position,
            'dsa': args.dsa, 'teacher_source': args.teacher_source, 'teacher_seed': args.teacher_seed,
            'teacher_loss_reduction': 'sum', 'seed': args.seed, 'model': args.model,
            'whitening_stats': None if whitening_stats is None else [tuple(value.cpu() for value in stats) for stats in whitening_stats],
            'd3s_stats': None if d3s_stats is None else [{name: value.cpu() for name, value in stats.items()} for stats in d3s_stats],
            'loss_history': torch.tensor(loss_history, dtype=torch.float64),
            'negative_mi_history': torch.tensor(negative_mi_history, dtype=torch.float64),
            'shift_loss_history': torch.tensor(shift_loss_history, dtype=torch.float64),
            'total_loss_history': torch.tensor(total_loss_history, dtype=torch.float64),
            'iic_params': {'tau': args.temperature, 'disp_weight': args.disp_weight,
                           'pair_weight': args.pair_weight, 'strategy': 'simclr_single_or_composed' if args.distill_aug_backend == 'simclr' else args.distill_aug_strategy,
                           'similarity': args.similarity, 'feature_layers': args.feature_layers, 'mmd_sigma': args.mmd_sigma,
                           'cosine_views': args.cosine_views if args.similarity == 'cosine' else 1,
                           'assignment_filter': args.assignment_filter, 'assignment_threshold': args.assignment_threshold,
                           'mmd_views': args.mmd_views if args.similarity == 'mmd-dist' else 1,
                           'dm_views': args.dm_views if args.similarity == 'dm' else 1,
                           'energy_views': args.energy_views, 'sinkhorn_views': args.sinkhorn_views,
                           'sinkhorn_epsilon': args.sinkhorn_epsilon, 'sinkhorn_iterations': args.sinkhorn_iterations,
                           'whitening_reg': args.whitening_reg, 'd3s_weight': args.d3s_weight, 'd3s_reg': args.d3s_reg,
                           'batch_real': args.batch_real}
        })
        print('Saved synthetic data to %s (iteration %d/%d)' % (result_path, iteration, args.Iteration), flush=True)

    if saved is None:
        save_checkpoint(0)
    del saved

    print('Hyper-parameters: \n', args.__dict__)
    print('P_net support: %s' % ', '.join(random_models), flush=True)
    if args.assignment_filter:
        print('Assignment filter: normalized KL > %g in both branches; empty draws skip optimization.' % args.assignment_threshold, flush=True)
    print('Objective: sum teacher losses (-MI + %g * displacement + %g * pair_similarity), averaged over %d augmentation draw(s).' %
          (args.disp_weight, args.pair_weight, args.num_random_networks), flush=True)
    print('%s distillation begins' % get_time(), flush=True)

    for iteration in range(start_iteration + 1, args.Iteration + 1):
        optimizer_img.zero_grad(set_to_none=True)
        sampled_losses = []
        sampled_negative_mi_losses = []
        sampled_shift_losses = []
        sampled_disp_losses = []
        sampled_pair_losses = []

        try:
            batch = next(real_iter)
        except StopIteration:
            real_order = torch.randperm(len(dst_train), generator=data_rng)
            real_position = 0
            real_iter = real_batches()
            batch = next(real_iter)

        real_imgs = _images_from_batch(batch).to(args.device)
        real_position += len(real_imgs)

        for _ in range(args.num_random_networks):
            # Reuse each augmentation draw across all teachers.
            seed_u_real = int(objective_rng.integers(0, 2**31 - 1))
            seed_v_real = int(objective_rng.integers(0, 2**31 - 1))
            seed_u_syn = int(objective_rng.integers(0, 2**31 - 1))
            seed_v_syn = int(objective_rng.integers(0, 2**31 - 1))
            if args.dsa:
                seed_u_syn, seed_v_syn = seed_u_real, seed_v_real
            view_seeds = [(seed_u_real, seed_v_real, seed_u_syn, seed_v_syn)]
            if args.similarity in ('cosine', 'mmd-dist', 'dm', 'energy', 'sinkhorn'):
                views = {'cosine': args.cosine_views, 'mmd-dist': args.mmd_views, 'dm': args.dm_views,
                         'energy': args.energy_views, 'sinkhorn': args.sinkhorn_views}[args.similarity]
                for _ in range(views - 1):
                    seeds = [int(objective_rng.integers(0, 2**31 - 1)) for _ in range(4)]
                    if args.dsa:
                        seeds[2:] = seeds[:2]
                    view_seeds.append(seeds)

            for teacher_index, teacher in enumerate(teachers):
                model_name = random_models[int(objective_rng.integers(len(random_models)))]
                network_seed = int(objective_rng.integers(0, 2**31 - 1))
                network = teacher if teacher is not None else get_network(model_name, channel, num_classes, im_size, seed=network_seed).to(args.device)
                encoder = frozen_embedder(network, args.feature_layers, args.similarity)

                def embed(images):
                    # Recompute frozen activations in backward instead of retaining two full synthetic graphs.
                    return torch.cat([checkpoint(encoder, block, use_reentrant=False) if block.requires_grad else encoder(block)
                                      for block in images.split(args.feature_batch)])

                feature_views = []
                for seed_u_real, seed_v_real, seed_u_syn, seed_v_syn in view_seeds:
                    with torch.no_grad():
                        features_u_real = embed(augmenter(real_imgs, seed_u_real))
                        features_v_real = embed(augmenter(real_imgs, seed_v_real))
                    features_u_syn = embed(augmenter(image_syn.expand(-1, channel, -1, -1), seed_u_syn))
                    features_v_syn = embed(augmenter(image_syn.expand(-1, channel, -1, -1), seed_v_syn))
                    feature_views.append((features_u_real, features_v_real, features_u_syn, features_v_syn))
                if args.similarity in ('mmd-dist', 'dm', 'energy', 'sinkhorn') or (args.similarity == 'cosine' and args.cosine_views > 1):
                    features_u_real, features_v_real, features_u_syn, features_v_syn = [
                        torch.stack(group, dim=1) for group in zip(*feature_views)]
                del feature_views

                negative_mi, disp_loss, pair_loss = bipartite_view_transition_loss(
                    features_u_real, features_v_real, features_u_syn, features_v_syn, tau=args.temperature,
                    similarity=args.similarity, mmd_sigma=args.mmd_sigma,
                    whitening=None if whitening_stats is None else whitening_stats[teacher_index],
                    sinkhorn_epsilon=args.sinkhorn_epsilon, sinkhorn_iterations=args.sinkhorn_iterations,
                    d3s_stats=None if d3s_stats is None else d3s_stats[teacher_index], d3s_weight=args.d3s_weight,
                    assignment_filter=args.assignment_filter, assignment_threshold=args.assignment_threshold
                )
                network_loss = (negative_mi + args.disp_weight * disp_loss
                                + args.pair_weight * pair_loss)

                if network_loss.requires_grad:
                    (network_loss / args.num_random_networks).backward()

                sampled_losses.append(network_loss.detach())
                sampled_negative_mi_losses.append(negative_mi.detach())
                # Keep the historical combined diagnostic compatible with old checkpoint histories.
                sampled_shift_losses.append((disp_loss + 0.5 * pair_loss).detach())
                sampled_disp_losses.append(disp_loss.detach())
                sampled_pair_losses.append(pair_loss.detach())
                del encoder, network, network_loss, negative_mi, disp_loss, pair_loss
                del features_u_real, features_v_real, features_u_syn, features_v_syn

        objective = torch.stack(sampled_losses).sum().item() / args.num_random_networks
        negative_mi_value = torch.stack(sampled_negative_mi_losses).sum().item() / args.num_random_networks
        shift_loss_value = torch.stack(sampled_shift_losses).sum().item() / args.num_random_networks
        disp_loss_value = torch.stack(sampled_disp_losses).sum().item() / args.num_random_networks
        pair_loss_value = torch.stack(sampled_pair_losses).sum().item() / args.num_random_networks
        total_loss_value = objective

        if image_syn.grad is not None:
            optimizer_img.step()
            scheduler_img.step()

        with torch.no_grad():
            if image_syn.grad is not None:
                image_syn.copy_(torch.maximum(torch.minimum(image_syn, pixel_max), pixel_min))
            drift = ((image_syn - image_syn_initial) * diag_std * 255).square().mean().sqrt().item()

        loss_history.append(objective)
        negative_mi_history.append(negative_mi_value)
        shift_loss_history.append(shift_loss_value)
        total_loss_history.append(total_loss_value)

        if iteration % 25 == 0:
            current_lr = scheduler_img.get_last_lr()[0]
            print('%s iter = %05d, neg_MI = %.6f, displacement = %.6f, pair_similarity = %.6f, total_loss = %.6f, pixel drift RMS = %.5f/255' %
                  (get_time(), iteration, negative_mi_value, disp_loss_value, pair_loss_value, total_loss_value, drift), flush=True)

        if iteration == 1 or iteration == args.Iteration:
            grid_path = output / (png_name % iteration)
            visible = (image_syn.detach() * diag_std + diag_mean).clamp(0, 1).expand(-1, channel, -1, -1).cpu()
            save_image(visible, grid_path, nrow=int(np.ceil(np.sqrt(num_syn))))

        if iteration == 1 or iteration % args.save_every == 0 or iteration == args.Iteration:
            save_checkpoint(iteration)

    print('DONE: %s' % result_path, flush=True)



if __name__ == '__main__':
    main()
