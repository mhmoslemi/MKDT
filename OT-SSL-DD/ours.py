"""Information-Theoretic Dataset Condensation: Optimizing synthetic pixels as Grouped IIC topological anchors.
Minimizes the negative mutual information between augmented views of real images assigned to grouped synthetic sub-prototypes in random feature spaces."""

import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image

from utils import DiffAugment, ParamDiffAug, evaluate_synset_SSL, get_dataset, get_eval_pool, get_network, get_time


def set_random_seed(seed):
    """Seed initialization/evaluation while retaining explicit objective RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _images_from_batch(batch):
    return batch[0] if isinstance(batch, (tuple, list)) else batch


def frozen_embedder(network):
    """Return a frozen random encoder; classifier parameters remain unused."""
    module = network.module if isinstance(network, nn.DataParallel) else network
    network.eval()
    for parameter in network.parameters():
        parameter.requires_grad_(False)
    return module.embed


def make_diff_augmenter(strategy, param):
    """Build a seeded differentiable augmenter with independent per-image draws."""
    cuda_devices = list(range(torch.cuda.device_count()))

    def augment(images, seed):
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(seed)
            return DiffAugment(images, strategy, seed=-1, param=param)

    return augment


def simclr_loss(features_u, features_v, temperature):
    """NT-Xent loss for two aligned batches of augmented synthetic images."""
    features = torch.cat((F.normalize(features_u, dim=1), F.normalize(features_v, dim=1)))
    logits = features @ features.t() / temperature
    logits.fill_diagonal_(-torch.inf)
    targets = (torch.arange(len(features), device=features.device) + len(features_u)) % len(features)
    return F.cross_entropy(logits, targets)


def barlow_twins_loss(features_u, features_v, off_diagonal_weight):
    """Barlow Twins cross-correlation loss for two aligned batches."""
    features_u = (features_u - features_u.mean(dim=0)) / (features_u.std(dim=0, unbiased=False) + 1e-8)
    features_v = (features_v - features_v.mean(dim=0)) / (features_v.std(dim=0, unbiased=False) + 1e-8)
    cross_correlation = features_u.t() @ features_v / features_u.shape[0]
    diagonal = torch.diagonal(cross_correlation)
    on_diagonal = (diagonal - 1).square().sum()
    off_diagonal = (cross_correlation - torch.diag(diagonal)).square().sum()
    return on_diagonal + off_diagonal_weight * off_diagonal


def encoder_train_loss(features_u, features_v, args):
    """Return the selected SSL objective for sampled-encoder adaptation."""
    if args.encoder_train_method == 'simclr':
        return simclr_loss(features_u, features_v, args.temperature)
    if args.encoder_train_method == 'barlowtwins':
        return barlow_twins_loss(features_u, features_v, args.barlow_lambda)
    if args.encoder_train_method == 'both':
        simclr = simclr_loss(features_u, features_v, args.temperature)
        barlow = barlow_twins_loss(features_u, features_v, args.barlow_lambda)
        return 0.5 * (simclr + barlow)
    raise ValueError('Unknown encoder training method: %s' % args.encoder_train_method)


def train_encoder_ssl(network, images, augmenter, args, objective_rng):
    """Briefly adapt a sampled encoder with SSL while keeping images detached."""
    if args.encoder_train_steps == 0:
        return 0.0
    module = network.module if isinstance(network, nn.DataParallel) else network
    network.train()
    for parameter in network.parameters():
        parameter.requires_grad_(True)
    with torch.no_grad():
        feature_dim = module.embed(images[:min(2, len(images))]).shape[1]
    projector = nn.Sequential(
        nn.Linear(feature_dim, args.projection_dim),
        nn.ReLU(),
        nn.Linear(args.projection_dim, args.projection_dim)
    ).to(images.device)
    optimizer = torch.optim.SGD(
        list(network.parameters()) + list(projector.parameters()),
        lr=args.lr_net,
        momentum=0.9,
        weight_decay=0.0005
    )
    losses = []
    detached_images = images.detach()
    batch_size = min(args.batch_train, len(detached_images))
    for _ in range(args.encoder_train_steps):
        indices = torch.randperm(len(detached_images), device=detached_images.device)[:batch_size]
        batch = detached_images[indices]
        view_u = augmenter(batch, int(objective_rng.integers(0, 2**31 - 1)))
        view_v = augmenter(batch, int(objective_rng.integers(0, 2**31 - 1)))
        features_u = projector(module.embed(view_u))
        features_v = projector(module.embed(view_v))
        loss = encoder_train_loss(features_u, features_v, args)
        if not torch.isfinite(loss):
            raise RuntimeError('Non-finite synthetic encoder warm-up loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(loss.detach())
    return torch.stack(losses).mean().item()


def total_variation_loss(img):
    """Calculates the Total Variation (TV) loss to encourage spatial smoothness and penalize high-frequency noise."""
    tv_h = torch.mean(torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :]))
    tv_w = torch.mean(torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1]))
    return tv_h + tv_w


def pixel_mse_loss(images, initial_images, channel_std):
    """Mean squared pixel drift in [0, 1] RGB units, with a fixed initial target."""
    return ((images - initial_images.detach()) * channel_std).square().mean()


@torch.no_grad()
def _group_fine_centers(centers, num_clusters, seed):
    """
    Partition M fine K-Means centers into K balanced semantic groups of size r = M // K.
    Returns a permutation of [0, ..., M-1] so that slice [k*r : (k+1)*r] contains
    the r fine sub-prototypes belonging to coarse cluster k.
    """
    M = centers.shape[0]
    K = min(num_clusters, M)
    if K >= M or M % K != 0:
        return torch.arange(M, device=centers.device)

    r = M // K
    norm_centers = F.normalize(centers, dim=1)

    # 1. Find K coarse centroids over the M fine centroids via K-Means
    rng = torch.Generator(device=centers.device).manual_seed(seed)
    coarse_indices = torch.randperm(M, generator=rng, device=centers.device)[:K]
    coarse_centers = norm_centers[coarse_indices].clone()

    for _ in range(50):
        dists = torch.cdist(norm_centers, coarse_centers)
        assigns = dists.argmin(dim=1)
        new_coarse = torch.stack([
            F.normalize(norm_centers[assigns == k].mean(dim=0, keepdim=True), dim=1).squeeze(0)
            if (assigns == k).any() else coarse_centers[k]
            for k in range(K)
        ])
        if torch.allclose(coarse_centers, new_coarse, atol=1e-4):
            break
        coarse_centers = new_coarse

    # 2. Balanced assignment: each of the K coarse clusters claims r closest fine centers
    dists = torch.cdist(coarse_centers, norm_centers)  # [K, M]
    available = torch.ones(M, dtype=torch.bool, device=centers.device)
    grouped_indices = torch.empty((K, r), dtype=torch.long, device=centers.device)

    # Greedily assign in r rounds so no coarse cluster gets starved
    for slot in range(r):
        # Order coarse clusters by their closest currently available fine center
        masked_dists = dists.clone()
        masked_dists[:, ~available] = float('inf')
        best_vals, _ = masked_dists.min(dim=1)
        cluster_order = best_vals.argsort()

        for k in cluster_order:
            row = dists[k].clone()
            row[~available] = float('inf')
            chosen = row.argmin()
            grouped_indices[k, slot] = chosen
            available[chosen] = False

    return grouped_indices.reshape(-1)


@torch.no_grad()
def deepcluster_initialize(
    dst_train, num_syn, num_clusters, device, seed, channel, num_classes, im_size, model_name='ConvNet'
):
    """Extract features with a random network, perform K-Means to find M synthetic prototypes,
    and organize them into K semantic groups of r = M // K sub-prototypes."""
    print(
        f'Initializing {num_syn} synthetic prototypes ({num_clusters} coarse clusters) '
        f'via DeepCluster K-Means on {model_name}...',
        flush=True
    )
    network = get_network(model_name, channel, num_classes, im_size, seed=seed).to(device)
    encoder = frozen_embedder(network)

    loader = torch.utils.data.DataLoader(dst_train, batch_size=1024, shuffle=False, num_workers=0)
    features_list, images_list = [], []

    for batch in loader:
        imgs = _images_from_batch(batch).to(device)
        feats = encoder(imgs)
        features_list.append(feats.cpu())
        images_list.append(imgs.cpu())

    features = torch.cat(features_list, dim=0)
    images = torch.cat(images_list, dim=0)

    N, D = features.shape
    M = num_syn

    # Initialize K-means centers
    rng = torch.Generator().manual_seed(seed)
    indices = torch.randperm(N, generator=rng)[:M]
    centers = features[indices].clone()

    # K-Means loop
    for _ in range(50):
        dists = torch.cdist(features, centers)
        assigns = dists.argmin(dim=1)
        new_centers = torch.stack([
            features[assigns == k].mean(dim=0) if (assigns == k).any() else centers[k]
            for k in range(M)
        ])
        if torch.allclose(centers, new_centers, atol=1e-4):
            break
        centers = new_centers

    # Assign initial synthetic images to the closest real image of each converged center
    dists = torch.cdist(centers, features)
    closest_indices = dists.argmin(dim=1)

    # Order the M prototypes into K coherent groups of size r = M // K
    perm = _group_fine_centers(centers, num_clusters=num_clusters, seed=seed)
    closest_indices = closest_indices[perm]
    initial_syn_images = images[closest_indices].clone()

    del network, encoder, features, dists, centers
    return initial_syn_images.to(device)


def information_theoretic_loss(features_u, features_v, features_S, tau, num_clusters):
    """
    Compute the negative Mutual Information (Grouped IIC loss) of cluster assignments.
    Each coarse cluster k in {1..K} is parameterized by r = M // K synthetic sub-prototypes.
    """
    # L2 normalize features
    u = F.normalize(features_u, dim=1)
    v = F.normalize(features_v, dim=1)
    S = F.normalize(features_S, dim=1)

    # Cosine similarities [Batch, M]
    sim_u = torch.mm(u, S.t())
    sim_v = torch.mm(v, S.t())

    # Fine-level soft assignments to the M synthetic sub-prototypes [Batch, M]
    p_u_fine = F.softmax(sim_u / tau, dim=1)
    p_v_fine = F.softmax(sim_v / tau, dim=1)

    B, M = p_u_fine.shape
    K = min(num_clusters, M)

    # Aggregate fine sub-prototype probabilities into K coarse cluster probabilities [Batch, K]
    if K < M and M % K == 0:
        r = M // K
        p_u = p_u_fine.view(B, K, r).sum(dim=2)
        p_v = p_v_fine.view(B, K, r).sum(dim=2)
    else:
        p_u = p_u_fine
        p_v = p_v_fine

    # Coarse joint probability matrix P [K, K]
    P = torch.mm(p_u.t(), p_v) / B

    # Symmetrize to enforce identical marginals
    P = 0.5 * (P + P.t())
    P = torch.clamp(P, min=1e-8)

    # Marginals [K, 1] and [1, K]
    P_i = P.sum(dim=1, keepdim=True)
    P_j = P.sum(dim=0, keepdim=True)

    # Mutual Information: \sum_{i,j=1}^K P_ij \log( P_ij / (P_i * P_j) )
    MI = (P * torch.log(P / (P_i * P_j))).sum()

    negative_mi = -MI
    return negative_mi


def main():
    parser = argparse.ArgumentParser(description='Information-Theoretic Dataset Condensation (Grouped DeepCluster + IIC)')

    # Data.
    parser.add_argument('--dataset', default='CIFAR10')
    parser.add_argument('--percentage', type=float, default=1.0)
    parser.add_argument('--data_path', default='/home/mmoslem3/scratch/data')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=0)

    # Information-Theoretic Objective.
    parser.add_argument('--Iteration', type=int, default=1000)
    parser.add_argument('--lr_img', type=float, default=0.005, help='Higher LR, decays via CosineAnnealing')
    parser.add_argument('--batch_real', type=int, default=512, help='Slightly noisier batch prevents deep proxy minima')
    parser.add_argument('--num_random_networks', type=int, default=5, help='Fresh samples from P_net per pixel update')
    parser.add_argument('--temperature', type=float, default=0.2, help='Softmax temperature for prototype assignment')
    parser.add_argument('--num_clusters', type=int, default=50, help='Number of coarse semantic clusters K (each cluster has r = M // K synthetic sub-prototypes)')
    parser.add_argument('--random_models', default='ConvNet', help='Comma-separated P_net support; forces geometric generalization')
    parser.add_argument('--tv_weight', type=float, default=0.01, help='Total Variation penalty to destroy adversarial high-frequencies')
    parser.add_argument('--pixel_mse_weight', type=float, default=0.0, help='Nonnegative weight for mean pixel MSE to initialization in [0,1] units; 0 disables it')
    parser.add_argument('--encoder_train_steps', type=int, default=2, help='SSL steps on the selected detached images before freezing each sampled encoder; 0 disables adaptation')
    parser.add_argument('--encoder_train_method', choices=['simclr', 'barlowtwins', 'barlow_twins', 'both'], default='simclr', help='SSL objective for sampled-encoder adaptation; both uses the mean of the SimCLR and Barlow Twins losses')
    parser.add_argument('--encoder_train_data', choices=['synthetic', 'real'], default='synthetic', help='Images used for the sampled encoder SSL warm-up')
    parser.add_argument('--distill_aug_strategy', default='color_crop_cutout_flip_scale_rotate')
    parser.add_argument('--distill_aug_mode', choices=['S', 'M'], default='M')

    # Downstream SSL evaluation.
    parser.add_argument('--model', default='ResNet18_cover')
    parser.add_argument('--lr_net', type=float, default=0.01)
    parser.add_argument('--batch_train', type=int, default=256)
    parser.add_argument('--ssl_method', default='simclr')
    parser.add_argument('--ssl_aug_strategy', default=None)
    parser.add_argument('--projection_dim', type=int, default=128)
    parser.add_argument('--barlow_lambda', type=float, default=0.005, help='Barlow Twins off-diagonal loss weight')
    parser.add_argument('--eval_mode', default='S')
    parser.add_argument('--num_eval', type=int, default=3)
    parser.add_argument('--epoch_eval_train', type=int, default=1200)
    parser.add_argument('--label_percentage', type=float, default=1.0)
    parser.add_argument('--epoch_linear_train', type=int, default=200)
    parser.add_argument('--lr_linear', type=float, default=0.1)
    parser.add_argument('--batch_linear', type=int, default=256)

    # Output.
    parser.add_argument('--save_path', default='ours')

    args = parser.parse_args()
    if args.encoder_train_method == 'barlow_twins':
        args.encoder_train_method = 'barlowtwins'
    random_models = [name.strip() for name in (args.random_models or args.model).split(',') if name.strip()]
    if args.ssl_aug_strategy is None:
        args.ssl_aug_strategy = args.distill_aug_strategy
    args.method = 'InformationTheoretic_Grouped_IIC'

    args.dsa_param = ParamDiffAug()
    args.dsa_param.aug_mode = 'S'
    distill_dsa_param = ParamDiffAug()
    distill_dsa_param.aug_mode = args.distill_aug_mode
    args.dsa = False
    set_random_seed(args.seed)

    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(args.data_path, exist_ok=True)
    channel, im_size, num_classes, _, mean, std, dst_train, _, testloader = get_dataset(args.dataset, args.data_path)
    num_syn = int(len(dst_train) * args.percentage / 100)
    effective_clusters = min(args.num_clusters, num_syn)
    sub_prototypes_per_cluster = num_syn // effective_clusters if num_syn % effective_clusters == 0 else 1

    # Grouped DeepCluster Initialization
    image_syn_init = deepcluster_initialize(
        dst_train,
        num_syn,
        effective_clusters,
        args.device,
        args.seed,
        channel,
        num_classes,
        im_size,
        model_name=random_models[0]
    )

    image_syn = image_syn_init.detach().requires_grad_(True)
    image_syn_initial = image_syn.detach().clone()

    diag_mean = torch.tensor(mean, device=args.device).view(1, channel, 1, 1)
    diag_std = torch.tensor(std, device=args.device).view(1, channel, 1, 1)
    pixel_min = -diag_mean / diag_std
    pixel_max = (1 - diag_mean) / diag_std

    def to_uint8(images):
        return ((images * diag_std + diag_mean) * 255 + 0.5).clamp(0, 255).to(torch.uint8)

    initial_uint8 = to_uint8(image_syn_initial)

    # Enable Shuffle for stochastic batch sampling during distillation
    real_loader = torch.utils.data.DataLoader(
        dst_train,
        batch_size=args.batch_real,
        shuffle=True,
        num_workers=0,
        pin_memory=args.device == 'cuda',
        drop_last=True
    )
    real_iter = iter(real_loader)

    augmenter = make_diff_augmenter(args.distill_aug_strategy, distill_dsa_param)
    objective_rng = np.random.default_rng(args.seed + 10_000)

    # Optimizer & Scheduler Setup
    optimizer_img = torch.optim.Adam([image_syn], lr=args.lr_img)
    scheduler_img = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_img, T_max=args.Iteration, eta_min=0.001)

    loss_history = []
    negative_mi_history = []
    pixel_mse_history = []
    total_loss_history = []
    encoder_ssl_history = []
    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model) if args.num_eval else []

    def evaluate(iteration):
        for model_index, model_eval in enumerate(model_eval_pool):
            print('-------------------------\nEvaluation\nmodel_train = %s, model_eval = %s, iteration = %d' % (args.model, model_eval, iteration))
            accuracies = []
            for eval_index in range(args.num_eval):
                eval_seed = args.seed + 20000 + eval_index
                set_random_seed(eval_seed)
                network = get_network(model_eval, channel, num_classes, im_size, seed=eval_seed).to(args.device)
                _, _, accuracy = evaluate_synset_SSL(eval_index, network, image_syn.detach().clone(), dst_train, testloader, args)
                accuracies.append(float(accuracy))
            print('Evaluate %d random %s, mean = %.4f std = %.4f\n-------------------------' % (len(accuracies), model_eval, np.mean(accuracies), np.std(accuracies)), flush=True)

    print('Hyper-parameters: \n', args.__dict__)
    print('P_net support: %s' % ', '.join(random_models), flush=True)
    print('Grouped Sub-Prototypes: M = %d total synthetic images -> K = %d coarse clusters (r = %d sub-prototypes per cluster).' %
          (num_syn, effective_clusters, sub_prototypes_per_cluster), flush=True)
    print('Each sampled encoder receives %d SSL step(s) using %s on detached %s images before freezing.' %
          (args.encoder_train_steps, args.encoder_train_method, args.encoder_train_data), flush=True)
    print('Objective: minimize mean(-MI) + %g * pixel_MSE_to_initial ([0,1] pixel units).' %
          (args.pixel_mse_weight,), flush=True)

    print('%s distillation begins' % get_time(), flush=True)

    for iteration in range(1, args.Iteration + 1):
        optimizer_img.zero_grad(set_to_none=True)
        sampled_losses = []
        sampled_negative_mi_losses = []
        sampled_models = []
        sampled_encoder_losses = []

        # Sample stochastic real batch
        try:
            batch = next(real_iter)
        except StopIteration:
            real_iter = iter(real_loader)
            batch = next(real_iter)

        real_imgs = _images_from_batch(batch).to(args.device)

        for _ in range(args.num_random_networks):
            model_name = random_models[int(objective_rng.integers(len(random_models)))]
            network_seed = int(objective_rng.integers(0, 2**31 - 1))
            network = get_network(model_name, channel, num_classes, im_size, seed=network_seed).to(args.device)
            encoder_train_images = image_syn if args.encoder_train_data == 'synthetic' else real_imgs
            encoder_ssl_loss = train_encoder_ssl(network, encoder_train_images, augmenter, args, objective_rng)
            encoder = frozen_embedder(network)

            # ASYMMETRIC AUGMENTATION: Draw independent augmentations for real views only
            view_u = augmenter(real_imgs, int(objective_rng.integers(0, 2**31 - 1)))
            view_v = augmenter(real_imgs, int(objective_rng.integers(0, 2**31 - 1)))

            with torch.no_grad():
                features_u = encoder(view_u)
                features_v = encoder(view_v)

            # Gradients flow ONLY into the synthetic pixels S
            features_S = encoder(image_syn)

            # Compute Grouped Negative Mutual Information
            negative_mi = information_theoretic_loss(
                features_u,
                features_v,
                features_S,
                tau=args.temperature,
                num_clusters=effective_clusters
            )
            network_loss = negative_mi

            # Gradient descent on negative MI maximizes mutual information.
            (network_loss / args.num_random_networks).backward()

            sampled_losses.append(network_loss.detach())
            sampled_negative_mi_losses.append(negative_mi.detach())
            sampled_encoder_losses.append(encoder_ssl_loss)
            sampled_models.append('%s(s=%d)' % (model_name, network_seed))
            del encoder, network, network_loss, negative_mi
            del features_u, features_v, features_S, view_u, view_v

        # Add the positive drift penalty ONCE per pixel update, independent of num_random_networks.
        pixel_mse = pixel_mse_loss(image_syn, image_syn_initial, diag_std)
        if args.pixel_mse_weight > 0:
            (args.pixel_mse_weight * pixel_mse).backward()

        objective = torch.stack(sampled_losses).mean().item()
        negative_mi_value = torch.stack(sampled_negative_mi_losses).mean().item()
        pixel_mse_value = pixel_mse.detach().item()
        total_loss_value = objective + args.pixel_mse_weight * pixel_mse_value
        encoder_ssl_value = float(np.mean(sampled_encoder_losses))

        optimizer_img.step()
        scheduler_img.step()

        with torch.no_grad():
            image_syn.copy_(torch.maximum(torch.minimum(image_syn, pixel_max), pixel_min))
            changed = (to_uint8(image_syn) != initial_uint8).float().mean().item() * 100
            drift = ((image_syn - image_syn_initial) * diag_std * 255).square().mean().sqrt().item()

        loss_history.append(objective)
        negative_mi_history.append(negative_mi_value)
        pixel_mse_history.append(pixel_mse_value)
        total_loss_history.append(total_loss_value)
        encoder_ssl_history.append(encoder_ssl_value)

        if iteration % 25 == 0:
            current_lr = scheduler_img.get_last_lr()[0]
            print(
                '%s iter = %05d, encoder_SSL = %.6f, neg_MI = %.6f, proxy_loss = %.6f, '
                'pixel_MSE = %.6f, total_loss = %.6f, pixel drift RMS = %.5f/255, '
                'PNG values changed = %.5f%%, lr = %.4f' % (
                    get_time(), iteration, encoder_ssl_value, negative_mi_value,
                    objective, pixel_mse_value, total_loss_value, drift, changed, current_lr
                ),
                flush=True
            )

        if iteration % 100 == 0 or iteration == args.Iteration:
            grid_path = os.path.join(
                args.save_path,
                'vis_IIC_%s_%s_%gpercent_iter%d_final.png' % (args.dataset, args.model, args.percentage, iteration)
            )
            visible = (image_syn.detach() * diag_std + diag_mean).clamp(0, 1).cpu()
            save_image(visible, grid_path, nrow=int(np.ceil(np.sqrt(num_syn))))

        if iteration % 100 == 0 or iteration == args.Iteration:
            result_path = os.path.join(
                args.save_path,
                'res_IIC_%s_%s_%gpercent_final.pt' % (args.dataset, args.model, args.percentage)
            )
            torch.save({
                'data': image_syn.detach().cpu(),
                'method': args.method,
                'iteration': args.Iteration,
                'loss_history': torch.tensor(loss_history, dtype=torch.float64),
                'negative_mi_history': torch.tensor(negative_mi_history, dtype=torch.float64),
                'pixel_mse_history': torch.tensor(pixel_mse_history, dtype=torch.float64),
                'total_loss_history': torch.tensor(total_loss_history, dtype=torch.float64),
                'encoder_ssl_history': torch.tensor(encoder_ssl_history, dtype=torch.float64),
                'seed': args.seed,
                'random_feature_distribution': {
                    'models': random_models,
                    'samples_per_update': args.num_random_networks,
                    'encoder_train_steps': args.encoder_train_steps,
                    'encoder_train_data': args.encoder_train_data,
                    'encoder_train_lr': args.lr_net,
                    'encoder_train_batch': args.batch_train,
                    'encoder_train_method': args.encoder_train_method,
                    'encoder_barlow_lambda': args.barlow_lambda
                },
                'iic_params': {
                    'tau': args.temperature,
                    'num_clusters': effective_clusters,
                    'sub_prototypes_per_cluster': sub_prototypes_per_cluster,
                    'strategy': args.distill_aug_strategy,
                    'batch_real': args.batch_real,
                    'tv_weight': args.tv_weight,
                    'pixel_mse_weight': args.pixel_mse_weight,
                    'pixel_mse_space': 'pixels_0_1'
                }
            }, result_path)
            print('Saved synthetic data to %s' % result_path, flush=True)


if __name__ == '__main__':
    main()