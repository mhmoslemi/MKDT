
"""Joint Augmented-View Dataset Condensation.
Matches the joint distributions of augmented real and synthetic view pairs
with multi-bandwidth product-RBF MMD in sampled encoder feature spaces.
"""

import argparse
import math
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
from dm_dsa import make_diff_augmenter as dm_augmenter
from dd_teachers import teacher_arguments, load_teacher

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
    """Return a frozen encoder; classifier parameters remain unused."""
    module = network.module if isinstance(network, nn.DataParallel) else network
    network.eval()
    for parameter in network.parameters():
        parameter.requires_grad_(False)
    return module.embed


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


def simclr_loss(features_u, features_v, temperature):
    """NT-Xent loss for two aligned batches of augmented images."""
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
    """Return the selected SSL objective for optional sampled-encoder adaptation."""
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
    """Optionally adapt an encoder while keeping its training images detached.

    No gradients are propagated through these optimizer steps. With the default
    encoder_train_steps=0, the feature distribution is independent of the pixels.
    """
    if args.encoder_train_steps == 0:
        return 0.0
    module = network.module if isinstance(network, nn.DataParallel) else network
    for parameter in network.parameters():
        parameter.requires_grad_(True)
    network.eval()
    with torch.no_grad():
        feature_dim = module.embed(images[:min(2, len(images))]).shape[1]
    network.train()
    projector = nn.Sequential(nn.Linear(feature_dim, args.projection_dim), nn.ReLU(), nn.Linear(args.projection_dim, args.projection_dim)).to(images.device)
    optimizer = torch.optim.SGD(list(network.parameters()) + list(projector.parameters()), lr=args.lr_net, momentum=0.9, weight_decay=0.0005)
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
            raise RuntimeError('Non-finite sampled-encoder warm-up loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(loss.detach())
    return torch.stack(losses).mean().item()


def total_variation_loss(img):
    """Encourage spatial smoothness; the caller supplies [0, 1] RGB units."""
    tv_h = torch.mean(torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :]))
    tv_w = torch.mean(torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1]))
    return tv_h + tv_w


def pixel_mse_loss(images, initial_images, channel_std):
    """Mean squared pixel drift in [0, 1] RGB units, with a fixed initial target."""
    return ((images - initial_images.detach()) * channel_std).square().mean()


@torch.no_grad()
def deepcluster_initialize(dst_train, num_syn, device, seed, channel, num_classes, im_size, model_name='ConvNet'):
    """Initialize synthetic images with real representatives of random-feature clusters."""
    print(f'Initializing {num_syn} synthetic images via random-feature K-Means on {model_name}...', flush=True)
    network = get_network(model_name, channel, num_classes, im_size, seed=seed).to(device)
    encoder = frozen_embedder(network)

    loader = torch.utils.data.DataLoader(dst_train, batch_size=1024, shuffle=False, num_workers=0)
    features_list, images_list = [], []

    for batch in loader:
        imgs = _images_from_batch(batch).to(device)
        features_list.append(encoder(imgs).cpu())
        images_list.append(imgs.cpu())

    features = torch.cat(features_list, dim=0)
    images = torch.cat(images_list, dim=0)
    N, D = features.shape
    M = num_syn

    rng = torch.Generator().manual_seed(seed)
    indices = torch.randperm(N, generator=rng)[:M]
    centers = features[indices].clone()

    for _ in range(5):
        dists = torch.cdist(features, centers)
        assigns = dists.argmin(dim=1)
        new_centers = torch.stack([
            features[assigns == k].mean(dim=0) if (assigns == k).any() else centers[k]
            for k in range(M)
        ])
        if torch.allclose(centers, new_centers, atol=1e-4):
            break
        centers = new_centers

    dists = torch.cdist(centers, features)
    closest_indices = dists.argmin(dim=1)
    initial_syn_images = images[closest_indices].clone()

    del network, encoder, features, dists, centers
    return initial_syn_images.to(device)


def pairwise_squared_distance(features_a, features_b):
    """Squared Euclidean distances without allocating [N, M, D] differences."""
    norm_a = features_a.square().sum(dim=1, keepdim=True)
    norm_b = features_b.square().sum(dim=1, keepdim=True).t()
    return (norm_a + norm_b - 2.0 * (features_a @ features_b.t())).clamp_min(0.0)


def multi_bandwidth_rbf(squared_distances, bandwidth_squared, bandwidth_scales):
    """Average RBF kernels using one detached, real-derived base bandwidth."""
    kernel = torch.zeros_like(squared_distances)
    for scale in bandwidth_scales:
        kernel = kernel + torch.exp(-squared_distances / (2.0 * bandwidth_squared * scale**2))
    return kernel / len(bandwidth_scales)


def within_kernel_mean(kernel, estimator):
    """Use all entries for the V-statistic or remove diagonals for the U-statistic."""
    if estimator == 'biased':
        return kernel.mean()
    count = kernel.shape[0]
    if count < 2:
        raise ValueError('Unbiased MMD requires at least two pairs in each batch')
    return (kernel.sum() - torch.diagonal(kernel).sum()) / (count * (count - 1))


def joint_view_distribution_loss(features_real_u, features_real_v, features_syn_u, features_syn_v, args):
    """MMD between joint augmented-view distributions in one frozen feature space.

    Each row contains BOTH views of the same source image. An RBF on their
    concatenation is a product kernel:

        k((u, v), (u', v')) = exp(-||u-u'||^2 / (2*sigma^2))
                            * exp(-||v-v'||^2 / (2*sigma^2)).

    Averaging these kernels over bandwidths compares joint distributions, not
    merely a sum of two marginal MMDs. Normalize each view separately, using
    the same cosine-feature convention as the original prototype objective.
    """
    real_u = F.normalize(features_real_u.detach(), dim=1)
    real_v = F.normalize(features_real_v.detach(), dim=1)
    syn_u = F.normalize(features_syn_u, dim=1)
    syn_v = F.normalize(features_syn_v, dim=1)

    real_pairs = torch.cat((real_u, real_v), dim=1)
    syn_pairs = torch.cat((syn_u, syn_v), dim=1)

    # Estimate a shared bandwidth from real pairs only. It is not optimized
    # through synthetic features and is reused in all three MMD terms.
    with torch.no_grad():
        dist_real_real = pairwise_squared_distance(real_pairs, real_pairs)
        count = min(args.mmd_bandwidth_samples, len(real_pairs))
        upper_indices = torch.triu_indices(count, count, offset=1, device=real_pairs.device)
        distances = dist_real_real[:count, :count][upper_indices[0], upper_indices[1]]
        distances = distances[distances > 1e-12]
        bandwidth_squared = distances.median() if distances.numel() else real_pairs.new_tensor(1.0)
        bandwidth_squared = bandwidth_squared.clamp_min(1e-6)
        kernel_real_real = multi_bandwidth_rbf(dist_real_real, bandwidth_squared, args.mmd_bandwidth_scales)
        real_real = within_kernel_mean(kernel_real_real, args.mmd_estimator)

    dist_syn_syn = pairwise_squared_distance(syn_pairs, syn_pairs)
    dist_real_syn = pairwise_squared_distance(real_pairs, syn_pairs)
    kernel_syn_syn = multi_bandwidth_rbf(dist_syn_syn, bandwidth_squared, args.mmd_bandwidth_scales)
    kernel_real_syn = multi_bandwidth_rbf(dist_real_syn, bandwidth_squared, args.mmd_bandwidth_scales)

    syn_syn = within_kernel_mean(kernel_syn_syn, args.mmd_estimator)
    real_syn = kernel_real_syn.mean()
    mmd_loss = real_real + syn_syn - 2.0 * real_syn

    # The unbiased estimate can be negative. Do not clamp it: doing so would
    # suppress its pixel gradients whenever the estimate falls below zero.
    return mmd_loss, real_real.detach(), syn_syn.detach(), real_syn.detach(), bandwidth_squared.sqrt().detach()



def main():
    parser = argparse.ArgumentParser(description='Joint Augmented-View Dataset Condensation (K-Means + Joint MMD)')

    # Data.
    parser.add_argument('--dataset', default='CIFAR10')
    parser.add_argument('--percentage', type=float, default=1.0)
    parser.add_argument('--data_path', default='/home/mmoslem3/scratch/data')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=0)

    # Joint Augmented-View Distribution Objective.
    parser.add_argument('--Iteration', type=int, default=1000)
    parser.add_argument('--lr_img', type=float, default=0.01)
    parser.add_argument('--batch_real', type=int, default=1024)
    parser.add_argument('--batch_syn', type=int, default=0, help='Synthetic source images per sampled encoder; 0 uses the full set')
    parser.add_argument('--num_random_networks', type=int, default=10, help='Fresh encoder samples per pixel update')
    parser.add_argument('--random_models', default='ConvNet', help='Comma-separated encoder architecture support')
    parser.add_argument('--mmd_bandwidths', default='0.5,1.0,2.0', help='Positive bandwidth multipliers around the real-pair median-distance bandwidth')
    parser.add_argument('--mmd_bandwidth_samples', type=int, default=512, help='Maximum number of real pairs used for the bandwidth estimate')
    parser.add_argument('--mmd_estimator', choices=['biased', 'unbiased'], default='biased', help='Biased: minibatch V-statistic; unbiased: off-diagonal U-statistic')
    parser.add_argument('--tv_weight', type=float, default=0.01, help='Weight of total variation in [0,1] RGB units')
    parser.add_argument('--pixel_mse_weight', type=float, default=0.0, help='Weight of pixel MSE to initialization in [0,1] RGB units')
    parser.add_argument('--encoder_train_steps', type=int, default=5, help='Optional detached SSL warm-up; 0 uses fresh untrained encoders')
    parser.add_argument('--encoder_train_method', choices=['simclr', 'barlowtwins', 'barlow_twins', 'both'], default='simclr')
    parser.add_argument('--encoder_train_data', choices=['synthetic', 'real'], default='synthetic')
    parser.add_argument('--distill_aug_strategy', '--dsa_strategy', '--dsa-strategy', default='color_crop_cutout_flip_scale_rotate')
    parser.add_argument('--dsa', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--distill_aug_mode', choices=['S', 'M'], default='M')

    # Downstream SSL evaluation.
    parser.add_argument('--model', default='ResNet18_cover')
    parser.add_argument('--lr_net', type=float, default=0.01)
    parser.add_argument('--batch_train', type=int, default=256)
    parser.add_argument('--ssl_method', default='simclr')
    parser.add_argument('--ssl_aug_strategy', default=None)
    parser.add_argument('--temperature', type=float, default=0.2, help='NT-Xent temperature for optional encoder warm-up and SSL evaluation')
    parser.add_argument('--projection_dim', type=int, default=128)
    parser.add_argument('--barlow_lambda', type=float, default=0.005)
    parser.add_argument('--eval_mode', default='S')
    parser.add_argument('--num_eval', type=int, default=3)
    parser.add_argument('--epoch_eval_train', type=int, default=1200)
    parser.add_argument('--label_percentage', type=float, default=1.0)
    parser.add_argument('--epoch_linear_train', type=int, default=200)
    parser.add_argument('--lr_linear', type=float, default=0.1)
    parser.add_argument('--batch_linear', type=int, default=256)

    # Output.
    parser.add_argument('--save_path', default='result_joint_view_mmd')

    teacher_arguments(parser)
    args = parser.parse_args()
    if args.teacher_source == 'pretrained':
        args.encoder_train_steps = 0
    if args.encoder_train_method == 'barlow_twins':
        args.encoder_train_method = 'barlowtwins'
    random_models = [name.strip() for name in (args.random_models or args.model).split(',') if name.strip()]
    try:
        args.mmd_bandwidth_scales = [float(value.strip()) for value in args.mmd_bandwidths.split(',')]
    except ValueError:
        parser.error('--mmd_bandwidths must contain comma-separated positive numbers')
    if not args.mmd_bandwidth_scales or any(not math.isfinite(value) or value <= 0 for value in args.mmd_bandwidth_scales):
        parser.error('--mmd_bandwidths must contain finite positive numbers')
    if not random_models:
        parser.error('At least one encoder architecture is required')
    if not 0 < args.percentage <= 100:
        parser.error('--percentage must be in (0, 100]')
    if args.Iteration < 1 or args.num_random_networks < 1:
        parser.error('--Iteration and --num_random_networks must be positive')
    if args.batch_real < 2 or args.mmd_bandwidth_samples < 2 or args.batch_syn < 0:
        parser.error('--batch_real and --mmd_bandwidth_samples must be >= 2; --batch_syn must be >= 0')
    if args.mmd_estimator == 'unbiased' and args.batch_syn == 1:
        parser.error('Unbiased MMD requires --batch_syn >= 2 or --batch_syn 0')
    if args.encoder_train_steps < 0 or args.batch_train < 2:
        parser.error('--encoder_train_steps must be nonnegative and --batch_train must be >= 2')
    if args.lr_img <= 0 or args.lr_net <= 0 or args.temperature <= 0:
        parser.error('Learning rates and --temperature must be positive')
    if args.tv_weight < 0 or args.pixel_mse_weight < 0:
        parser.error('Regularization weights must be nonnegative')
    if args.ssl_aug_strategy is None:
        args.ssl_aug_strategy = args.distill_aug_strategy
    args.method = 'JointViewMMD'

    args.dsa_param = ParamDiffAug()
    args.dsa_param.aug_mode = 'S'
    distill_dsa_param = ParamDiffAug()
    distill_dsa_param.aug_mode = 'S' if args.dsa else args.distill_aug_mode
    set_random_seed(args.seed)

    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(args.data_path, exist_ok=True)
    channel, im_size, num_classes, _, mean, std, dst_train, _, testloader = get_dataset(args.dataset, args.data_path)
    num_syn = int(len(dst_train) * args.percentage / 100)
    pretrained_teacher = load_teacher(args, mean, std, im_size)
    if len(dst_train) < 2 or num_syn < 1:
        parser.error('The dataset must contain at least two images and the requested synthetic set must be nonempty')
    if num_syn < 2 and (args.mmd_estimator == 'unbiased' or args.encoder_train_steps > 0):
        parser.error('Unbiased MMD and optional SSL warm-up require at least two synthetic images')

    # Keep the original random-feature K-Means initialization.
    image_syn_init = deepcluster_initialize(
        dst_train, num_syn, args.device, args.seed, channel, num_classes, im_size, model_name=random_models[0]
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

    real_loader = torch.utils.data.DataLoader(dst_train, batch_size=min(args.batch_real, len(dst_train)), shuffle=True, num_workers=0, pin_memory=args.device == 'cuda', drop_last=True)
    real_iter = iter(real_loader)
    augmenter = make_diff_augmenter(args.distill_aug_strategy, distill_dsa_param, args.dsa)
    objective_rng = np.random.default_rng(args.seed + 10_000)

    optimizer_img = torch.optim.Adam([image_syn], lr=args.lr_img)
    scheduler_img = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_img, T_max=args.Iteration, eta_min=args.lr_img)

    loss_history = []
    real_real_history = []
    syn_syn_history = []
    real_syn_history = []
    bandwidth_history = []
    tv_history = []
    pixel_mse_history = []
    total_loss_history = []
    encoder_ssl_history = []
    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model) if args.num_eval else []

    def evaluate(iteration):
        for model_eval in model_eval_pool:
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
    print('Each sampled encoder receives %d SSL step(s) using %s on detached %s images before freezing.' %
          (args.encoder_train_steps, args.encoder_train_method, args.encoder_train_data), flush=True)
    print('Objective: minimize mean(joint_view_MMD) + %g * TV + %g * pixel_MSE_to_initial ([0,1] RGB units).' %
          (args.tv_weight, args.pixel_mse_weight), flush=True)
    print('MMD: %s estimator; bandwidth scales = %s; synthetic batch = %d (0 means all).' %
          (args.mmd_estimator, args.mmd_bandwidth_scales, args.batch_syn), flush=True)
    if args.encoder_train_steps > 0 and args.encoder_train_data == 'synthetic':
        print('Synthetic warm-up uses detached adaptation; encoder-state dependence on the pixels is not differentiated.', flush=True)
    # evaluate(0)

    print('%s distillation begins' % get_time(), flush=True)

    for iteration in range(1, args.Iteration + 1):
        optimizer_img.zero_grad(set_to_none=True)
        sampled_losses = []
        sampled_real_real = []
        sampled_syn_syn = []
        sampled_real_syn = []
        sampled_bandwidths = []
        sampled_encoder_losses = []

        try:
            batch = next(real_iter)
        except StopIteration:
            real_iter = iter(real_loader)
            batch = next(real_iter)
        real_imgs = _images_from_batch(batch).to(args.device)

        for _ in range(args.num_random_networks):
            model_name = random_models[int(objective_rng.integers(len(random_models)))]
            network_seed = int(objective_rng.integers(0, 2**31 - 1))
            if pretrained_teacher is not None:
                network_seed = args.teacher_seed
            network = pretrained_teacher if pretrained_teacher is not None else get_network(model_name, channel, num_classes, im_size, seed=network_seed).to(args.device)
            encoder_train_images = image_syn if args.encoder_train_data == 'synthetic' else real_imgs
            encoder_ssl_loss = train_encoder_ssl(network, encoder_train_images, augmenter, args, objective_rng)
            encoder = frozen_embedder(network)

            # Each pair contains two independent augmentations of ONE source
            # image. DSA shares each view's draw across the two datasets.
            real_seed_u, real_seed_v = objective_rng.integers(0, 100000, size=2).tolist()
            with torch.no_grad():
                real_view_u = augmenter(real_imgs, real_seed_u)
                real_view_v = augmenter(real_imgs, real_seed_v)
                features_real_u = encoder(real_view_u)
                features_real_v = encoder(real_view_v)

            syn_batch_size = min(args.batch_syn, num_syn) if args.batch_syn else num_syn
            if syn_batch_size == num_syn:
                syn_imgs = image_syn
            else:
                indices = torch.as_tensor(objective_rng.choice(num_syn, size=syn_batch_size, replace=False), device=image_syn.device, dtype=torch.long)
                syn_imgs = image_syn[indices]

            # Gradients pass through BOTH synthetic augmentations and the
            # frozen encoder into the selected synthetic pixels.
            syn_seed_u, syn_seed_v = (real_seed_u, real_seed_v) if args.dsa else objective_rng.integers(0, 2**31 - 1, size=2).tolist()
            syn_view_u = augmenter(syn_imgs, syn_seed_u)
            syn_view_v = augmenter(syn_imgs, syn_seed_v)
            features_syn_u = encoder(syn_view_u)
            features_syn_v = encoder(syn_view_v)

            network_loss, real_real, syn_syn, real_syn, bandwidth = joint_view_distribution_loss(
                features_real_u, features_real_v, features_syn_u, features_syn_v, args
            )
            if not torch.isfinite(network_loss):
                raise RuntimeError('Non-finite joint-view MMD loss')

            (network_loss / args.num_random_networks).backward()
            sampled_losses.append(network_loss.detach())
            sampled_real_real.append(real_real)
            sampled_syn_syn.append(syn_syn)
            sampled_real_syn.append(real_syn)
            sampled_bandwidths.append(bandwidth)
            sampled_encoder_losses.append(encoder_ssl_loss)

            del encoder, network, network_loss
            del features_real_u, features_real_v, features_syn_u, features_syn_v
            del real_view_u, real_view_v, syn_view_u, syn_view_v, syn_imgs

        # Apply regularization ONCE per pixel update, independent of the
        # number of sampled encoders. Both penalties use [0,1] RGB units.
        pixel_mse = pixel_mse_loss(image_syn, image_syn_initial, diag_std)
        tv = total_variation_loss(image_syn * diag_std + diag_mean)
        if args.tv_weight > 0 or args.pixel_mse_weight > 0:
            regularizer = args.tv_weight * tv + args.pixel_mse_weight * pixel_mse
            if not torch.isfinite(regularizer):
                raise RuntimeError('Non-finite pixel regularization loss')
            regularizer.backward()

        objective = torch.stack(sampled_losses).mean().item()
        real_real_value = torch.stack(sampled_real_real).mean().item()
        syn_syn_value = torch.stack(sampled_syn_syn).mean().item()
        real_syn_value = torch.stack(sampled_real_syn).mean().item()
        bandwidth_value = torch.stack(sampled_bandwidths).mean().item()
        tv_value = tv.detach().item()
        pixel_mse_value = pixel_mse.detach().item()
        total_loss_value = objective + args.tv_weight * tv_value + args.pixel_mse_weight * pixel_mse_value
        encoder_ssl_value = float(np.mean(sampled_encoder_losses))
        optimizer_img.step()
        scheduler_img.step()

        with torch.no_grad():
            image_syn.copy_(torch.maximum(torch.minimum(image_syn, pixel_max), pixel_min))
            changed = (to_uint8(image_syn) != initial_uint8).float().mean().item() * 100
            drift = ((image_syn - image_syn_initial) * diag_std * 255).square().mean().sqrt().item()

        loss_history.append(objective)
        real_real_history.append(real_real_value)
        syn_syn_history.append(syn_syn_value)
        real_syn_history.append(real_syn_value)
        bandwidth_history.append(bandwidth_value)
        tv_history.append(tv_value)
        pixel_mse_history.append(pixel_mse_value)
        total_loss_history.append(total_loss_value)
        encoder_ssl_history.append(encoder_ssl_value)

        if iteration % 25 == 0:
            current_lr = scheduler_img.get_last_lr()[0]
            print('%s iter = %05d, encoder_SSL = %.6f, joint_MMD = %.6f, real_real = %.6f, syn_syn = %.6f, real_syn = %.6f, bandwidth = %.6f, TV = %.6f, pixel_MSE = %.6f, total_loss = %.6f, pixel drift RMS = %.5f/255, PNG values changed = %.5f%%, lr = %.4f' %
                  (get_time(), iteration, encoder_ssl_value, objective, real_real_value, syn_syn_value, real_syn_value, bandwidth_value, tv_value, pixel_mse_value, total_loss_value, drift, changed, current_lr), flush=True)

        # if iteration % 100 == 0:
        #     evaluate(iteration)

        if iteration % 100 == 0 or iteration == args.Iteration:
            grid_path = os.path.join(args.save_path, 'vis_JointViewMMD_%s_%s_%gpercent_iter%d_final.png' % (args.dataset, args.model, args.percentage, iteration))
            visible = (image_syn.detach() * diag_std + diag_mean).clamp(0, 1).cpu()
            save_image(visible, grid_path, nrow=int(np.ceil(np.sqrt(num_syn))))

        if iteration % 200 == 0 or iteration == args.Iteration:
            result_path = os.path.join(args.save_path, 'res_JointViewMMD_%s_%s_%gpercent_final.pt' % (args.dataset, args.model, args.percentage))
            torch.save({
                'data': image_syn.detach().cpu(),
                'method': args.method,
                'iteration': iteration,
                'dataset': args.dataset, 'percentage': args.percentage, 'normalization': [mean, std],
                'dsa': args.dsa, 'teacher_source': args.teacher_source, 'teacher_seed': args.teacher_seed,
                'loss_history': torch.tensor(loss_history, dtype=torch.float64),
                'real_real_history': torch.tensor(real_real_history, dtype=torch.float64),
                'syn_syn_history': torch.tensor(syn_syn_history, dtype=torch.float64),
                'real_syn_history': torch.tensor(real_syn_history, dtype=torch.float64),
                'bandwidth_history': torch.tensor(bandwidth_history, dtype=torch.float64),
                'tv_history': torch.tensor(tv_history, dtype=torch.float64),
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
                    'encoder_barlow_lambda': args.barlow_lambda,
                    'projection_dim': args.projection_dim,
                    'temperature': args.temperature,
                },
                'mmd_params': {
                    'kernel': 'mixture_of_product_rbf',
                    'estimator': args.mmd_estimator,
                    'bandwidth_scales': args.mmd_bandwidth_scales,
                    'bandwidth_samples': args.mmd_bandwidth_samples,
                    'bandwidth_source': 'detached_real_joint_pairs',
                    'feature_normalization': 'per_view_l2',
                    'strategy': args.distill_aug_strategy,
                    'aug_mode': args.distill_aug_mode,
                    'batch_real': min(args.batch_real, len(dst_train)),
                    'batch_syn': args.batch_syn,
                    'tv_weight': args.tv_weight,
                    'pixel_mse_weight': args.pixel_mse_weight,
                    'regularizer_space': 'pixels_0_1',
                },
            }, result_path)
            print('Saved synthetic data to %s' % result_path, flush=True)


if __name__ == '__main__':
    main()

# """Information-Theoretic Dataset Condensation: Optimizing synthetic pixels as Grouped IIC topological anchors.
# Minimizes the negative mutual information between augmented views of real images assigned to grouped synthetic sub-prototypes in random feature spaces."""

# import argparse
# import os
# import random

# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torchvision.utils import save_image

# from utils import DiffAugment, ParamDiffAug, evaluate_synset_SSL, get_dataset, get_eval_pool, get_network, get_time


# def set_random_seed(seed):
#     """Seed initialization/evaluation while retaining explicit objective RNGs."""
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)


# def _images_from_batch(batch):
#     return batch[0] if isinstance(batch, (tuple, list)) else batch


# def frozen_embedder(network):
#     """Return a frozen random encoder; classifier parameters remain unused."""
#     module = network.module if isinstance(network, nn.DataParallel) else network
#     network.eval()
#     for parameter in network.parameters():
#         parameter.requires_grad_(False)
#     return module.embed


# def make_diff_augmenter(strategy, param):
#     """Build a seeded differentiable augmenter with independent per-image draws."""
#     cuda_devices = list(range(torch.cuda.device_count()))

#     def augment(images, seed):
#         with torch.random.fork_rng(devices=cuda_devices):
#             torch.manual_seed(seed)
#             return DiffAugment(images, strategy, seed=-1, param=param)

#     return augment


# def simclr_loss(features_u, features_v, temperature):
#     """NT-Xent loss for two aligned batches of augmented synthetic images."""
#     features = torch.cat((F.normalize(features_u, dim=1), F.normalize(features_v, dim=1)))
#     logits = features @ features.t() / temperature
#     logits.fill_diagonal_(-torch.inf)
#     targets = (torch.arange(len(features), device=features.device) + len(features_u)) % len(features)
#     return F.cross_entropy(logits, targets)


# def barlow_twins_loss(features_u, features_v, off_diagonal_weight):
#     """Barlow Twins cross-correlation loss for two aligned batches."""
#     features_u = (features_u - features_u.mean(dim=0)) / (features_u.std(dim=0, unbiased=False) + 1e-8)
#     features_v = (features_v - features_v.mean(dim=0)) / (features_v.std(dim=0, unbiased=False) + 1e-8)
#     cross_correlation = features_u.t() @ features_v / features_u.shape[0]
#     diagonal = torch.diagonal(cross_correlation)
#     on_diagonal = (diagonal - 1).square().sum()
#     off_diagonal = (cross_correlation - torch.diag(diagonal)).square().sum()
#     return on_diagonal + off_diagonal_weight * off_diagonal


# def encoder_train_loss(features_u, features_v, args):
#     """Return the selected SSL objective for sampled-encoder adaptation."""
#     if args.encoder_train_method == 'simclr':
#         return simclr_loss(features_u, features_v, args.temperature)
#     if args.encoder_train_method == 'barlowtwins':
#         return barlow_twins_loss(features_u, features_v, args.barlow_lambda)
#     if args.encoder_train_method == 'both':
#         simclr = simclr_loss(features_u, features_v, args.temperature)
#         barlow = barlow_twins_loss(features_u, features_v, args.barlow_lambda)
#         return 0.5 * (simclr + barlow)
#     raise ValueError('Unknown encoder training method: %s' % args.encoder_train_method)


# def train_encoder_ssl(network, images, augmenter, args, objective_rng):
#     """Briefly adapt a sampled encoder with SSL while keeping images detached."""
#     if args.encoder_train_steps == 0:
#         return 0.0
#     module = network.module if isinstance(network, nn.DataParallel) else network
#     network.train()
#     for parameter in network.parameters():
#         parameter.requires_grad_(True)
#     with torch.no_grad():
#         feature_dim = module.embed(images[:min(2, len(images))]).shape[1]
#     projector = nn.Sequential(
#         nn.Linear(feature_dim, args.projection_dim),
#         nn.ReLU(),
#         nn.Linear(args.projection_dim, args.projection_dim)
#     ).to(images.device)
#     optimizer = torch.optim.SGD(
#         list(network.parameters()) + list(projector.parameters()),
#         lr=args.lr_net,
#         momentum=0.9,
#         weight_decay=0.0005
#     )
#     losses = []
#     detached_images = images.detach()
#     batch_size = min(args.batch_train, len(detached_images))
#     for _ in range(args.encoder_train_steps):
#         indices = torch.randperm(len(detached_images), device=detached_images.device)[:batch_size]
#         batch = detached_images[indices]
#         view_u = augmenter(batch, int(objective_rng.integers(0, 2**31 - 1)))
#         view_v = augmenter(batch, int(objective_rng.integers(0, 2**31 - 1)))
#         features_u = projector(module.embed(view_u))
#         features_v = projector(module.embed(view_v))
#         loss = encoder_train_loss(features_u, features_v, args)
#         if not torch.isfinite(loss):
#             raise RuntimeError('Non-finite synthetic encoder warm-up loss')
#         optimizer.zero_grad(set_to_none=True)
#         loss.backward()
#         optimizer.step()
#         losses.append(loss.detach())
#     return torch.stack(losses).mean().item()


# def total_variation_loss(img):
#     """Calculates the Total Variation (TV) loss to encourage spatial smoothness and penalize high-frequency noise."""
#     tv_h = torch.mean(torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :]))
#     tv_w = torch.mean(torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1]))
#     return tv_h + tv_w


# def pixel_mse_loss(images, initial_images, channel_std):
#     """Mean squared pixel drift in [0, 1] RGB units, with a fixed initial target."""
#     return ((images - initial_images.detach()) * channel_std).square().mean()


# @torch.no_grad()
# def _group_fine_centers(centers, num_clusters, seed):
#     """
#     Partition M fine K-Means centers into K balanced semantic groups of size r = M // K.
#     Returns a permutation of [0, ..., M-1] so that slice [k*r : (k+1)*r] contains
#     the r fine sub-prototypes belonging to coarse cluster k.
#     """
#     M = centers.shape[0]
#     K = min(num_clusters, M)
#     if K >= M or M % K != 0:
#         return torch.arange(M, device=centers.device)

#     r = M // K
#     norm_centers = F.normalize(centers, dim=1)

#     # 1. Find K coarse centroids over the M fine centroids via K-Means
#     rng = torch.Generator(device=centers.device).manual_seed(seed)
#     coarse_indices = torch.randperm(M, generator=rng, device=centers.device)[:K]
#     coarse_centers = norm_centers[coarse_indices].clone()

#     for _ in range(50):
#         dists = torch.cdist(norm_centers, coarse_centers)
#         assigns = dists.argmin(dim=1)
#         new_coarse = torch.stack([
#             F.normalize(norm_centers[assigns == k].mean(dim=0, keepdim=True), dim=1).squeeze(0)
#             if (assigns == k).any() else coarse_centers[k]
#             for k in range(K)
#         ])
#         if torch.allclose(coarse_centers, new_coarse, atol=1e-4):
#             break
#         coarse_centers = new_coarse

#     # 2. Balanced assignment: each of the K coarse clusters claims r closest fine centers
#     dists = torch.cdist(coarse_centers, norm_centers)  # [K, M]
#     available = torch.ones(M, dtype=torch.bool, device=centers.device)
#     grouped_indices = torch.empty((K, r), dtype=torch.long, device=centers.device)

#     # Greedily assign in r rounds so no coarse cluster gets starved
#     for slot in range(r):
#         # Order coarse clusters by their closest currently available fine center
#         masked_dists = dists.clone()
#         masked_dists[:, ~available] = float('inf')
#         best_vals, _ = masked_dists.min(dim=1)
#         cluster_order = best_vals.argsort()

#         for k in cluster_order:
#             row = dists[k].clone()
#             row[~available] = float('inf')
#             chosen = row.argmin()
#             grouped_indices[k, slot] = chosen
#             available[chosen] = False

#     return grouped_indices.reshape(-1)


# @torch.no_grad()
# def deepcluster_initialize(
#     dst_train, num_syn, num_clusters, device, seed, channel, num_classes, im_size, model_name='ConvNet'
# ):
#     """Extract features with a random network, perform K-Means to find M synthetic prototypes,
#     and organize them into K semantic groups of r = M // K sub-prototypes."""
#     print(
#         f'Initializing {num_syn} synthetic prototypes ({num_clusters} coarse clusters) '
#         f'via DeepCluster K-Means on {model_name}...',
#         flush=True
#     )
#     network = get_network(model_name, channel, num_classes, im_size, seed=seed).to(device)
#     encoder = frozen_embedder(network)

#     loader = torch.utils.data.DataLoader(dst_train, batch_size=1024, shuffle=False, num_workers=0)
#     features_list, images_list = [], []

#     for batch in loader:
#         imgs = _images_from_batch(batch).to(device)
#         feats = encoder(imgs)
#         features_list.append(feats.cpu())
#         images_list.append(imgs.cpu())

#     features = torch.cat(features_list, dim=0)
#     images = torch.cat(images_list, dim=0)

#     N, D = features.shape
#     M = num_syn

#     # Initialize K-means centers
#     rng = torch.Generator().manual_seed(seed)
#     indices = torch.randperm(N, generator=rng)[:M]
#     centers = features[indices].clone()

#     # K-Means loop
#     for _ in range(50):
#         dists = torch.cdist(features, centers)
#         assigns = dists.argmin(dim=1)
#         new_centers = torch.stack([
#             features[assigns == k].mean(dim=0) if (assigns == k).any() else centers[k]
#             for k in range(M)
#         ])
#         if torch.allclose(centers, new_centers, atol=1e-4):
#             break
#         centers = new_centers

#     # Assign initial synthetic images to the closest real image of each converged center
#     dists = torch.cdist(centers, features)
#     closest_indices = dists.argmin(dim=1)

#     # Order the M prototypes into K coherent groups of size r = M // K
#     perm = _group_fine_centers(centers, num_clusters=num_clusters, seed=seed)
#     closest_indices = closest_indices[perm]
#     initial_syn_images = images[closest_indices].clone()

#     del network, encoder, features, dists, centers
#     return initial_syn_images.to(device)


# def information_theoretic_loss(features_u, features_v, features_S, tau, num_clusters):
#     """
#     Compute the negative Mutual Information (Grouped IIC loss) of cluster assignments.
#     Each coarse cluster k in {1..K} is parameterized by r = M // K synthetic sub-prototypes.
#     """
#     # L2 normalize features
#     u = F.normalize(features_u, dim=1)
#     v = F.normalize(features_v, dim=1)
#     S = F.normalize(features_S, dim=1)

#     # Cosine similarities [Batch, M]
#     sim_u = torch.mm(u, S.t())
#     sim_v = torch.mm(v, S.t())

#     # Fine-level soft assignments to the M synthetic sub-prototypes [Batch, M]
#     p_u_fine = F.softmax(sim_u / tau, dim=1)
#     p_v_fine = F.softmax(sim_v / tau, dim=1)

#     B, M = p_u_fine.shape
#     K = min(num_clusters, M)

#     # Aggregate fine sub-prototype probabilities into K coarse cluster probabilities [Batch, K]
#     if K < M and M % K == 0:
#         r = M // K
#         p_u = p_u_fine.view(B, K, r).sum(dim=2)
#         p_v = p_v_fine.view(B, K, r).sum(dim=2)
#     else:
#         p_u = p_u_fine
#         p_v = p_v_fine

#     # Coarse joint probability matrix P [K, K]
#     P = torch.mm(p_u.t(), p_v) / B

#     # Symmetrize to enforce identical marginals
#     P = 0.5 * (P + P.t())
#     P = torch.clamp(P, min=1e-8)

#     # Marginals [K, 1] and [1, K]
#     P_i = P.sum(dim=1, keepdim=True)
#     P_j = P.sum(dim=0, keepdim=True)

#     # Mutual Information: \sum_{i,j=1}^K P_ij \log( P_ij / (P_i * P_j) )
#     MI = (P * torch.log(P / (P_i * P_j))).sum()

#     negative_mi = -MI
#     return negative_mi


# def main():
#     parser = argparse.ArgumentParser(description='Information-Theoretic Dataset Condensation (Grouped DeepCluster + IIC)')

#     # Data.
#     parser.add_argument('--dataset', default='CIFAR10')
#     parser.add_argument('--percentage', type=float, default=1.0)
#     parser.add_argument('--data_path', default='/home/mmoslem3/scratch/data')
#     parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda' if torch.cuda.is_available() else 'cpu')
#     parser.add_argument('--seed', type=int, default=0)

#     # Information-Theoretic Objective.
#     parser.add_argument('--Iteration', type=int, default=1000)
#     parser.add_argument('--lr_img', type=float, default=0.005, help='Higher LR, decays via CosineAnnealing')
#     parser.add_argument('--batch_real', type=int, default=512, help='Slightly noisier batch prevents deep proxy minima')
#     parser.add_argument('--num_random_networks', type=int, default=5, help='Fresh samples from P_net per pixel update')
#     parser.add_argument('--temperature', type=float, default=0.2, help='Softmax temperature for prototype assignment')
#     parser.add_argument('--num_clusters', type=int, default=50, help='Number of coarse semantic clusters K (each cluster has r = M // K synthetic sub-prototypes)')
#     parser.add_argument('--random_models', default='ConvNet', help='Comma-separated P_net support; forces geometric generalization')
#     parser.add_argument('--tv_weight', type=float, default=0.01, help='Total Variation penalty to destroy adversarial high-frequencies')
#     parser.add_argument('--pixel_mse_weight', type=float, default=0.0, help='Nonnegative weight for mean pixel MSE to initialization in [0,1] units; 0 disables it')
#     parser.add_argument('--encoder_train_steps', type=int, default=2, help='SSL steps on the selected detached images before freezing each sampled encoder; 0 disables adaptation')
#     parser.add_argument('--encoder_train_method', choices=['simclr', 'barlowtwins', 'barlow_twins', 'both'], default='simclr', help='SSL objective for sampled-encoder adaptation; both uses the mean of the SimCLR and Barlow Twins losses')
#     parser.add_argument('--encoder_train_data', choices=['synthetic', 'real'], default='synthetic', help='Images used for the sampled encoder SSL warm-up')
#     parser.add_argument('--distill_aug_strategy', default='color_crop_cutout_flip_scale_rotate')
#     parser.add_argument('--distill_aug_mode', choices=['S', 'M'], default='M')

#     # Downstream SSL evaluation.
#     parser.add_argument('--model', default='ResNet18_cover')
#     parser.add_argument('--lr_net', type=float, default=0.01)
#     parser.add_argument('--batch_train', type=int, default=256)
#     parser.add_argument('--ssl_method', default='simclr')
#     parser.add_argument('--ssl_aug_strategy', default=None)
#     parser.add_argument('--projection_dim', type=int, default=128)
#     parser.add_argument('--barlow_lambda', type=float, default=0.005, help='Barlow Twins off-diagonal loss weight')
#     parser.add_argument('--eval_mode', default='S')
#     parser.add_argument('--num_eval', type=int, default=3)
#     parser.add_argument('--epoch_eval_train', type=int, default=1200)
#     parser.add_argument('--label_percentage', type=float, default=1.0)
#     parser.add_argument('--epoch_linear_train', type=int, default=200)
#     parser.add_argument('--lr_linear', type=float, default=0.1)
#     parser.add_argument('--batch_linear', type=int, default=256)

#     # Output.
#     parser.add_argument('--save_path', default='ours')

#     args = parser.parse_args()
#     if args.encoder_train_method == 'barlow_twins':
#         args.encoder_train_method = 'barlowtwins'
#     random_models = [name.strip() for name in (args.random_models or args.model).split(',') if name.strip()]
#     if args.ssl_aug_strategy is None:
#         args.ssl_aug_strategy = args.distill_aug_strategy
#     args.method = 'InformationTheoretic_Grouped_IIC'

#     args.dsa_param = ParamDiffAug()
#     args.dsa_param.aug_mode = 'S'
#     distill_dsa_param = ParamDiffAug()
#     distill_dsa_param.aug_mode = args.distill_aug_mode
#     args.dsa = False
#     set_random_seed(args.seed)

#     os.makedirs(args.save_path, exist_ok=True)
#     os.makedirs(args.data_path, exist_ok=True)
#     channel, im_size, num_classes, _, mean, std, dst_train, _, testloader = get_dataset(args.dataset, args.data_path)
#     num_syn = int(len(dst_train) * args.percentage / 100)
#     effective_clusters = min(args.num_clusters, num_syn)
#     sub_prototypes_per_cluster = num_syn // effective_clusters if num_syn % effective_clusters == 0 else 1

#     # Grouped DeepCluster Initialization
#     image_syn_init = deepcluster_initialize(
#         dst_train,
#         num_syn,
#         effective_clusters,
#         args.device,
#         args.seed,
#         channel,
#         num_classes,
#         im_size,
#         model_name=random_models[0]
#     )

#     image_syn = image_syn_init.detach().requires_grad_(True)
#     image_syn_initial = image_syn.detach().clone()

#     diag_mean = torch.tensor(mean, device=args.device).view(1, channel, 1, 1)
#     diag_std = torch.tensor(std, device=args.device).view(1, channel, 1, 1)
#     pixel_min = -diag_mean / diag_std
#     pixel_max = (1 - diag_mean) / diag_std

#     def to_uint8(images):
#         return ((images * diag_std + diag_mean) * 255 + 0.5).clamp(0, 255).to(torch.uint8)

#     initial_uint8 = to_uint8(image_syn_initial)

#     # Enable Shuffle for stochastic batch sampling during distillation
#     real_loader = torch.utils.data.DataLoader(
#         dst_train,
#         batch_size=args.batch_real,
#         shuffle=True,
#         num_workers=0,
#         pin_memory=args.device == 'cuda',
#         drop_last=True
#     )
#     real_iter = iter(real_loader)

#     augmenter = make_diff_augmenter(args.distill_aug_strategy, distill_dsa_param)
#     objective_rng = np.random.default_rng(args.seed + 10_000)

#     # Optimizer & Scheduler Setup
#     optimizer_img = torch.optim.Adam([image_syn], lr=args.lr_img)
#     scheduler_img = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_img, T_max=args.Iteration, eta_min=0.001)

#     loss_history = []
#     negative_mi_history = []
#     pixel_mse_history = []
#     total_loss_history = []
#     encoder_ssl_history = []
#     model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model) if args.num_eval else []

#     def evaluate(iteration):
#         for model_index, model_eval in enumerate(model_eval_pool):
#             print('-------------------------\nEvaluation\nmodel_train = %s, model_eval = %s, iteration = %d' % (args.model, model_eval, iteration))
#             accuracies = []
#             for eval_index in range(args.num_eval):
#                 eval_seed = args.seed + 20000 + eval_index
#                 set_random_seed(eval_seed)
#                 network = get_network(model_eval, channel, num_classes, im_size, seed=eval_seed).to(args.device)
#                 _, _, accuracy = evaluate_synset_SSL(eval_index, network, image_syn.detach().clone(), dst_train, testloader, args)
#                 accuracies.append(float(accuracy))
#             print('Evaluate %d random %s, mean = %.4f std = %.4f\n-------------------------' % (len(accuracies), model_eval, np.mean(accuracies), np.std(accuracies)), flush=True)

#     print('Hyper-parameters: \n', args.__dict__)
#     print('P_net support: %s' % ', '.join(random_models), flush=True)
#     print('Grouped Sub-Prototypes: M = %d total synthetic images -> K = %d coarse clusters (r = %d sub-prototypes per cluster).' %
#           (num_syn, effective_clusters, sub_prototypes_per_cluster), flush=True)
#     print('Each sampled encoder receives %d SSL step(s) using %s on detached %s images before freezing.' %
#           (args.encoder_train_steps, args.encoder_train_method, args.encoder_train_data), flush=True)
#     print('Objective: minimize mean(-MI) + %g * pixel_MSE_to_initial ([0,1] pixel units).' %
#           (args.pixel_mse_weight,), flush=True)

#     print('%s distillation begins' % get_time(), flush=True)

#     for iteration in range(1, args.Iteration + 1):
#         optimizer_img.zero_grad(set_to_none=True)
#         sampled_losses = []
#         sampled_negative_mi_losses = []
#         sampled_models = []
#         sampled_encoder_losses = []

#         # Sample stochastic real batch
#         try:
#             batch = next(real_iter)
#         except StopIteration:
#             real_iter = iter(real_loader)
#             batch = next(real_iter)

#         real_imgs = _images_from_batch(batch).to(args.device)

#         for _ in range(args.num_random_networks):
#             model_name = random_models[int(objective_rng.integers(len(random_models)))]
#             network_seed = int(objective_rng.integers(0, 2**31 - 1))
#             network = get_network(model_name, channel, num_classes, im_size, seed=network_seed).to(args.device)
#             encoder_train_images = image_syn if args.encoder_train_data == 'synthetic' else real_imgs
#             encoder_ssl_loss = train_encoder_ssl(network, encoder_train_images, augmenter, args, objective_rng)
#             encoder = frozen_embedder(network)

#             # ASYMMETRIC AUGMENTATION: Draw independent augmentations for real views only
#             view_u = augmenter(real_imgs, int(objective_rng.integers(0, 2**31 - 1)))
#             view_v = augmenter(real_imgs, int(objective_rng.integers(0, 2**31 - 1)))

#             with torch.no_grad():
#                 features_u = encoder(view_u)
#                 features_v = encoder(view_v)

#             # Gradients flow ONLY into the synthetic pixels S
#             features_S = encoder(image_syn)

#             # Compute Grouped Negative Mutual Information
#             negative_mi = information_theoretic_loss(
#                 features_u,
#                 features_v,
#                 features_S,
#                 tau=args.temperature,
#                 num_clusters=effective_clusters
#             )
#             network_loss = negative_mi

#             # Gradient descent on negative MI maximizes mutual information.
#             (network_loss / args.num_random_networks).backward()

#             sampled_losses.append(network_loss.detach())
#             sampled_negative_mi_losses.append(negative_mi.detach())
#             sampled_encoder_losses.append(encoder_ssl_loss)
#             sampled_models.append('%s(s=%d)' % (model_name, network_seed))
#             del encoder, network, network_loss, negative_mi
#             del features_u, features_v, features_S, view_u, view_v

#         # Add the positive drift penalty ONCE per pixel update, independent of num_random_networks.
#         pixel_mse = pixel_mse_loss(image_syn, image_syn_initial, diag_std)
#         if args.pixel_mse_weight > 0:
#             (args.pixel_mse_weight * pixel_mse).backward()

#         objective = torch.stack(sampled_losses).mean().item()
#         negative_mi_value = torch.stack(sampled_negative_mi_losses).mean().item()
#         pixel_mse_value = pixel_mse.detach().item()
#         total_loss_value = objective + args.pixel_mse_weight * pixel_mse_value
#         encoder_ssl_value = float(np.mean(sampled_encoder_losses))

#         optimizer_img.step()
#         scheduler_img.step()

#         with torch.no_grad():
#             image_syn.copy_(torch.maximum(torch.minimum(image_syn, pixel_max), pixel_min))
#             changed = (to_uint8(image_syn) != initial_uint8).float().mean().item() * 100
#             drift = ((image_syn - image_syn_initial) * diag_std * 255).square().mean().sqrt().item()

#         loss_history.append(objective)
#         negative_mi_history.append(negative_mi_value)
#         pixel_mse_history.append(pixel_mse_value)
#         total_loss_history.append(total_loss_value)
#         encoder_ssl_history.append(encoder_ssl_value)

#         if iteration % 25 == 0:
#             current_lr = scheduler_img.get_last_lr()[0]
#             print(
#                 '%s iter = %05d, encoder_SSL = %.6f, neg_MI = %.6f, proxy_loss = %.6f, '
#                 'pixel_MSE = %.6f, total_loss = %.6f, pixel drift RMS = %.5f/255, '
#                 'PNG values changed = %.5f%%, lr = %.4f' % (
#                     get_time(), iteration, encoder_ssl_value, negative_mi_value,
#                     objective, pixel_mse_value, total_loss_value, drift, changed, current_lr
#                 ),
#                 flush=True
#             )

#         if iteration % 100 == 0 or iteration == args.Iteration:
#             grid_path = os.path.join(
#                 args.save_path,
#                 'vis_IIC_%s_%s_%gpercent_iter%d_final.png' % (args.dataset, args.model, args.percentage, iteration)
#             )
#             visible = (image_syn.detach() * diag_std + diag_mean).clamp(0, 1).cpu()
#             save_image(visible, grid_path, nrow=int(np.ceil(np.sqrt(num_syn))))

#         if iteration % 100 == 0 or iteration == args.Iteration:
#             result_path = os.path.join(
#                 args.save_path,
#                 'res_IIC_%s_%s_%gpercent_final.pt' % (args.dataset, args.model, args.percentage)
#             )
#             torch.save({
#                 'data': image_syn.detach().cpu(),
#                 'method': args.method,
#                 'iteration': args.Iteration,
#                 'loss_history': torch.tensor(loss_history, dtype=torch.float64),
#                 'negative_mi_history': torch.tensor(negative_mi_history, dtype=torch.float64),
#                 'pixel_mse_history': torch.tensor(pixel_mse_history, dtype=torch.float64),
#                 'total_loss_history': torch.tensor(total_loss_history, dtype=torch.float64),
#                 'encoder_ssl_history': torch.tensor(encoder_ssl_history, dtype=torch.float64),
#                 'seed': args.seed,
#                 'random_feature_distribution': {
#                     'models': random_models,
#                     'samples_per_update': args.num_random_networks,
#                     'encoder_train_steps': args.encoder_train_steps,
#                     'encoder_train_data': args.encoder_train_data,
#                     'encoder_train_lr': args.lr_net,
#                     'encoder_train_batch': args.batch_train,
#                     'encoder_train_method': args.encoder_train_method,
#                     'encoder_barlow_lambda': args.barlow_lambda
#                 },
#                 'iic_params': {
#                     'tau': args.temperature,
#                     'num_clusters': effective_clusters,
#                     'sub_prototypes_per_cluster': sub_prototypes_per_cluster,
#                     'strategy': args.distill_aug_strategy,
#                     'batch_real': args.batch_real,
#                     'tv_weight': args.tv_weight,
#                     'pixel_mse_weight': args.pixel_mse_weight,
#                     'pixel_mse_space': 'pixels_0_1'
#                 }
#             }, result_path)
#             print('Saved synthetic data to %s' % result_path, flush=True)


# if __name__ == '__main__':
#     main()
