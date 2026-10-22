"""Breakthrough 2: Dual-Subspace Coding Rate & Jensen-Bregman LogDet Information Bottleneck (Dual-IB).
Aligns both the Invariant Subspace z+ = (u + v)/sqrt(2) and the Augmentation Subspace z- = (u - v)/sqrt(2)
via Log-Det Coding Rate divergence and Bipartite Mutual Information."""

import argparse
import json
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

from eval import ROOT, SyntheticImages, evaluate_synset_SSL, reference_modules
from utils import DiffAugment, ParamDiffAug, get_dataset, get_network, get_time


SSL_EPOCHS = {
    'CIFAR10': {0.2: 10000, 1: 8000, 2: 6000},
    'CIFAR100': {0.2: 10000, 1: 8000, 2: 6000},
    'TinyImageNet': {0.2: 6000, 1: 5000, 2: 4000},
}
PROBE_EPOCHS = {1: 100, 5: 100}


def set_random_seed(seed):
    """Seed initialization/evaluation while retaining explicit objective RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _images_from_batch(batch):
    return batch[0] if isinstance(batch, (tuple, list)) else batch


def frozen_embedder(network, normalization=None, encoder_normalization=None):
    """Return the frozen backbone, without its classifier or SSL projector."""
    module = network.module if isinstance(network, nn.DataParallel) else network
    network.eval()
    for parameter in network.parameters():
        parameter.requires_grad_(False)
    embed = module.encoder if hasattr(module, 'encoder') else module.embed
    if normalization is None:
        return embed

    device = next(module.parameters()).device
    mean, std = (torch.tensor(value, device=device).view(1, 3, 1, 1) for value in normalization)
    encoder_mean, encoder_std = (torch.tensor(value, device=device).view(1, 3, 1, 1) for value in encoder_normalization)

    def normalized_embed(images):
        return embed((images * std + mean - encoder_mean) / encoder_std)

    return normalized_embed


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
    """Use exactly the benchmark NT-Xent implementation."""
    return reference_modules()['train_ssl'].simclr_loss(features_u, features_v, temperature)


def barlow_twins_loss(features_u, features_v, off_diagonal_weight, loss_scale=0.1, eps=1e-5):
    """Use the benchmark variance normalization, epsilon and loss scaling."""
    return reference_modules()['train_ssl'].barlow_loss(features_u, features_v, off_diagonal_weight, loss_scale, eps)


def encoder_train_loss(features_u, features_v, args):
    """Return the selected SSL objective for sampled-encoder adaptation."""
    recipe = args.encoder_ssl_protocol['methods'][args.encoder_train_method]
    if args.encoder_train_method == 'simclr':
        return simclr_loss(features_u, features_v, recipe['temperature'])
    if args.encoder_train_method == 'barlowtwins':
        return barlow_twins_loss(features_u, features_v, recipe['lambda'], recipe['loss_scale'], recipe['correlation_eps'])
    raise ValueError('Unknown encoder training method: %s' % args.encoder_train_method)


def train_encoder_ssl(network, images, args, objective_rng, normalization):
    """Benchmark SSL components, capped at the requested short adaptation budget."""
    if args.encoder_train_steps == 0:
        return 0.0
    reference = reference_modules()
    common, data, ssl = reference['common'], reference['data'], reference['train_ssl']
    protocol = args.encoder_ssl_protocol
    cfg, recipe = protocol['ssl'], protocol['methods'][args.encoder_train_method]
    module = network.module if isinstance(network, nn.DataParallel) else network
    network.train()
    for parameter in network.parameters():
        parameter.requires_grad_(True)
    mean, std = (torch.tensor(value).view(1, 3, 1, 1) for value in normalization)
    pixels = images.detach().cpu() * std + mean
    if not torch.isfinite(pixels).all() or len(pixels) < 2:
        raise ValueError('Encoder SSL requires at least two finite images')
    dataset = SyntheticImages(pixels.clamp(0, 1).contiguous())
    batch_size = min(cfg['batch_size'], len(dataset))
    seed = int(objective_rng.integers(0, 2**31 - 1))
    stream = data.loader(data.ImageView(dataset, data.ssl_transform(protocol, args.dataset, args.encoder_train_method)),
                         batch_size, 0, seed, shuffle=True, drop_last=True)
    peak = recipe['lr'] * (batch_size / cfg['batch_size'] if cfg['scale_lr_with_batch'] else 1)
    optimizer = ssl.LARS(network.parameters(), peak, cfg)
    epochs = math.ceil(args.encoder_train_steps / len(stream))
    warmup_steps = min(cfg['warmup_epochs'], max(0, epochs - 1)) * len(stream)
    losses = []
    rng, numpy_rng = common.rng_state(), np.random.get_state()
    settings = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
                torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic)
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        step = 0
        for epoch in range(epochs):
            epoch_seed = seed * 1000003 + epoch + 10000
            common.seed_all(epoch_seed)
            stream.generator.manual_seed(epoch_seed)
            for (view_u, view_v), _ in stream:
                lr = ssl.cosine_lr(step, args.encoder_train_steps, peak, warmup_steps,
                                   cfg['warmup_start_lr'], cfg['minimum_lr'])
                for group in optimizer.param_groups:
                    group['lr'] = lr
                features_u, features_v = module(view_u.to(images.device), view_v.to(images.device))
                loss = encoder_train_loss(features_u, features_v, args)
                if not torch.isfinite(loss):
                    raise RuntimeError('Non-finite synthetic encoder warm-up loss')
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                finite = torch.stack([torch.isfinite(parameter.grad).all() for parameter in network.parameters() if parameter.grad is not None]).all()
                if not finite:
                    raise RuntimeError('Non-finite synthetic encoder warm-up gradient')
                optimizer.step()
                losses.append(loss.detach())
                step += 1
                if step == args.encoder_train_steps:
                    break
    finally:
        common.restore_rng(rng)
        np.random.set_state(numpy_rng)
        (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
         torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic) = settings
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
def deepcluster_initialize(dst_train, num_syn, device, seed, channel, num_classes, im_size, model_name='ConvNet'):
    """Extract features with a random network and perform K-Means to initialize synthetic images as cluster centroids."""
    print(f'Initializing {num_syn} synthetic prototypes via DeepCluster K-Means on {model_name}...', flush=True)
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


def jbld_coding_rate_divergence(Z_syn, Z_real, eps=1e-2):
    """
    Compute the Jensen-Bregman LogDet Divergence (JBLD) + Frobenius spectrum alignment
    between the coding-rate covariance matrices of Z_syn and Z_real:
        Sigma = eps * I + (1 - eps) * (Z^T Z / N)
        JBLD(Sigma_S, Sigma_T) = logdet((Sigma_S + Sigma_T)/2) - 0.5 * logdet(Sigma_S) - 0.5 * logdet(Sigma_T)
    Strictly symmetric, bounded, and requires no unstable matrix inversion.
    """
    N_s, D = Z_syn.shape
    N_r, _ = Z_real.shape
    eye = torch.eye(D, device=Z_syn.device, dtype=Z_syn.dtype)

    cov_syn = (1.0 - eps) * (torch.mm(Z_syn.t(), Z_syn) / max(N_s, 1)) + eps * eye
    cov_real = (1.0 - eps) * (torch.mm(Z_real.t(), Z_real) / max(N_r, 1)) + eps * eye
    cov_mid = 0.5 * (cov_syn + cov_real)

    _, logdet_mid = torch.linalg.slogdet(cov_mid)
    _, logdet_syn = torch.linalg.slogdet(cov_syn)
    _, logdet_real = torch.linalg.slogdet(cov_real)

    jbld = logdet_mid - 0.5 * logdet_syn - 0.5 * logdet_real
    frob = F.mse_loss(cov_syn, cov_real) * D
    return jbld + 0.5 * frob


def dual_subspace_coding_rate_loss(features_u_real, features_v_real, features_u_syn, features_v_syn, tau, eps=1e-2):
    """
    Breakthrough 2 Objective:
    Decomposes view pairs into orthogonal Invariant (z+) and Augmentation/Nuisance (z-) subspaces:
        z+ = (u + v) / sqrt(2)   [Semantic invariance subspace]
        z- = (u - v) / sqrt(2)   [Augmentation variance subspace that SSL quotients out]
    Aligns the continuous Gaussian Coding Rate (LogDet entropy) across both subspaces while
    maximizing Bipartite Mutual Information.
    """
    u_r = F.normalize(features_u_real, dim=1)
    v_r = F.normalize(features_v_real, dim=1)
    u_s = F.normalize(features_u_syn, dim=1)
    v_s = F.normalize(features_v_syn, dim=1)

    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    # Orthogonal Decomposition into Invariant (+) and Augmentation (-) Subspaces
    z_plus_real = (u_r + v_r) * inv_sqrt2
    z_minus_real = (u_r - v_r) * inv_sqrt2

    z_plus_syn = (u_s + v_s) * inv_sqrt2
    z_minus_syn = (u_s - v_s) * inv_sqrt2

    # 1. Dual-Subspace Coding Rate Alignment (JBLD + Covariance Matching)
    loss_subspace_plus = jbld_coding_rate_divergence(z_plus_syn, z_plus_real, eps=eps)
    loss_subspace_minus = jbld_coding_rate_divergence(z_minus_syn, z_minus_real, eps=eps)
    coding_rate_loss = loss_subspace_plus + loss_subspace_minus

    # 2. Bipartite Mutual Information in the Invariant + View Space
    sim_u = torch.mm(u_r, u_s.t())
    sim_v = torch.mm(v_r, v_s.t())

    p_u = F.softmax(sim_u / tau, dim=1)
    p_v = F.softmax(sim_v / tau, dim=1)

    B = u_r.shape[0]
    P = torch.mm(p_u.t(), p_v) / B
    P = 0.5 * (P + P.t())
    P = torch.clamp(P, min=1e-8)

    P_i = P.sum(dim=1, keepdim=True)
    P_j = P.sum(dim=0, keepdim=True)
    MI = (P * torch.log(P / (P_i * P_j))).sum()
    negative_mi = -MI

    # 3. Prototype-to-Real Soft Coverage
    log_batch_size = math.log(B)
    coverage_u = -tau * (torch.logsumexp(sim_u / tau, dim=0) - log_batch_size).mean()
    coverage_v = -tau * (torch.logsumexp(sim_v / tau, dim=0) - log_batch_size).mean()
    coverage_loss = 0.5 * (coverage_u + coverage_v)

    return negative_mi, coding_rate_loss, coverage_loss


def main():
    parser = argparse.ArgumentParser(description='Breakthrough 2: Dual-Subspace Coding Rate & JBLD Alignment')

    # Data.
    parser.add_argument('--dataset', default='CIFAR10')
    parser.add_argument('--percentage', type=float, default=0.2)
    parser.add_argument('--data_path', default='/home/mmoslem3/scratch/data')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=0)

    # Information-Theoretic Objective.
    parser.add_argument('--Iteration', type=int, default=1000)
    parser.add_argument('--lr_img', type=float, default=0.005)
    parser.add_argument('--batch_real', type=int, default=512)
    parser.add_argument('--num_random_networks', type=int, default=5)
    parser.add_argument('--temperature', type=float, default=0.2)
    parser.add_argument('--coding_rate_weight', type=float, default=0.1, help='Weight for Dual-Subspace JBLD Coding Rate alignment')
    parser.add_argument('--coding_eps', type=float, default=0.02, help='Diagonal shrinkage epsilon for LogDet stability')
    
    parser.add_argument('--random_models', default='ConvNet')
    
    parser.add_argument('--tv_weight', type=float, default=0)
    parser.add_argument('--pixel_mse_weight', type=float, default=0.0)
    parser.add_argument('--coverage_gamma', type=float, default=0.0, help='Weight of prototype-to-real soft coverage')

    parser.add_argument('--encoder_train_steps', type=int, default=3)
    parser.add_argument('--encoder_train_method', choices=['simclr', 'barlowtwins', 'barlow_twins'], default='simclr')
    parser.add_argument('--encoder_train_data', choices=['synthetic', 'real'], default='synthetic')
    parser.add_argument('--distill_aug_strategy', '--dsa_strategy', '--dsa-strategy', default='color_crop_cutout_flip_scale_rotate')
    parser.add_argument('--dsa', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--distill_aug_mode', choices=['S', 'M'], default='M')

    # Downstream SSL evaluation.
    parser.add_argument('--model', default='ResNet18')
    parser.add_argument('--lr_net', type=float, default=None)
    parser.add_argument('--batch_train', type=int, default=256)
    parser.add_argument('--ssl_method', choices=['simclr', 'barlowtwins'], default='simclr')
    parser.add_argument('--ssl_aug_strategy', default=None)
    parser.add_argument('--projection_dim', type=int, default=None)
    parser.add_argument('--barlow_lambda', type=float, default=None)
    parser.add_argument('--eval_mode', default='S')
    parser.add_argument('--num_eval', type=int, default=0)
    parser.add_argument('--epoch_eval_train', type=int, default=None)
    parser.add_argument('--label_percentage', type=float, default=1.0)
    parser.add_argument('--epoch_linear_train', type=int, default=None)
    parser.add_argument('--lr_linear', type=float, default=0.1)
    parser.add_argument('--batch_linear', type=int, default=256)

    # Output.
    parser.add_argument('--save_path', default='method2')
    parser.add_argument('--resume', default=None, help='continue saved synthetic images and iteration')

    teacher_arguments(parser)
    args = parser.parse_args()
    if args.teacher_source == 'pretrained':
        args.encoder_train_steps = 0
    if args.num_eval < 0:
        parser.error('num_eval must be nonnegative')
    if args.num_eval:
        if args.dataset not in SSL_EPOCHS:
            parser.error('Reference evaluation supports CIFAR10, CIFAR100 and TinyImageNet')
        if args.epoch_eval_train is None:
            args.epoch_eval_train = SSL_EPOCHS.get(args.dataset, {}).get(args.percentage)
        if args.epoch_linear_train is None:
            args.epoch_linear_train = PROBE_EPOCHS.get(args.label_percentage)
        if args.epoch_eval_train is None or args.epoch_linear_train is None:
            parser.error('No evaluation epoch schedule for this dataset/size/label budget; set overrides explicitly')
        if args.label_percentage not in PROBE_EPOCHS:
            parser.error('Reference evaluation supports 1% and 5% labels')
        if min(args.epoch_eval_train, args.epoch_linear_train, args.batch_linear) < 1 or args.lr_linear <= 0:
            parser.error('Evaluation epochs, batch size and learning rate must be positive')
        if args.eval_mode not in ('S', 'SS', 'M'):
            parser.error('Reference eval_mode must be S/SS or M')
        if args.model not in ('ConvNet', 'ConvNetBN', 'VGG11', 'VGG11BN', 'ResNet18', 'ResNet18BN'):
            parser.error('Reference evaluation requires ConvNet, VGG11 or ResNet18 (BN)')
    if args.encoder_train_method == 'barlow_twins':
        args.encoder_train_method = 'barlowtwins'
    random_models = [name.strip() for name in (args.random_models or args.model).split(',') if name.strip()]
    if args.encoder_train_steps < 0:
        parser.error('encoder_train_steps must be nonnegative')
    args.encoder_ssl_protocol = None
    if args.encoder_train_steps:
        if args.dataset not in SSL_EPOCHS or not random_models:
            parser.error('Encoder SSL adaptation requires CIFAR10/CIFAR100/TinyImageNet and nonempty random_models')
        if any(name not in ('ConvNet', 'ConvNetBN', 'VGG11', 'VGG11BN', 'ResNet18', 'ResNet18BN') for name in random_models):
            parser.error('Encoder SSL adaptation uses the benchmark ConvNet/VGG11/ResNet18 BN backbones')
        args.encoder_ssl_protocol = json.loads((ROOT / 'Evaluate_benchamrk/protocol.json').read_text())
        recipe = args.encoder_ssl_protocol['methods'][args.encoder_train_method]
        args.lr_net = recipe['lr'] if args.lr_net is None else args.lr_net
        args.projection_dim = recipe['projector'][-1] if args.projection_dim is None else args.projection_dim
        args.barlow_lambda = args.encoder_ssl_protocol['methods']['barlowtwins']['lambda'] if args.barlow_lambda is None else args.barlow_lambda
        recipe['lr'] = args.lr_net
        recipe['projector'][-1] = args.projection_dim
        args.encoder_ssl_protocol['methods']['barlowtwins']['lambda'] = args.barlow_lambda
        args.encoder_ssl_protocol['ssl']['batch_size'] = args.batch_train

    args.method = 'DualSubspace_CodingRate_MI'

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

    resumed = torch.load(args.resume, map_location='cpu', weights_only=True) if args.resume else None
    start_iteration = 0
    if resumed is not None:
        if (any(resumed[key] != getattr(args, key) for key in
                ('dataset', 'percentage', 'seed', 'dsa', 'teacher_source', 'teacher_seed', 'method'))
                or resumed['normalization'] != [mean, std]
                or tuple(resumed['data'].shape) != (num_syn, channel, *im_size)
                or not torch.isfinite(resumed['data']).all().item()):
            raise ValueError('Resume checkpoint does not match this run or contains invalid images')
        start_iteration = resumed['iteration']
        if not 0 <= start_iteration <= args.Iteration:
            raise ValueError('Resume iteration exceeds the requested stopping iteration')
        if args.pixel_mse_weight and 'image_syn_initial' not in resumed:
            raise ValueError('Original pixel-MSE anchor is missing from the resume checkpoint')
        image_syn_init = resumed['data'].to(args.device).detach().clone()
    else:
        image_syn_init = deepcluster_initialize(
            dst_train, num_syn, args.device, args.seed, channel, num_classes, im_size, model_name=random_models[0]
        )

    image_syn = image_syn_init.detach().requires_grad_(True)
    image_syn_initial = image_syn.detach().clone()
    if resumed is not None and 'image_syn_initial' in resumed:
        image_syn_initial = resumed['image_syn_initial'].to(args.device)

    diag_mean = torch.tensor(mean, device=args.device).view(1, channel, 1, 1)
    diag_std = torch.tensor(std, device=args.device).view(1, channel, 1, 1)
    pixel_min = -diag_mean / diag_std
    pixel_max = (1 - diag_mean) / diag_std

    def to_uint8(images):
        return ((images * diag_std + diag_mean) * 255 + 0.5).clamp(0, 255).to(torch.uint8)

    initial_uint8 = to_uint8(image_syn_initial)

    real_loader = torch.utils.data.DataLoader(
        dst_train, batch_size=args.batch_real, shuffle=True, num_workers=0, pin_memory=args.device == 'cuda', drop_last=True
    )
    real_iter = iter(real_loader)

    augmenter = make_diff_augmenter(args.distill_aug_strategy, distill_dsa_param, args.dsa)
    objective_rng = np.random.default_rng(args.seed + 10_000)

    optimizer_img = torch.optim.Adam([image_syn], lr=args.lr_img)
    scheduler_img = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_img, T_max=args.Iteration, eta_min=args.lr_img)
    if resumed is not None:
        if 'optimizer_state' in resumed:
            optimizer_img.load_state_dict(resumed['optimizer_state'])
            scheduler_img.load_state_dict(resumed['scheduler_state'])
            objective_rng.bit_generator.state = resumed['objective_rng']
        else:
            print('Old checkpoint has no Adam state; continuing saved images with fresh Adam and resume-relative drift.', flush=True)
        print('Resuming saved iteration %d -> %d from %s' % (start_iteration, args.Iteration, args.resume), flush=True)

    loss_history = []
    negative_mi_history = []
    coding_rate_history = []
    coverage_loss_history = []
    weighted_coverage_history = []
    pixel_mse_history = []
    total_loss_history = []
    encoder_ssl_history = []
    if resumed is not None:
        for name, history in (
            ('loss_history', loss_history), ('negative_mi_history', negative_mi_history),
            ('coding_rate_history', coding_rate_history), ('coverage_loss_history', coverage_loss_history),
            ('weighted_coverage_history', weighted_coverage_history), ('pixel_mse_history', pixel_mse_history),
            ('total_loss_history', total_loss_history), ('encoder_ssl_history', encoder_ssl_history),
        ):
            history.extend(resumed[name].tolist())
    model_eval_pool = ([args.model] if args.eval_mode in ('S', 'SS') else ['ConvNet', 'VGG11', 'ResNet18']) if args.num_eval else []

    def evaluate(iteration):
        for model_eval in model_eval_pool:
            print('-------------------------\nEvaluation\nmodel_train = %s, model_eval = %s, iteration = %d' % (args.model, model_eval, iteration))
            accuracies = []
            for eval_index in range(args.num_eval):
                eval_seed = args.seed + 20000 + eval_index
                output = os.path.join(args.save_path, 'evaluation', 'iter_%05d' % iteration, model_eval, 'seed_%03d' % eval_seed)
                _, _, accuracy = evaluate_synset_SSL(eval_index, model_eval, image_syn.detach(), args,
                                                     output, normalization=[mean, std], seed=eval_seed)
                accuracies.append(float(accuracy))
            print('Evaluate %d random %s, mean = %.4f std = %.4f\n-------------------------' %
                  (len(accuracies), model_eval, np.mean(accuracies), np.std(accuracies, ddof=1) if len(accuracies) > 1 else float('nan')), flush=True)

    print('Hyper-parameters: \n', args.__dict__)
    print('P_net support: %s' % ', '.join(random_models), flush=True)
    print('Objective: minimize mean(-MI + %g * dual_coding_rate + %g * coverage) + %g * TV + %g * pixel_MSE.' %
          (args.coding_rate_weight, args.coverage_gamma, args.tv_weight, args.pixel_mse_weight), flush=True)
    print('%s distillation begins' % get_time(), flush=True)

    for iteration in range(start_iteration + 1, args.Iteration + 1):
        optimizer_img.zero_grad(set_to_none=True)
        sampled_losses = []
        sampled_negative_mi_losses = []
        sampled_coding_losses = []
        sampled_coverage_losses = []
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
            if args.encoder_train_steps:
                torch.manual_seed(network_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(network_seed)
                encoder_model = {'ConvNetBN': 'ConvNet', 'VGG11BN': 'VGG11', 'ResNet18BN': 'ResNet18'}.get(model_name, model_name)
                network = reference_modules()['train_ssl'].SSLModel(
                    encoder_model, im_size[0], args.encoder_ssl_protocol['methods'][args.encoder_train_method]).to(args.device)
            else:
                network = pretrained_teacher if pretrained_teacher is not None else get_network(model_name, channel, num_classes, im_size, seed=network_seed).to(args.device)
            encoder_train_images = image_syn if args.encoder_train_data == 'synthetic' else real_imgs
            encoder_ssl_loss = train_encoder_ssl(network, encoder_train_images, args, objective_rng, [mean, std])
            encoder = frozen_embedder(network, [mean, std], args.encoder_ssl_protocol['normalization'][args.dataset]) if args.encoder_train_steps else frozen_embedder(network)

            seed_u_real = int(objective_rng.integers(0, 2**31 - 1))
            seed_v_real = int(objective_rng.integers(0, 2**31 - 1))
            seed_u_syn = int(objective_rng.integers(0, 2**31 - 1))
            seed_v_syn = int(objective_rng.integers(0, 2**31 - 1))
            if args.dsa:
                seed_u_syn, seed_v_syn = seed_u_real, seed_v_real

            view_u_real = augmenter(real_imgs, seed_u_real)
            view_v_real = augmenter(real_imgs, seed_v_real)

            with torch.no_grad():
                features_u_real = encoder(view_u_real)
                features_v_real = encoder(view_v_real)

            view_u_syn = augmenter(image_syn, seed_u_syn)
            view_v_syn = augmenter(image_syn, seed_v_syn)
            features_u_syn = encoder(view_u_syn)
            features_v_syn = encoder(view_v_syn)

            negative_mi, coding_loss, coverage_loss = dual_subspace_coding_rate_loss(
                features_u_real, features_v_real, features_u_syn, features_v_syn,
                tau=args.temperature, eps=args.coding_eps
            )
            network_loss = negative_mi + args.coding_rate_weight * coding_loss + args.coverage_gamma * coverage_loss

            (network_loss / args.num_random_networks).backward()

            sampled_losses.append(network_loss.detach())
            sampled_negative_mi_losses.append(negative_mi.detach())
            sampled_coding_losses.append(coding_loss.detach())
            sampled_coverage_losses.append(coverage_loss.detach())
            sampled_encoder_losses.append(encoder_ssl_loss)
            del encoder, network, network_loss, negative_mi, coding_loss, coverage_loss
            del features_u_real, features_v_real, features_u_syn, features_v_syn
            del view_u_real, view_v_real, view_u_syn, view_v_syn

        pixel_mse = pixel_mse_loss(image_syn, image_syn_initial, diag_std)
        tv_val = total_variation_loss(image_syn)
        reg_loss = 0.0
        if args.pixel_mse_weight > 0:
            reg_loss = reg_loss + args.pixel_mse_weight * pixel_mse
        if args.tv_weight > 0:
            reg_loss = reg_loss + args.tv_weight * tv_val
        if isinstance(reg_loss, torch.Tensor):
            reg_loss.backward()

        objective = torch.stack(sampled_losses).mean().item()
        negative_mi_value = torch.stack(sampled_negative_mi_losses).mean().item()
        coding_loss_value = torch.stack(sampled_coding_losses).mean().item()
        coverage_value = torch.stack(sampled_coverage_losses).mean().item()
        weighted_coverage_value = args.coverage_gamma * coverage_value
        pixel_mse_value = pixel_mse.detach().item()
        total_loss_value = objective + args.pixel_mse_weight * pixel_mse_value + args.tv_weight * tv_val.detach().item()
        encoder_ssl_value = float(np.mean(sampled_encoder_losses))

        optimizer_img.step()
        scheduler_img.step()

        with torch.no_grad():
            image_syn.copy_(torch.maximum(torch.minimum(image_syn, pixel_max), pixel_min))
            changed = (to_uint8(image_syn) != initial_uint8).float().mean().item() * 100
            drift = ((image_syn - image_syn_initial) * diag_std * 255).square().mean().sqrt().item()

        loss_history.append(objective)
        negative_mi_history.append(negative_mi_value)
        coding_rate_history.append(coding_loss_value)
        coverage_loss_history.append(coverage_value)
        weighted_coverage_history.append(weighted_coverage_value)
        pixel_mse_history.append(pixel_mse_value)
        total_loss_history.append(total_loss_value)
        encoder_ssl_history.append(encoder_ssl_value)

        if iteration % 25 == 0:
            current_lr = scheduler_img.get_last_lr()[0]
            print('%s iter = %05d, encoder_SSL = %.6f, neg_MI = %.6f, coding_rate = %.6f, coverage = %.6f, proxy_loss = %.6f, pixel_MSE = %.6f, total_loss = %.6f, pixel drift RMS = %.5f/255, PNG changed = %.5f%%, lr = %.4f' %
                  (get_time(), iteration, encoder_ssl_value, negative_mi_value, coding_loss_value, coverage_value, objective, pixel_mse_value, total_loss_value, drift, changed, current_lr), flush=True)

        if iteration % 100 == 0 or iteration == args.Iteration:
            grid_path = os.path.join(args.save_path, 'vis_IIC_%s_%s_%gpercent_iter%d_final.png' % (args.dataset, args.model, args.percentage, iteration))
            visible = (image_syn.detach() * diag_std + diag_mean).clamp(0, 1).cpu()
            save_image(visible, grid_path, nrow=int(np.ceil(np.sqrt(num_syn))))

        if iteration % 50 == 0 or iteration == args.Iteration:
            result_path = os.path.join(args.save_path, 'res_IIC_%s_%s_%gpercent_final.pt' % (args.dataset, args.model, args.percentage))
            reference_modules()['common'].atomic_save(result_path, {
                'data': image_syn.detach().cpu(),
                'dataset': args.dataset,
                'percentage': args.percentage,
                'normalization': [mean, std],
                'method': args.method,
                'iteration': iteration,
                'dsa': args.dsa, 'teacher_source': args.teacher_source, 'teacher_seed': args.teacher_seed,
                'loss_history': torch.tensor(loss_history, dtype=torch.float64),
                'negative_mi_history': torch.tensor(negative_mi_history, dtype=torch.float64),
                'coding_rate_history': torch.tensor(coding_rate_history, dtype=torch.float64),
                'coverage_loss_history': torch.tensor(coverage_loss_history, dtype=torch.float64),
                'weighted_coverage_history': torch.tensor(weighted_coverage_history, dtype=torch.float64),
                'pixel_mse_history': torch.tensor(pixel_mse_history, dtype=torch.float64),
                'total_loss_history': torch.tensor(total_loss_history, dtype=torch.float64),
                'encoder_ssl_history': torch.tensor(encoder_ssl_history, dtype=torch.float64),
                'encoder_ssl_protocol': args.encoder_ssl_protocol,
                'seed': args.seed,
                'model': args.model, 'resumed_from': args.resume,
                'optimizer_state': optimizer_img.state_dict(),
                'scheduler_state': scheduler_img.state_dict(),
                'objective_rng': objective_rng.bit_generator.state,
                'image_syn_initial': image_syn_initial.detach().cpu(),
                'iic_params': {
                    'tau': args.temperature,
                    'coding_rate_weight': args.coding_rate_weight,
                    'coding_eps': args.coding_eps,
                    'coverage_gamma': args.coverage_gamma,
                    'strategy': args.distill_aug_strategy,
                    'batch_real': args.batch_real,
                    'tv_weight': args.tv_weight,
                    'pixel_mse_weight': args.pixel_mse_weight,
                }
            })
            print('Saved synthetic data to %s' % result_path, flush=True)


    result_path = os.path.join(args.save_path, 'res_IIC_%s_%s_%gpercent_final.pt' % (args.dataset, args.model, args.percentage))
    reference_modules()['common'].atomic_save(result_path, {
        'data': image_syn.detach().cpu(),
        'dataset': args.dataset,
        'percentage': args.percentage,
        'normalization': [mean, std],
        'method': args.method,
        'iteration': args.Iteration,
        'dsa': args.dsa, 'teacher_source': args.teacher_source, 'teacher_seed': args.teacher_seed,
        'loss_history': torch.tensor(loss_history, dtype=torch.float64),
        'negative_mi_history': torch.tensor(negative_mi_history, dtype=torch.float64),
        'coding_rate_history': torch.tensor(coding_rate_history, dtype=torch.float64),
        'coverage_loss_history': torch.tensor(coverage_loss_history, dtype=torch.float64),
        'weighted_coverage_history': torch.tensor(weighted_coverage_history, dtype=torch.float64),
        'pixel_mse_history': torch.tensor(pixel_mse_history, dtype=torch.float64),
        'total_loss_history': torch.tensor(total_loss_history, dtype=torch.float64),
        'encoder_ssl_history': torch.tensor(encoder_ssl_history, dtype=torch.float64),
        'encoder_ssl_protocol': args.encoder_ssl_protocol,
        'seed': args.seed,
        'model': args.model, 'resumed_from': args.resume,
        'optimizer_state': optimizer_img.state_dict(),
        'scheduler_state': scheduler_img.state_dict(),
        'objective_rng': objective_rng.bit_generator.state,
        'image_syn_initial': image_syn_initial.detach().cpu(),
        'iic_params': {
            'tau': args.temperature,
            'coding_rate_weight': args.coding_rate_weight,
            'coding_eps': args.coding_eps,
            'coverage_gamma': args.coverage_gamma,
            'strategy': args.distill_aug_strategy,
            'batch_real': args.batch_real,
            'tv_weight': args.tv_weight,
            'pixel_mse_weight': args.pixel_mse_weight,
        }
    })
    print('Saved synthetic data to %s' % result_path, flush=True)



if __name__ == '__main__':
    main()
