"""Information-Theoretic Dataset Condensation: Optimizing synthetic pixels as IIC topological anchors.
Minimizes the negative mutual information between augmented views of real images assigned to synthetic prototypes in random feature spaces."""

import argparse
import math
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image

from eval import evaluate_synset_SSL
from utils import DiffAugment, ParamDiffAug, get_dataset, get_network, get_time


SSL_EPOCHS = {
    'CIFAR10': {0.2: 10000, 1: 8000, 2: 6000},
    'CIFAR100': {0.2: 10000, 1: 8000, 2: 6000},
    'TinyImageNet': {0.2: 6000, 1: 5000, 2: 4000},
}
PROBE_EPOCHS = {1: 100, 5: 100}


# def clear_directory(directory):
#     """Remove everything inside directory while keeping directory itself."""
#     for root, directories, files in os.walk(directory, topdown=False):
#         for filename in files:
#             os.remove(os.path.join(root, filename))
#         for dirname in directories:
#             path = os.path.join(root, dirname)
#             if os.path.islink(path):
#                 os.remove(path)
#             else:
#                 os.rmdir(path)


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
    # Dataset normalization is (pixels - mean) / std; the mean cancels in a difference.
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
    
    # Initialize K-means centers
    rng = torch.Generator().manual_seed(seed)
    indices = torch.randperm(N, generator=rng)[:M]
    centers = features[indices].clone()

    # K-Means loop
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

    # Assign initial synthetic images to the closest real image of each converged center
    dists = torch.cdist(centers, features)
    closest_indices = dists.argmin(dim=1)
    initial_syn_images = images[closest_indices].clone()

    del network, encoder, features, dists, centers
    return initial_syn_images.to(device)


def information_theoretic_loss(features_u, features_v, features_S, tau):
    """
    Compute the negative Mutual Information (IIC loss) of cluster assignments.
    Forces the physical synthetic set S to act as invariant topological anchors for augmented real views.
    """
    # L2 normalize features
    u = F.normalize(features_u, dim=1)
    v = F.normalize(features_v, dim=1)
    S = F.normalize(features_S, dim=1)

    # Cosine similarities [Batch, Num_Syn]
    sim_u = torch.mm(u, S.t())
    sim_v = torch.mm(v, S.t())

    # Soft assignments to synthetic prototypes
    p_u = F.softmax(sim_u / tau, dim=1)
    p_v = F.softmax(sim_v / tau, dim=1)

    # Joint probability matrix P [Num_Syn, Num_Syn]
    B = features_u.shape[0]
    P = torch.mm(p_u.t(), p_v) / B

    # Symmetrize to enforce identical marginals
    P = (P + P.t()) / 2.0
    P = torch.clamp(P, min=1e-8)

    # Marginals
    P_i = P.sum(dim=1, keepdim=True)
    P_j = P.sum(dim=0, keepdim=True)

    # Mutual Information: \sum P_ij \log( P_ij / (P_i * P_j) )
    MI = (P * torch.log(P / (P_i * P_j))).sum()

    negative_mi = -MI
    # Directed soft-Chamfer coverage from every synthetic prototype to the
    # real batch. Normalizing over dim=0 ensures every prototype contributes.
    log_batch_size = math.log(B)
    coverage_u = -tau * (
        torch.logsumexp(sim_u / tau, dim=0) - log_batch_size
    ).mean()
    coverage_v = -tau * (
        torch.logsumexp(sim_v / tau, dim=0) - log_batch_size
    ).mean()
    coverage_loss = 0.5 * (coverage_u + coverage_v)

    return negative_mi, coverage_loss


def main():
    parser = argparse.ArgumentParser(description='Information-Theoretic Dataset Condensation (DeepCluster + IIC)')

    # Data.
    parser.add_argument('--dataset', default='CIFAR10')
    parser.add_argument('--percentage', type=float, default=1.0)
    parser.add_argument('--data_path', default='/home/mmoslem3/scratch/data') 
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=0)

    # Information-Theoretic Objective.
    parser.add_argument('--Iteration', type=int, default=1000)
    parser.add_argument('--lr_img', type=float, default=0.005, help='Higher LR, decays via CosineAnnealing')
    parser.add_argument('--batch_real', type=int, default=1024, help='Slightly noisier batch prevents deep proxy minima')
    parser.add_argument('--num_random_networks', type=int, default=5, help='Fresh samples from P_net per pixel update')
    parser.add_argument('--temperature', type=float, default=0.2, help='Softmax temperature for prototype assignment')
    parser.add_argument('--coverage_gamma', type=float, default=0.00, help='Weight of prototype-to-real soft coverage; 0 preserves the original pure-IIC objective')
    # parser.add_argument('--random_models', default='ConvNet,ResNet18BN', help='Comma-separated P_net support; forces geometric generalization')
    parser.add_argument('--random_models', default='ConvNet', help='Comma-separated P_net support; forces geometric generalization')
    parser.add_argument('--tv_weight', type=float, default=0.01, help='Total Variation penalty to destroy adversarial high-frequencies')
    parser.add_argument('--pixel_mse_weight', type=float, default=0.0, help='Nonnegative weight for mean pixel MSE to initialization in [0,1] units; 0 disables it')
    parser.add_argument('--encoder_train_steps', type=int, default=5, help='SSL steps on the selected detached images before freezing each sampled encoder; 0 disables adaptation')
    parser.add_argument('--encoder_train_method', choices=['simclr', 'barlowtwins', 'barlow_twins', 'both'], default='simclr', help='SSL objective for sampled-encoder adaptation; both uses the mean of the SimCLR and Barlow Twins losses')
    parser.add_argument('--encoder_train_data', choices=['synthetic', 'real'], default='synthetic', help='Images used for the sampled encoder SSL warm-up')
    parser.add_argument('--distill_aug_strategy', default='color_crop_cutout_flip_scale_rotate')
    parser.add_argument('--distill_aug_mode', choices=['S', 'M'], default='M')

    # Downstream SSL evaluation.
    parser.add_argument('--model', default='ResNet18')
    parser.add_argument('--lr_net', type=float, default=0.01)
    parser.add_argument('--batch_train', type=int, default=256)
    parser.add_argument('--ssl_method', choices=['simclr', 'barlowtwins'], default='simclr')
    parser.add_argument('--ssl_aug_strategy', default=None, help='Legacy option; reference evaluation uses the benchmark view transforms')
    parser.add_argument('--projection_dim', type=int, default=128)
    parser.add_argument('--barlow_lambda', type=float, default=0.005, help='Barlow Twins off-diagonal loss weight')
    parser.add_argument('--eval_mode', default='S')
    parser.add_argument('--num_eval', type=int, default=3)
    parser.add_argument('--epoch_eval_train', type=int, default=None, help='Default: SSL_EPOCHS[dataset][percentage]')
    parser.add_argument('--label_percentage', type=float, default=1.0)
    parser.add_argument('--epoch_linear_train', type=int, default=None, help='Default: PROBE_EPOCHS[label_percentage]')
    parser.add_argument('--lr_linear', type=float, default=0.1)
    parser.add_argument('--batch_linear', type=int, default=256)

    # Output.
    parser.add_argument('--save_path', default='result_both')

    args = parser.parse_args()
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
            parser.error('No evaluation epoch schedule for this dataset/size/label budget; set the epoch overrides explicitly')
        if args.label_percentage not in PROBE_EPOCHS:
            parser.error('Reference evaluation supports 1% and 5% labels')
        if min(args.epoch_eval_train, args.epoch_linear_train, args.batch_linear) < 1 or args.lr_linear <= 0:
            parser.error('Evaluation epochs, batch size and learning rate must be positive')
        if args.eval_mode not in ('S', 'SS', 'M'):
            parser.error('Reference eval_mode must be S/SS (selected model) or M (all three BN models)')
        if args.model not in ('ConvNet', 'ConvNetBN', 'VGG11', 'VGG11BN', 'ResNet18', 'ResNet18BN'):
            parser.error('Reference evaluation requires ConvNet, VGG11 or ResNet18 (BN); no AP/cover substitution')
    if args.encoder_train_method == 'barlow_twins':
        args.encoder_train_method = 'barlowtwins'
    random_models = [name.strip() for name in (args.random_models or args.model).split(',') if name.strip()]
    if args.ssl_aug_strategy is None:
        args.ssl_aug_strategy = args.distill_aug_strategy
    args.method = 'InformationTheoretic_IIC'

    args.dsa_param = ParamDiffAug()
    args.dsa_param.aug_mode = 'S'
    distill_dsa_param = ParamDiffAug()
    distill_dsa_param.aug_mode = args.distill_aug_mode
    args.dsa = False
    set_random_seed(args.seed)

    os.makedirs(args.save_path, exist_ok=True)
    # clear_directory(args.save_path)
    # print('Cleared all previous files from %s' % args.save_path, flush=True)
    os.makedirs(args.data_path, exist_ok=True)
    channel, im_size, num_classes, _, mean, std, dst_train, _, testloader = get_dataset(args.dataset, args.data_path)
    num_syn = int(len(dst_train) * args.percentage / 100)

    # DeepCluster Initialization replacing random subset
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
    
    # Enable Shuffle for stochastic batch sampling during distillation
    real_loader = torch.utils.data.DataLoader(dst_train, batch_size=args.batch_real, shuffle=True, num_workers=0, pin_memory=args.device == 'cuda', drop_last=True)
    real_iter = iter(real_loader)
    
    augmenter = make_diff_augmenter(args.distill_aug_strategy, distill_dsa_param)
    objective_rng = np.random.default_rng(args.seed + 10_000)
    
    # Optimizer & Scheduler Setup
    optimizer_img = torch.optim.Adam([image_syn], lr=args.lr_img)
    scheduler_img = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_img, T_max=args.Iteration, eta_min=0.001)
    
    loss_history = []
    negative_mi_history = []
    coverage_loss_history = []
    weighted_coverage_history = []
    pixel_mse_history = []
    total_loss_history = []
    encoder_ssl_history = []
    # Preserve BN during evaluation; the legacy S pool strips the BN suffix.
    model_eval_pool = ([args.model] if args.eval_mode in ('S', 'SS') else ['ConvNet', 'VGG11', 'ResNet18']) if args.num_eval else []

    def evaluate(iteration):
        for model_index, model_eval in enumerate(model_eval_pool):
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
    print('Each sampled encoder receives %d SSL step(s) using %s on detached %s images before freezing.' %
          (args.encoder_train_steps, args.encoder_train_method, args.encoder_train_data), flush=True)
    print('Objective: minimize mean(-MI + %g * prototype_coverage) + %g * pixel_MSE_to_initial ([0,1] pixel units).' %
          (args.coverage_gamma, args.pixel_mse_weight), flush=True)
    # evaluate(0)
    # Evaluation
    # model_train = ConvNet, model_eval = ConvNet, iteration = 0
    # [2026-10-07 16:13:57] Evaluate_SSL_00: method = simclr ssl epoch = 1200 linear epoch = 0200 labeled = 1.00% train time = 54 s ssl loss = 1.654803 train loss = 0.013506 train acc = 1.0000, test acc = 0.3909
    # [2026-10-07 16:14:52] Evaluate_SSL_01: method = simclr ssl epoch = 1200 linear epoch = 0200 labeled = 1.00% train time = 54 s ssl loss = 1.672578 train loss = 0.012457 train acc = 1.0000, test acc = 0.3986
    # [2026-10-07 16:15:47] Evaluate_SSL_02: method = simclr ssl epoch = 1200 linear epoch = 0200 labeled = 1.00% train time = 54 s ssl loss = 1.643402 train loss = 0.009727 train acc = 1.0000, test acc = 0.3652
    # [2026-10-07 16:16:42] Evaluate_SSL_03: method = simclr ssl epoch = 1200 linear epoch = 0200 labeled = 1.00% train time = 53 s ssl loss = 1.672449 train loss = 0.013573 train acc = 1.0000, test acc = 0.3915
    # Evaluate 4 random ConvNet, mean = 0.3866 std = 0.0127

    print('%s distillation begins' % get_time(), flush=True)

    for iteration in range(1, args.Iteration + 1):
        optimizer_img.zero_grad(set_to_none=True)
        sampled_losses = []
        sampled_negative_mi_losses = []
        sampled_coverage_losses = []
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
            # No strong augmentation on S during distillation forces clean, stable anchors
            features_S = encoder(image_syn) 

            # Compute Negative Mutual Information
            negative_mi, coverage_loss = information_theoretic_loss(
                features_u, features_v, features_S, tau=args.temperature
            )
            if args.coverage_gamma == 0:
                network_loss = negative_mi
            else:
                network_loss = negative_mi + args.coverage_gamma * coverage_loss
            
            # Gradient descent on negative MI maximizes mutual information.
            (network_loss / args.num_random_networks).backward()

            sampled_losses.append(network_loss.detach())
            sampled_negative_mi_losses.append(negative_mi.detach())
            sampled_coverage_losses.append(coverage_loss.detach())
            sampled_encoder_losses.append(encoder_ssl_loss)
            sampled_models.append('%s(s=%d)' % (model_name, network_seed))
            del encoder, network, network_loss, negative_mi, coverage_loss
            del features_u, features_v, features_S, view_u, view_v

        # Add the positive drift penalty ONCE per pixel update, independent of
        # num_random_networks. The target is the fixed, unaugmented initialization.
        pixel_mse = pixel_mse_loss(image_syn, image_syn_initial, diag_std)
        if args.pixel_mse_weight > 0:
            (args.pixel_mse_weight * pixel_mse).backward()
        objective = torch.stack(sampled_losses).mean().item()
        negative_mi_value = torch.stack(sampled_negative_mi_losses).mean().item()
        coverage_value = torch.stack(sampled_coverage_losses).mean().item()
        weighted_coverage_value = args.coverage_gamma * coverage_value
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
        coverage_loss_history.append(coverage_value)
        weighted_coverage_history.append(weighted_coverage_value)
        pixel_mse_history.append(pixel_mse_value)
        total_loss_history.append(total_loss_value)
        encoder_ssl_history.append(encoder_ssl_value)
        if iteration % 25 == 0:
            current_lr = scheduler_img.get_last_lr()[0]
            print('%s iter = %05d, encoder_SSL = %.6f, neg_MI = %.6f, coverage = %.6f, proxy_loss = %.6f, pixel_MSE = %.6f, total_loss = %.6f, pixel drift RMS = %.5f/255, PNG values changed = %.5f%%, lr = %.4f' %
                (get_time(), iteration, encoder_ssl_value, negative_mi_value, coverage_value, objective, pixel_mse_value, total_loss_value, drift, changed, current_lr), flush=True)

        # if iteration % 100 == 0:
        #     evaluate(iteration)

        if iteration % 100 == 0 or iteration == args.Iteration:
            grid_path = os.path.join(args.save_path, 'vis_IIC_%s_%s_%gpercent_iter%d_final.png' % (args.dataset, args.model, args.percentage, iteration))
            visible = (image_syn.detach() * diag_std + diag_mean).clamp(0, 1).cpu()
            save_image(visible, grid_path, nrow=int(np.ceil(np.sqrt(num_syn))))

            # result_path = os.path.join(args.save_path, 'res_IIC_%s_%s_%gpercent.pt' % (args.dataset, args.model, args.percentage))
            # torch.save({
            #     'data': image_syn.detach().cpu(),
            #     'method': args.method,
            #     'iteration': args.Iteration,
            #     'loss_history': torch.tensor(loss_history, dtype=torch.float64),
            #     'pixel_mse_history': torch.tensor(pixel_mse_history, dtype=torch.float64),
            #     'total_loss_history': torch.tensor(total_loss_history, dtype=torch.float64),
            #     'encoder_ssl_history': torch.tensor(encoder_ssl_history, dtype=torch.float64),
            #     'seed': args.seed,
            #     'random_feature_distribution': {'models': random_models, 'samples_per_update': args.num_random_networks, 'encoder_train_steps': args.encoder_train_steps, 'encoder_train_data': args.encoder_train_data, 'encoder_train_lr': args.lr_net, 'encoder_train_batch': args.batch_train, 'encoder_train_method': args.encoder_train_method},
            #     'iic_params': {'tau': args.temperature, 'strategy': args.distill_aug_strategy, 'batch_real': args.batch_real, 'tv_weight': args.tv_weight,
            #                 'pixel_mse_weight': args.pixel_mse_weight, 'pixel_mse_space': 'pixels_0_1'}
            # }, result_path)
        if iteration % 200 == 0 or iteration == args.Iteration:
            result_path = os.path.join(args.save_path, 'res_IIC_%s_%s_%gpercent_final.pt' % (args.dataset, args.model, args.percentage))
            torch.save({
                'data': image_syn.detach().cpu(), 
                'dataset': args.dataset,
                'percentage': args.percentage,
                'normalization': [mean, std],
                'method': args.method, 
                'iteration': args.Iteration, 
                'loss_history': torch.tensor(loss_history, dtype=torch.float64), 
                'negative_mi_history': torch.tensor(negative_mi_history, dtype=torch.float64),
                'coverage_loss_history': torch.tensor(coverage_loss_history, dtype=torch.float64),
                'weighted_coverage_history': torch.tensor(weighted_coverage_history, dtype=torch.float64),
                'pixel_mse_history': torch.tensor(pixel_mse_history, dtype=torch.float64),
                'total_loss_history': torch.tensor(total_loss_history, dtype=torch.float64),
                'encoder_ssl_history': torch.tensor(encoder_ssl_history, dtype=torch.float64),
                'seed': args.seed, 
                'random_feature_distribution': {'models': random_models, 'samples_per_update': args.num_random_networks, 'encoder_train_steps': args.encoder_train_steps, 'encoder_train_data': args.encoder_train_data, 'encoder_train_lr': args.lr_net, 'encoder_train_batch': args.batch_train, 'encoder_train_method': args.encoder_train_method, 'encoder_barlow_lambda': args.barlow_lambda},
                'iic_params': {'tau': args.temperature, 'coverage_gamma': args.coverage_gamma, 'strategy': args.distill_aug_strategy, 'batch_real': args.batch_real, 'tv_weight': args.tv_weight,
                            'pixel_mse_weight': args.pixel_mse_weight, 'pixel_mse_space': 'pixels_0_1'}
            }, result_path)
            print('Saved synthetic data to %s' % result_path, flush=True)


if __name__ == '__main__':
    main()
