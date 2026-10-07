"""Information-Theoretic Dataset Condensation: Optimizing synthetic pixels as IIC topological anchors.
Minimizes the negative mutual information between augmented views of real images assigned to synthetic prototypes in random feature spaces."""

import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image

from utils import DiffAugment, ParamDiffAug, evaluate_synset_SSL, get_dataset, get_eval_pool, get_network, get_time


def clear_directory(directory):
    """Remove everything inside directory while keeping directory itself."""
    for root, directories, files in os.walk(directory, topdown=False):
        for filename in files:
            os.remove(os.path.join(root, filename))
        for dirname in directories:
            path = os.path.join(root, dirname)
            if os.path.islink(path):
                os.remove(path)
            else:
                os.rmdir(path)


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


def total_variation_loss(img):
    """Calculates the Total Variation (TV) loss to encourage spatial smoothness and penalize high-frequency noise."""
    tv_h = torch.mean(torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :]))
    tv_w = torch.mean(torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1]))
    return tv_h + tv_w


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

    # We maximize MI by minimizing its negative
    return -MI


def main():
    parser = argparse.ArgumentParser(description='Information-Theoretic Dataset Condensation (DeepCluster + IIC)')

    # Data.
    parser.add_argument('--dataset', default='CIFAR10')
    parser.add_argument('--percentage', type=float, default=1.0)
    parser.add_argument('--data_path', default='/home/mmoslem3/scratch/data')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=0)

    # Information-Theoretic Objective.
    parser.add_argument('--Iteration', type=int, default=2000)
    parser.add_argument('--lr_img', type=float, default=0.005, help='Higher LR, decays via CosineAnnealing')
    parser.add_argument('--batch_real', type=int, default=1024, help='Slightly noisier batch prevents deep proxy minima')
    parser.add_argument('--num_random_networks', type=int, default=20, help='Fresh samples from P_net per pixel update')
    parser.add_argument('--temperature', type=float, default=0.2, help='Softmax temperature for prototype assignment')
    # parser.add_argument('--random_models', default='ConvNet,ResNet18', help='Comma-separated P_net support; forces geometric generalization')
    parser.add_argument('--random_models', default='ConvNet', help='Comma-separated P_net support; forces geometric generalization')
    parser.add_argument('--tv_weight', type=float, default=0.01, help='Total Variation penalty to destroy adversarial high-frequencies')
    parser.add_argument('--distill_aug_strategy', default='color_crop_cutout_flip_scale_rotate')
    parser.add_argument('--distill_aug_mode', choices=['S', 'M'], default='S')

    # Downstream SSL evaluation.
    parser.add_argument('--model', default='ConvNet')
    parser.add_argument('--lr_net', type=float, default=0.01)
    parser.add_argument('--batch_train', type=int, default=256)
    parser.add_argument('--ssl_method', default='simclr')
    parser.add_argument('--ssl_aug_strategy', default=None)
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
    parser.add_argument('--save_path', default='result')

    args = parser.parse_args()
    random_models = [name.strip() for name in (args.random_models or args.model).split(',') if name.strip()]
    if args.ssl_aug_strategy is None:
        args.ssl_aug_strategy = args.distill_aug_strategy
    args.method = 'InformationTheoretic_IIC'
    args.dsa_param = ParamDiffAug()
    args.dsa_param.aug_mode = args.distill_aug_mode
    args.dsa = False
    set_random_seed(args.seed)

    os.makedirs(args.save_path, exist_ok=True)
    clear_directory(args.save_path)
    print('Cleared all previous files from %s' % args.save_path, flush=True)
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
    
    augmenter = make_diff_augmenter(args.distill_aug_strategy, args.dsa_param)
    objective_rng = np.random.default_rng(args.seed + 10_000)
    
    # Optimizer & Scheduler Setup
    optimizer_img = torch.optim.Adam([image_syn], lr=args.lr_img)
    scheduler_img = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_img, T_max=args.Iteration, eta_min=0.005)
    
    loss_history = []
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
    print('Objective: Maximize IIC Mutual Information with TV Regularization and Asymmetric Augmentation.', flush=True)
    evaluate(0)
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
        sampled_models = []
        
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
            network_loss = information_theoretic_loss(features_u, features_v, features_S, tau=args.temperature)
            
            # Compute Total Variation penalty for Structural Regularization
            # tv_penalty = args.tv_weight * total_variation_loss(image_syn)
            
            # Combine and backpropagate
            # total_loss = network_loss + tv_penalty * 5
            total_loss = network_loss
            (total_loss / args.num_random_networks).backward()
            
            sampled_losses.append(network_loss.detach())
            sampled_models.append('%s(s=%d)' % (model_name, network_seed))
            del encoder, network, network_loss, total_loss, features_u, features_v, features_S, view_u, view_v

        objective = torch.stack(sampled_losses).mean().item()
        optimizer_img.step()
        scheduler_img.step()
        
        with torch.no_grad():
            image_syn.copy_(torch.maximum(torch.minimum(image_syn, pixel_max), pixel_min))
            changed = (to_uint8(image_syn) != initial_uint8).float().mean().item() * 100
            drift = ((image_syn - image_syn_initial) * diag_std * 255).square().mean().sqrt().item()
            
        loss_history.append(objective)
        if iteration % 50 == 0:
            current_lr = scheduler_img.get_last_lr()[0]
            # print('%s iter = %05d, neg_MI = %.6f, pixel drift RMS = %.5f/255, PNG values changed = %.5f%%, lr = %.4f, models = %s' % 
            #     (get_time(), iteration, objective, drift, changed, current_lr, '; '.join(sampled_models)), flush=True)
            print('%s iter = %05d, neg_MI = %.6f, pixel drift RMS = %.5f/255, PNG values changed = %.5f%%, lr = %.4f' % 
                (get_time(), iteration, objective, drift, changed, current_lr), flush=True)

        if iteration % 250 == 0:
            evaluate(iteration)

        if iteration % 100 == 0:
            grid_path = os.path.join(args.save_path, 'vis_IIC_%s_%s_%gpercent_iter%d.png' % (args.dataset, args.model, args.percentage, iteration))
            visible = (image_syn.detach() * diag_std + diag_mean).clamp(0, 1).cpu()
            save_image(visible, grid_path, nrow=int(np.ceil(np.sqrt(num_syn))))

    result_path = os.path.join(args.save_path, 'res_IIC_%s_%s_%gpercent.pt' % (args.dataset, args.model, args.percentage))
    torch.save({
        'data': image_syn.detach().cpu(), 
        'method': args.method, 
        'iteration': args.Iteration, 
        'loss_history': torch.tensor(loss_history, dtype=torch.float64), 
        'seed': args.seed, 
        'random_feature_distribution': {'models': random_models, 'samples_per_update': args.num_random_networks}, 
        'iic_params': {'tau': args.temperature, 'strategy': args.distill_aug_strategy, 'batch_real': args.batch_real, 'tv_weight': args.tv_weight}
    }, result_path)
    print('Saved synthetic data to %s' % result_path, flush=True)


if __name__ == '__main__':
    main()