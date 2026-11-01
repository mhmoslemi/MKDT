"""MIPA: bipartite view-pair mutual information with a frozen encoder."""

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
from dm_dsa import make_diff_augmenter
from dd_teachers import load_teacher

from eval import ROOT, SYNTHETIC_NORMALIZATION, reference_modules
from utils import ParamDiffAug, get_time


def set_random_seed(seed):
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


def frozen_embedder(network):
    module = network.module if isinstance(network, nn.DataParallel) else network
    network.eval()
    for parameter in network.parameters():
        parameter.requires_grad_(False)
    return module.encoder if hasattr(module, 'encoder') else module.embed


@torch.no_grad()
def random_initialize(dst_train, num_syn, device, seed):
    # Same indices and order as Evaluate_benchamrk/data.py's Random baseline.
    indices = np.random.default_rng(seed + 1000).permutation(len(dst_train))[:num_syn]
    assert len(indices) == num_syn, 'Not enough real images for initialization'
    print(f'Initializing {num_syn} synthetic images from the Random-baseline training images (seed={seed})...', flush=True)
    return torch.stack([_images_from_batch(dst_train[int(index)]) for index in indices]).to(device)


def bipartite_view_transition_loss(features_u_real, features_v_real, features_u_syn, features_v_syn, tau, beta=1.1, pool_size=0):
    u_r = F.normalize(features_u_real, dim=1)
    v_r = F.normalize(features_v_real, dim=1)
    u_s = F.normalize(features_u_syn, dim=1)
    v_s = F.normalize(features_v_syn, dim=1)

    num_pools = math.ceil(len(u_s) / pool_size) if pool_size else 1
    losses = []
    # Fixed, balanced pools preserve image order and include every prototype.
    for us, vs in zip(u_s.tensor_split(num_pools), v_s.tensor_split(num_pools)):
        sim_u = torch.mm(u_r, us.t())
        sim_v = torch.mm(v_r, vs.t())
        effective_tau = tau * math.log(100.0) / math.log(max(len(us), 100))
        p_u = F.softmax(sim_u / effective_tau, dim=1)
        p_v = F.softmax(sim_v / effective_tau, dim=1)

        P = torch.mm(p_u.t(), p_v) / len(p_u)
        P = 0.5 * (P + P.t())
        P_i = P.sum(dim=1, keepdim=True)
        P_j = P.sum(dim=0, keepdim=True)
        tiny = torch.finfo(P.dtype).tiny
        joint_entropy = -(P * P.clamp_min(tiny).log()).sum()
        marginal_entropy = -(P_i * P_i.clamp_min(tiny).log()).sum() - (P_j * P_j.clamp_min(tiny).log()).sum()
        losses.append(joint_entropy - beta * marginal_entropy)
    return torch.stack(losses).mean()


# def bipartite_view_transition_loss(features_u_real, features_v_real, features_u_syn, features_v_syn, tau, beta=1.1):
#     u_r = F.normalize(features_u_real, dim=1)
#     v_r = F.normalize(features_v_real, dim=1)
#     u_s = F.normalize(features_u_syn, dim=1)
#     v_s = F.normalize(features_v_syn, dim=1)

#     num_syn = u_s.shape[0]
#     effective_tau = tau * math.log(100.0) / math.log(max(num_syn, 100))

#     sim_u = torch.mm(u_r, u_s.t())
#     sim_v = torch.mm(v_r, v_s.t())

#     # Fix 1: Forward-KL log-barrier on unmasked probabilities so d(-log P_k)/ds_ik ~ q_ik / P_k = O(1)
#     p_u_full = F.softmax(sim_u / effective_tau, dim=1)
#     p_v_full = F.softmax(sim_v / effective_tau, dim=1)
#     p_marginal = 0.5 * (p_u_full.mean(dim=0) + p_v_full.mean(dim=0))
#     log_barrier = -torch.log(p_marginal.clamp_min(1e-8)).mean()

#     # Fix 2a: Top-K masking prevents prototypes from averaging opposing gradients from unrelated real images
#     if num_syn > 64:
#         k_top = max(32, num_syn // 5)
#         topk_u = sim_u.topk(k_top, dim=1).values[:, -1:]
#         topk_v = sim_v.topk(k_top, dim=1).values[:, -1:]
#         sim_u = torch.where(sim_u >= topk_u, sim_u, torch.full_like(sim_u, -1e4))
#         sim_v = torch.where(sim_v >= topk_v, sim_v, torch.full_like(sim_v, -1e4))

#     p_u = F.softmax(sim_u / effective_tau, dim=1)
#     p_v = F.softmax(sim_v / effective_tau, dim=1)

#     # Fix 2b: Sub-batch into chunks of 128 so larger batch_real does not blur P
#     sub_batch = 128
#     mi_losses = []
#     for cu, cv in zip(p_u.split(sub_batch, dim=0), p_v.split(sub_batch, dim=0)):
#         if len(cu) < 32:
#             continue
#         P = torch.mm(cu.t(), cv) / len(cu)
#         P = 0.5 * (P + P.t())
#         P_i = P.sum(dim=1, keepdim=True).clamp_min(1e-8)
#         P_j = P.sum(dim=0, keepdim=True).clamp_min(1e-8)
#         joint_entropy = -(P * P.clamp_min(1e-8).log()).sum()
#         marginal_entropy = -(P_i * P_i.log()).sum() - (P_j * P_j.log()).sum()
#         mi_losses.append(joint_entropy - beta * marginal_entropy)

#     return torch.stack(mi_losses).mean() + 0.5 * log_barrier




def main():
    parser = argparse.ArgumentParser(description='MIPA: Bipartite View-Pair Mutual Information')

    # Data.
    parser.add_argument('--dataset', choices=['CIFAR10', 'CIFAR100', 'TinyImageNet', 'FashionMNIST', 'SVHN'], default='CIFAR10')
    parser.add_argument('--percentage', type=float, default=0.2)
    parser.add_argument('--data_path', default='/home/mmoslem3/scratch/data')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--single_channel', action='store_true', help='FashionMNIST: optimize one grayscale channel and repeat it to RGB for teachers and saved images')

    # Information-Theoretic Objective.
    parser.add_argument('--Iteration', type=int, default=10000)
    parser.add_argument('--lr_img', type=float, default=0.0015, help='Synthetic-image learning rate')
    parser.add_argument('--batch_real', type=int, default=128)
    parser.add_argument('--temperature', type=float, default=0.2, help='Base assignment temperature, scaled by prototype count')
    parser.add_argument('--beta', type=float, default=1, help='Weight of both marginal entropies; 1 gives standard mutual information')
    parser.add_argument('--prototype_pool_size', type=int, default=0, help='Maximum prototypes per fixed balanced MI pool; 0 uses the full set, 100 enables smaller pools')

    # Augmentation.
    parser.add_argument('--distill_aug_strategy', '--dsa_strategy', '--dsa-strategy', default=None, help='Default: color_crop_cutout_scale for SVHN; color_crop_cutout_flip_scale_rotate otherwise')
    # Output.
    parser.add_argument('--save_path', default='less_aug_proto')
    parser.add_argument('--resume', action='store_true', help='Resume the checkpoint in save_path, or skip a completed run')
    parser.add_argument('--save_every', type=int, default=500, help='Replace the single checkpoint every N iterations')
    parser.add_argument('--feature_batch', type=int, default=128, help='Frozen feature extraction chunk size')

    args = parser.parse_args()
    assert args.prototype_pool_size == 0 or args.prototype_pool_size >= 2, 'prototype_pool_size must be 0 or at least 2'
    if args.distill_aug_strategy is None:
        args.distill_aug_strategy = 'color_crop_cutout_scale' if args.dataset == 'SVHN' else 'color_crop_cutout_flip_scale_rotate'
    args.distill_aug_strategy = 'color_cutout_scale'
    teacher_folder = {'FashionMNIST': 'fashionmnist/full/checkpoints','SVHN': 'digits_simclr/checkpoints'}.get(args.dataset, 'full_ssl_checkpoints_1000ep_3seeds')
    teacher_path = ROOT / 'Evaluate_benchamrk' / teacher_folder
    teacher_seed = 1 if args.dataset == 'TinyImageNet' else 0

    output = Path(args.save_path)
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / ('res_IIC_%s_ResNet18BN_%gpercent_final.pt' % (args.dataset, args.percentage))
    png_name = 'vis_IIC_%s_ResNet18BN_%gpercent_iter%%d.png' % (args.dataset, args.percentage)
    run_args = {key: getattr(args, key) for key in ('dataset', 'percentage', 'seed', 'Iteration', 'lr_img', 'batch_real','temperature', 'beta', 'prototype_pool_size', 'distill_aug_strategy','feature_batch', 'single_channel')}
    saved = None
    if args.resume and result_path.exists():
        saved = torch.load(result_path, map_location='cpu', weights_only=True)
        saved_args = {key: value for key, value in saved['run_args'].items() if key in run_args}
        saved_args.setdefault('prototype_pool_size', 0)
        assert (saved.get('init') == 'random' and saved.get('teacher_source') == 'pretrained' and saved.get('teacher_seed') == teacher_seed and saved_args == run_args), 'Incompatible resume checkpoint'
        assert 0 <= saved['iteration'] <= args.Iteration, 'Invalid saved iteration'
        if saved.get('complete'):
            print('SKIP completed: %s (iteration %d)' % (result_path, saved['iteration']), flush=True)
            return

    method = 'Bipartite_ViewTransition_MI'
    distill_dsa_param = ParamDiffAug()
    set_random_seed(args.seed)

    os.makedirs(args.data_path, exist_ok=True)
    data = reference_modules()['data']
    if args.dataset in ('FashionMNIST', 'SVHN'):
        raw_train, (mean, std) = digit_training_data(args)
    else:
        raw_train = data.RawDataset(args.data_path, args.dataset, True)
        mean, std = SYNTHETIC_NORMALIZATION[args.dataset]
    channel, im_size = 3, ((64, 64) if args.dataset == 'TinyImageNet' else (32, 32))
    dst_train = data.ImageView(raw_train, transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)]))
    num_syn = int(len(dst_train) * args.percentage / 100)
    num_pools = math.ceil(num_syn / args.prototype_pool_size) if args.prototype_pool_size else 1
    pool_sizes = [num_syn // num_pools + (index < num_syn % num_pools) for index in range(num_pools)]
    teacher_args = argparse.Namespace(**vars(args), model='ResNet18BN', teacher_source='pretrained', teacher_seed=teacher_seed, teacher_ssl='simclr', teacher_path=str(teacher_path))
    teacher = load_teacher(teacher_args, mean, std, im_size)
    encoder = frozen_embedder(teacher)

    if saved is None:
        image_syn = random_initialize(dst_train, num_syn, args.device, args.seed).detach().requires_grad_(True)
        image_syn_initial = image_syn.detach().clone()
    else:
        assert (saved['data'].shape == (num_syn, channel, *im_size) and saved['normalization'] == [mean, std]), 'Checkpoint data differs from this dataset'
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

    data_rng = torch.Generator().manual_seed(args.seed + 20_000)
    real_order = torch.randperm(len(dst_train), generator=data_rng)
    real_position = 0

    def real_batches():
        return iter(torch.utils.data.DataLoader(dst_train, batch_size=args.batch_real, sampler=real_order[real_position:].tolist(), num_workers=0, pin_memory=args.device == 'cuda', drop_last=True, generator=torch.Generator().manual_seed(args.seed)))

    augmenter = make_diff_augmenter(args.distill_aug_strategy, distill_dsa_param)
    objective_rng = np.random.default_rng(args.seed + 10_000)

    def embed(images):
        return torch.cat([checkpoint(encoder, block, use_reentrant=False) if block.requires_grad else encoder(block) for block in images.split(args.feature_batch)])

    # optimizer_img = torch.optim.Adam([image_syn], lr=args.lr_img)
    # scheduler_img = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_img, T_max=args.Iteration, eta_min=0.0)

    optimizer_img = torch.optim.Adam([image_syn], lr=args.lr_img, eps=1e-12)
    scheduler_img = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_img, T_max=args.Iteration, eta_min=args.lr_img * 0.01)

    loss_history = []
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
        assert len(loss_history) == start_iteration and 0 <= real_position <= len(real_order) == len(dst_train)
        print('RESUME iteration %d -> %d, including Adam, sampling and RNG state: %s' % (start_iteration, args.Iteration, result_path), flush=True)
    real_iter = real_batches()

    def save_checkpoint(iteration):
        numpy_rng = np.random.get_state()
        common.atomic_save(result_path, {
            'data': image_syn.detach().expand(-1, channel, -1, -1).contiguous().cpu(),
            'initial_data': image_syn_initial.detach().expand(-1, channel, -1, -1).contiguous().cpu(),
            'dataset': args.dataset, 'percentage': args.percentage, 'normalization': [mean, std],
            'method': method, 'iteration': iteration, 'complete': iteration == args.Iteration,
            'run_args': run_args, 'init': 'random', 'optimizer': optimizer_img.state_dict(), 'scheduler': scheduler_img.state_dict(),
            'rng': common.rng_state(), 'numpy_rng': (numpy_rng[0], numpy_rng[1].tolist(), *numpy_rng[2:]),
            'objective_rng': objective_rng.bit_generator.state, 'data_rng': data_rng.get_state(),
            'real_order': real_order, 'real_position': real_position,
            'dsa': True, 'teacher_source': 'pretrained', 'teacher_seed': teacher_seed,
            'seed': args.seed, 'model': 'ResNet18BN',
            'loss_history': torch.tensor(loss_history, dtype=torch.float64),
            'iic_params': {'tau': args.temperature,
                           'effective_tau': args.temperature * math.log(100.0) / math.log(max(pool_sizes[0], 100)),
                           'prototype_pool_size': args.prototype_pool_size,
                           'pool_sizes': pool_sizes,
                           'pool_temperatures': [args.temperature * math.log(100.0) / math.log(max(size, 100)) for size in pool_sizes],
                           'beta': args.beta,
                           'strategy': args.distill_aug_strategy,
                           'batch_real': args.batch_real}
        })
        print('Saved synthetic data to %s (iteration %d/%d)' % (result_path, iteration, args.Iteration), flush=True)

    if saved is None:
        save_checkpoint(0)
    del saved

    print('Hyper-parameters: \n', args.__dict__)
    print('Objective: negative beta-weighted bipartite mutual information, beta=%g, frozen teacher seed %d, pool sizes=%s.' % (args.beta, teacher_seed, pool_sizes), flush=True)
    print('%s distillation begins' % get_time(), flush=True)

    for iteration in range(start_iteration + 1, args.Iteration + 1):
        optimizer_img.zero_grad(set_to_none=True)
        try:
            batch = next(real_iter)
        except StopIteration:
            real_order = torch.randperm(len(dst_train), generator=data_rng)
            real_position = 0
            real_iter = real_batches()
            batch = next(real_iter)
        real_imgs = _images_from_batch(batch).to(args.device)
        real_position += len(real_imgs)

        seed_u, seed_v = [int(objective_rng.integers(0, 2**31 - 1)) for _ in range(2)]
        with torch.no_grad():
            features_u_real = embed(augmenter(real_imgs, seed_u))
            features_v_real = embed(augmenter(real_imgs, seed_v))
        images = image_syn.expand(-1, channel, -1, -1)
        features_u_syn = embed(augmenter(images, seed_u))
        features_v_syn = embed(augmenter(images, seed_v))
        # loss = bipartite_view_transition_loss(features_u_real, features_v_real, features_u_syn, features_v_syn, tau=args.temperature, beta=args.beta)
        # loss.backward()
        # optimizer_img.step()
        # scheduler_img.step()

        loss = bipartite_view_transition_loss(features_u_real, features_v_real, features_u_syn, features_v_syn, tau=args.temperature, beta=args.beta, pool_size=args.prototype_pool_size)
        loss.backward()
        if image_syn.grad is not None:
            # Smooth 1-pixel high-frequency teacher artifacts while preserving Adam's per-prototype step gating
            k = image_syn.grad.new_tensor([[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]]) / 16.0
            k = k.view(1, 1, 3, 3).repeat(image_syn.shape[1], 1, 1, 1)
            image_syn.grad.copy_(F.conv2d(image_syn.grad, k, padding=1, groups=image_syn.shape[1]))
        optimizer_img.step()
        scheduler_img.step()

        with torch.no_grad():
            image_syn.copy_(torch.maximum(torch.minimum(image_syn, pixel_max), pixel_min))

        objective = loss.item()
        loss_history.append(objective)
        del loss, features_u_real, features_v_real, features_u_syn, features_v_syn
        if iteration % 100 == 0:
            with torch.no_grad():
                drift = ((image_syn - image_syn_initial) * diag_std * 255).square().mean().sqrt().item()
            print('%s iter = %05d, loss = %.6f, pixel drift RMS = %.5f/255' % (get_time(), iteration, objective, drift), flush=True)
        if iteration == 1 or iteration == args.Iteration:
            grid_path = output / (png_name % iteration)
            visible = (image_syn.detach() * diag_std + diag_mean).clamp(0, 1).expand(-1, channel, -1, -1).cpu()
            save_image(visible, grid_path, nrow=int(np.ceil(np.sqrt(num_syn))))
        if iteration == 1 or iteration % args.save_every == 0 or iteration == args.Iteration:
            save_checkpoint(iteration)

    print('DONE: %s' % result_path, flush=True)


if __name__ == '__main__':
    main()
