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


def frozen_embedder(network):
    """Return the frozen feature extractor."""
    module = network.module if isinstance(network, nn.DataParallel) else network
    network.eval()
    for parameter in network.parameters():
        parameter.requires_grad_(False)
    return module.encoder if hasattr(module, 'encoder') else module.embed


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
    rng = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dst_train), generator=rng)[:num_syn]
    assert len(indices) == num_syn, 'Not enough real images for initialization'
    if init == 'random':
        print(f'Initializing {num_syn} synthetic images from random real training images (seed={seed})...', flush=True)
        return torch.stack([_images_from_batch(dst_train[int(index)]) for index in indices]).to(device)

    assert init == 'kmeans'
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


def bipartite_view_transition_loss(features_u_real, features_v_real, features_u_syn, features_v_syn, tau):
    """Compute assignment MI, displacement loss, and view-pair similarity loss."""
    u_r = F.normalize(features_u_real, dim=1)
    v_r = F.normalize(features_v_real, dim=1)
    u_s = F.normalize(features_u_syn, dim=1)
    v_s = F.normalize(features_v_syn, dim=1)

    # Bipartite similarities: u_r -> u_s and v_r -> v_s [Batch, Num_Syn]
    sim_u = torch.mm(u_r, u_s.t())
    sim_v = torch.mm(v_r, v_s.t())

    # Soft assignments to paired synthetic prototypes
    p_u = F.softmax(sim_u / tau, dim=1)
    p_v = F.softmax(sim_v / tau, dim=1)

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

    # Information-Theoretic Objective.
    parser.add_argument('--Iteration', type=int, default=2000)
    parser.add_argument('--lr_img', type=float, default=0.002, help='Higher LR, decays via CosineAnnealing')
    parser.add_argument('--batch_real', type=int, default=256)

    parser.add_argument('--temperature', type=float, default=0.2, help='Softmax temperature for prototype assignment')
    parser.add_argument('--disp_weight', type=nonnegative_float, default=0.0, help='Weight for feature displacement matching')
    parser.add_argument('--pair_weight', type=nonnegative_float, default=0.0, help='Weight for view-pair similarity matching')
    

    parser.add_argument('--num_random_networks', type=int, default=1)

    parser.add_argument('--random_models', default='ResNet18BN', help='Comma-separated P_net support')
    parser.add_argument('--distill_aug_strategy', '--dsa_strategy', '--dsa-strategy', default='color_crop_cutout_flip_scale_rotate')
    parser.add_argument('--dsa', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--distill_aug_mode', choices=['S', 'M'], default='M')

    # Teacher.
    parser.add_argument('--teacher_source', choices=['fresh', 'pretrained'], default='pretrained')
    parser.add_argument('--teacher_seed', type=int, nargs='+', default=[1], help='Saved teacher seeds; their losses are summed')
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
    if min(args.Iteration, args.save_every, args.feature_batch, args.batch_real, args.num_random_networks) < 1:
        parser.error('Iterations, saving interval, batches and augmentation draws must be positive')
    if not 0 < args.percentage <= 100:
        parser.error('percentage must be in (0, 100]')
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
    saved = None
    if args.resume and result_path.exists():
        saved = torch.load(result_path, map_location='cpu', weights_only=True)
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

    diag_mean = torch.tensor(mean, device=args.device).view(1, channel, 1, 1)
    diag_std = torch.tensor(std, device=args.device).view(1, channel, 1, 1)
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
            'data': image_syn.detach().cpu(), 'initial_data': image_syn_initial.detach().cpu(),
            'dataset': args.dataset, 'percentage': args.percentage, 'normalization': [mean, std],
            'method': method, 'iteration': iteration, 'complete': iteration == args.Iteration,
            'run_args': run_args, 'init': args.init, 'optimizer': optimizer_img.state_dict(), 'scheduler': scheduler_img.state_dict(),
            'rng': common.rng_state(), 'numpy_rng': (numpy_rng[0], numpy_rng[1].tolist(), *numpy_rng[2:]),
            'objective_rng': objective_rng.bit_generator.state, 'data_rng': data_rng.get_state(),
            'real_order': real_order, 'real_position': real_position,
            'dsa': args.dsa, 'teacher_source': args.teacher_source, 'teacher_seed': args.teacher_seed,
            'teacher_loss_reduction': 'sum', 'seed': args.seed, 'model': args.model,
            'loss_history': torch.tensor(loss_history, dtype=torch.float64),
            'negative_mi_history': torch.tensor(negative_mi_history, dtype=torch.float64),
            'shift_loss_history': torch.tensor(shift_loss_history, dtype=torch.float64),
            'total_loss_history': torch.tensor(total_loss_history, dtype=torch.float64),
            'iic_params': {'tau': args.temperature, 'disp_weight': args.disp_weight,
                           'pair_weight': args.pair_weight, 'strategy': args.distill_aug_strategy,
                           'batch_real': args.batch_real}
        })
        print('Saved synthetic data to %s (iteration %d/%d)' % (result_path, iteration, args.Iteration), flush=True)

    if saved is None:
        save_checkpoint(0)
    del saved

    print('Hyper-parameters: \n', args.__dict__)
    print('P_net support: %s' % ', '.join(random_models), flush=True)
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

            for teacher in teachers:
                model_name = random_models[int(objective_rng.integers(len(random_models)))]
                network_seed = int(objective_rng.integers(0, 2**31 - 1))
                network = teacher if teacher is not None else get_network(model_name, channel, num_classes, im_size, seed=network_seed).to(args.device)
                encoder = frozen_embedder(network)

                def embed(images):
                    # Recompute frozen activations in backward instead of retaining two full synthetic graphs.
                    return torch.cat([checkpoint(encoder, block, use_reentrant=False) if block.requires_grad else encoder(block)
                                      for block in images.split(args.feature_batch)])

                view_u_real = augmenter(real_imgs, seed_u_real)
                view_v_real = augmenter(real_imgs, seed_v_real)

                with torch.no_grad():
                    features_u_real = embed(view_u_real)
                    features_v_real = embed(view_v_real)

                # Differentiable views of S force spatial heterogeneity across crops
                view_u_syn = augmenter(image_syn, seed_u_syn)
                view_v_syn = augmenter(image_syn, seed_v_syn)
                features_u_syn = embed(view_u_syn)
                features_v_syn = embed(view_v_syn)

                negative_mi, disp_loss, pair_loss = bipartite_view_transition_loss(
                    features_u_real, features_v_real, features_u_syn, features_v_syn, tau=args.temperature
                )
                network_loss = (negative_mi + args.disp_weight * disp_loss
                                + args.pair_weight * pair_loss)

                (network_loss / args.num_random_networks).backward()

                sampled_losses.append(network_loss.detach())
                sampled_negative_mi_losses.append(negative_mi.detach())
                # Keep the historical combined diagnostic compatible with old checkpoint histories.
                sampled_shift_losses.append((disp_loss + 0.5 * pair_loss).detach())
                sampled_disp_losses.append(disp_loss.detach())
                sampled_pair_losses.append(pair_loss.detach())
                del encoder, network, network_loss, negative_mi, disp_loss, pair_loss
                del features_u_real, features_v_real, features_u_syn, features_v_syn
                del view_u_real, view_v_real, view_u_syn, view_v_syn

        objective = torch.stack(sampled_losses).sum().item() / args.num_random_networks
        negative_mi_value = torch.stack(sampled_negative_mi_losses).sum().item() / args.num_random_networks
        shift_loss_value = torch.stack(sampled_shift_losses).sum().item() / args.num_random_networks
        disp_loss_value = torch.stack(sampled_disp_losses).sum().item() / args.num_random_networks
        pair_loss_value = torch.stack(sampled_pair_losses).sum().item() / args.num_random_networks
        total_loss_value = objective

        optimizer_img.step()
        scheduler_img.step()

        with torch.no_grad():
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
            visible = (image_syn.detach() * diag_std + diag_mean).clamp(0, 1).cpu()
            save_image(visible, grid_path, nrow=int(np.ceil(np.sqrt(num_syn))))

        if iteration == 1 or iteration % args.save_every == 0 or iteration == args.Iteration:
            save_checkpoint(iteration)

    print('DONE: %s' % result_path, flush=True)



if __name__ == '__main__':
    main()
