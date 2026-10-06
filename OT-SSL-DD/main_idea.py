"""Teacher-free distillation with random-feature cross-covariance matching.

For a frozen randomly initialized encoder ``phi`` and two independent views,

    Sigma_D(phi) = E_t1,t2 [Phi(t1(D)).T @ Phi(t2(D)) / |D|].

The synthetic pixels minimize the Monte Carlo estimate of

    E_phi ||Sigma_T(phi) - Sigma_S(phi)||_F^2.

No encoder is trained during distillation. The only optimized tensor is the
synthetic image set; SSL training is used solely for optional final evaluation.
The statistic is intentionally *uncentered*, exactly as in the supplied
formulation (despite the conventional use of "cross-covariance" for a centered
statistic).
"""

import argparse
import math
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import save_image

from utils import (
    AUGMENT_FNS,
    DiffAugment,
    ParamDiffAug,
    evaluate_synset_SSL,
    get_dataset,
    get_eval_pool,
    get_network,
    get_time,
)


def set_random_seed(seed):
    """Seed initialization/evaluation while retaining explicit objective RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _images_from_batch(batch):
    return batch[0] if isinstance(batch, (tuple, list)) else batch


def cross_covariance(features_a, features_b):
    """Return the uncentered paired-view statistic ``A.T @ B / N``."""
    if features_a.ndim != 2 or features_b.ndim != 2:
        raise ValueError('Cross-covariance features must be rank-two matrices.')
    if len(features_a) == 0 or len(features_a) != len(features_b):
        raise ValueError('Paired views must contain the same positive number of rows.')
    return features_a.T @ features_b / len(features_a)


@torch.no_grad()
def estimate_real_cross_covariance(encoder, loader, augmenter, num_aug_pairs,
                                   rng, device):
    """Stream a dataset average and a Monte Carlo view expectation.

    ``augmenter(images, seed)`` must preserve the number/order of image rows.
    Every call receives an independently sampled seed, including the two calls
    within a pair. Only the final feature_dim-by-feature_dim target is stored.
    """
    if num_aug_pairs < 1:
        raise ValueError('num_aug_pairs must be positive.')
    total = None
    count = 0
    feature_shape = None
    for batch in loader:
        images = _images_from_batch(batch).to(device)
        if len(images) == 0:
            continue
        for _ in range(num_aug_pairs):
            view_a = augmenter(images, int(rng.integers(0, 2**31 - 1)))
            view_b = augmenter(images, int(rng.integers(0, 2**31 - 1)))
            features_a = encoder(view_a)
            features_b = encoder(view_b)
            if features_a.ndim != 2 or features_b.ndim != 2:
                raise ValueError('The random encoder must return rank-two features.')
            if len(features_a) != len(images) or features_a.shape != features_b.shape:
                raise ValueError('The two encoded views must have matching rows and dimensions.')
            if feature_shape is None:
                feature_shape = features_a.shape[1:]
                total = features_a.new_zeros((features_a.shape[1], features_b.shape[1]))
            elif features_a.shape[1:] != feature_shape:
                raise ValueError('The random encoder feature dimension changed between batches.')
            total.addmm_(features_a.T, features_b)
        count += len(images)
    if count == 0:
        raise ValueError('The real dataset must contain at least one image.')
    return total.div_(count * num_aug_pairs)


def synthetic_feature_pairs(encoder, images, augmenter, num_aug_pairs,
                            batch_size, rng):
    """Encode independent synthetic view pairs without detaching pixel graphs."""
    if images.ndim != 4 or len(images) == 0:
        raise ValueError('Synthetic images must be a nonempty NCHW tensor.')
    if num_aug_pairs < 1 or batch_size < 1:
        raise ValueError('num_aug_pairs and batch_size must be positive.')
    chunks = [[[], []] for _ in range(num_aug_pairs)]
    for image_batch in images.split(batch_size):
        for pair in chunks:
            view_a = augmenter(image_batch, int(rng.integers(0, 2**31 - 1)))
            view_b = augmenter(image_batch, int(rng.integers(0, 2**31 - 1)))
            pair[0].append(encoder(view_a))
            pair[1].append(encoder(view_b))
    pairs = [(torch.cat(a, dim=0), torch.cat(b, dim=0)) for a, b in chunks]
    first_shape = pairs[0][0].shape
    if first_shape[0] != len(images) or len(first_shape) != 2:
        raise ValueError('The random encoder must return one feature row per image.')
    if any(a.shape != first_shape or b.shape != first_shape for a, b in pairs):
        raise ValueError('All random-feature views must have the same shape.')
    return pairs


def cross_covariance_matching_loss(synthetic_pairs, real_cross_covariance,
                                   covariance_block_size=256):
    """Compute ``||mean_pair(A.T @ B / M) - Sigma_real||_F^2`` exactly.

    Row blocking bounds the largest differentiable temporary while retaining
    every matrix entry. Averaging the matrices before squaring matches the
    expectation inside Sigma in the mathematical formulation.
    """
    if not synthetic_pairs:
        raise ValueError('At least one synthetic augmentation pair is required.')
    if covariance_block_size < 1:
        raise ValueError('covariance_block_size must be positive.')
    first_a, first_b = synthetic_pairs[0]
    if first_a.ndim != 2 or first_b.ndim != 2 or len(first_a) == 0:
        raise ValueError('Synthetic feature pairs must be nonempty rank-two matrices.')
    if len(first_a) != len(first_b):
        raise ValueError('Synthetic view pairs must have corresponding image rows.')
    expected_shape = (first_a.shape[1], first_b.shape[1])
    if real_cross_covariance.shape != expected_shape:
        raise ValueError('Real and synthetic cross-covariance dimensions must match.')
    for features_a, features_b in synthetic_pairs:
        if features_a.shape != first_a.shape or features_b.shape != first_b.shape:
            raise ValueError('All synthetic augmentation pairs must have matching shapes.')

    denominator = len(first_a) * len(synthetic_pairs)
    loss = first_a.new_zeros(())
    for start in range(0, first_a.shape[1], covariance_block_size):
        end = min(start + covariance_block_size, first_a.shape[1])
        estimate = first_a.new_zeros((end - start, first_b.shape[1]))
        for features_a, features_b in synthetic_pairs:
            estimate = estimate + features_a[:, start:end].T @ features_b
        error = estimate.div(denominator) - real_cross_covariance[start:end]
        loss = loss + error.square().sum()
    return loss


def frozen_embedder(network):
    """Return a frozen random encoder; classifier parameters remain unused."""
    module = network.module if isinstance(network, nn.DataParallel) else network
    if not hasattr(module, 'embed'):
        raise ValueError('%s does not provide embed().' % type(module).__name__)
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
            # seed=-1 means DiffAugment samples each image independently. The
            # enclosing RNG seed makes the sampled view replayable for tests.
            return DiffAugment(images, strategy, seed=-1, param=param)

    return augment


def main():
    parser = argparse.ArgumentParser(
        description='Teacher-free random-feature cross-covariance distillation'
    )

    # Data.
    parser.add_argument('--dataset', default='CIFAR10')
    parser.add_argument('--percentage', type=float, default=1.0)
    parser.add_argument('--data_path', default='/home/mmoslem3/scratch/data')
    parser.add_argument('--device', choices=['cpu', 'cuda'],
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=0)

    # Random-feature cross-covariance objective.
    parser.add_argument('--Iteration', type=int, default=100)
    parser.add_argument('--lr_img', type=float, default=0.01)
    parser.add_argument('--batch_real', type=int, default=256)
    parser.add_argument('--batch_syn', type=int, default=256)
    parser.add_argument('--num_random_networks', type=int, default=1,
                        help='fresh samples from P_net per pixel update')
    parser.add_argument('--num_aug_pairs', type=int, default=1,
                        help='Monte Carlo samples of independent (t1, t2) per network')
    parser.add_argument('--covariance_block_size', type=int, default=256)
    parser.add_argument('--random_models', default=None,
                        help='comma-separated P_net support; defaults to --model')
    parser.add_argument('--distill_aug_strategy',
                        default='color_crop_cutout_flip_scale_rotate')
    parser.add_argument('--distill_aug_mode', choices=['S', 'M'], default='M')

    # Downstream SSL evaluation (not part of distillation).
    parser.add_argument('--model', default='ConvNet')
    parser.add_argument('--lr_net', type=float, default=0.01)
    parser.add_argument('--batch_train', type=int, default=256)
    parser.add_argument('--ssl_method', default='simclr')
    parser.add_argument('--ssl_aug_strategy', default=None)
    parser.add_argument('--projection_dim', type=int, default=128)
    parser.add_argument('--temperature', type=float, default=0.5)
    parser.add_argument('--barlow_lambda', type=float, default=0.005)
    parser.add_argument('--eval_mode', default='S')
    parser.add_argument('--num_eval', type=int, default=1)
    parser.add_argument('--epoch_eval_train', type=int, default=1000)
    parser.add_argument('--label_percentage', type=float, default=1.0)
    parser.add_argument('--epoch_linear_train', type=int, default=100)
    parser.add_argument('--lr_linear', type=float, default=0.1)
    parser.add_argument('--batch_linear', type=int, default=256)

    # Output.
    parser.add_argument('--save_path', default='result')
    parser.add_argument('--save_every', type=int, default=20,
                        help='visualization interval; 0 saves only the final grid')

    args = parser.parse_args()
    if not 0 < args.percentage <= 100:
        parser.error('--percentage must be in (0, 100].')
    if args.Iteration < 0 or args.num_eval < 0:
        parser.error('--Iteration and --num_eval must be nonnegative.')
    if min(args.batch_real, args.batch_syn, args.num_random_networks,
           args.num_aug_pairs, args.covariance_block_size) < 1:
        parser.error('Objective counts and batch/block sizes must be positive.')
    if not math.isfinite(args.lr_img) or args.lr_img <= 0:
        parser.error('--lr_img must be finite and positive.')
    if args.seed < 0 or args.save_every < 0:
        parser.error('--seed and --save_every must be nonnegative.')
    if args.distill_aug_strategy.lower() not in ('', 'none'):
        unknown = set(args.distill_aug_strategy.split('_')) - set(AUGMENT_FNS)
        if unknown:
            parser.error('Unknown augmentation groups: ' + ', '.join(sorted(unknown)))

    random_models = [name.strip() for name in
                     (args.random_models or args.model).split(',') if name.strip()]
    if not random_models:
        parser.error('--random_models must include at least one architecture.')
    if args.ssl_aug_strategy is None:
        args.ssl_aug_strategy = args.distill_aug_strategy
    args.method = 'RandomFeatureCrossCovariance'
    args.dsa_param = ParamDiffAug()
    args.dsa_param.aug_mode = args.distill_aug_mode
    args.dsa = False
    set_random_seed(args.seed)

    os.makedirs(args.data_path, exist_ok=True)
    os.makedirs(args.save_path, exist_ok=True)
    channel, im_size, num_classes, _, mean, std, dst_train, _, testloader = get_dataset(
        args.dataset, args.data_path
    )
    num_syn = int(len(dst_train) * args.percentage / 100)
    if len(dst_train) == 0 or num_syn < 1:
        parser.error('The real and synthetic datasets must each contain at least one image.')

    initial_indices = np.random.RandomState(args.seed).permutation(len(dst_train))[:num_syn]
    image_syn = torch.stack([dst_train[int(index)][0] for index in initial_indices])
    image_syn = image_syn.to(args.device).detach().requires_grad_(True)
    image_syn_initial = image_syn.detach().clone()
    diag_mean = torch.tensor(mean, device=args.device).view(1, channel, 1, 1)
    diag_std = torch.tensor(std, device=args.device).view(1, channel, 1, 1)
    pixel_min = -diag_mean / diag_std
    pixel_max = (1 - diag_mean) / diag_std

    def to_uint8(images):
        return ((images * diag_std + diag_mean) * 255 + 0.5).clamp(0, 255).to(torch.uint8)

    initial_uint8 = to_uint8(image_syn_initial)
    real_loader = torch.utils.data.DataLoader(
        dst_train, batch_size=args.batch_real, shuffle=False, num_workers=0,
        pin_memory=args.device == 'cuda', drop_last=False,
    )
    augmenter = make_diff_augmenter(args.distill_aug_strategy, args.dsa_param)
    objective_rng = np.random.default_rng(args.seed + 10_000)
    optimizer_img = torch.optim.Adam([image_syn], lr=args.lr_img)
    loss_history = []

    print('Hyper-parameters: \n', args.__dict__)
    print('P_net support: %s' % ', '.join(random_models), flush=True)
    print('Objective: mean_phi ||Sigma_T(phi) - Sigma_S(phi)||_F^2; '
          'encoders are randomly initialized and never trained.', flush=True)
    print('%s distillation begins' % get_time(), flush=True)

    for iteration in range(1, args.Iteration + 1):
        optimizer_img.zero_grad(set_to_none=True)
        sampled_losses = []
        sampled_models = []
        for _ in range(args.num_random_networks):
            model_name = random_models[int(objective_rng.integers(len(random_models)))]
            network_seed = int(objective_rng.integers(0, 2**31 - 1))
            network = get_network(
                model_name, channel, num_classes, im_size, seed=network_seed
            ).to(args.device)
            encoder = frozen_embedder(network)
            real_statistic = estimate_real_cross_covariance(
                encoder, real_loader, augmenter, args.num_aug_pairs,
                objective_rng, args.device,
            )
            synthetic_pairs = synthetic_feature_pairs(
                encoder, image_syn, augmenter, args.num_aug_pairs,
                args.batch_syn, objective_rng,
            )
            network_loss = cross_covariance_matching_loss(
                synthetic_pairs, real_statistic, args.covariance_block_size
            )
            if not torch.isfinite(network_loss).item():
                raise FloatingPointError('Non-finite cross-covariance loss.')
            (network_loss / args.num_random_networks).backward()
            sampled_losses.append(network_loss.detach())
            sampled_models.append('%s(seed=%d,d=%d)' % (
                model_name, network_seed, real_statistic.shape[0]
            ))
            del synthetic_pairs, real_statistic, encoder, network, network_loss

        if image_syn.grad is None or not torch.isfinite(image_syn.grad).all().item():
            raise FloatingPointError('Non-finite or missing synthetic pixel gradient.')
        objective = torch.stack(sampled_losses).mean().item()
        optimizer_img.step()
        with torch.no_grad():
            image_syn.copy_(torch.maximum(torch.minimum(image_syn, pixel_max), pixel_min))
            if not torch.isfinite(image_syn).all().item():
                raise FloatingPointError('Non-finite synthetic pixel update.')
            changed = (to_uint8(image_syn) != initial_uint8).float().mean().item() * 100
            drift = ((image_syn - image_syn_initial) * diag_std * 255).square().mean().sqrt().item()
        loss_history.append(objective)
        print('%s iter = %05d, loss = %.10f, pixel drift RMS = %.5f/255, '
              'PNG values changed = %.5f%%, samples = %s' % (
                  get_time(), iteration, objective, drift, changed,
                  '; '.join(sampled_models)
              ), flush=True)

        if args.save_every and iteration % args.save_every == 0:
            grid_path = os.path.join(
                args.save_path,
                'vis_RFC_%s_%s_%gpercent_iter%d.png' % (
                    args.dataset, args.model, args.percentage, iteration
                ),
            )
            visible = (image_syn.detach() * diag_std + diag_mean).clamp(0, 1).cpu()
            save_image(visible, grid_path, nrow=int(np.ceil(np.sqrt(num_syn))))

    final_grid_path = os.path.join(
        args.save_path,
        'vis_RFC_%s_%s_%gpercent_iter%d.png' % (
            args.dataset, args.model, args.percentage, args.Iteration
        ),
    )
    visible = (image_syn.detach() * diag_std + diag_mean).clamp(0, 1).cpu()
    save_image(visible, final_grid_path, nrow=int(np.ceil(np.sqrt(num_syn))))

    result_path = os.path.join(
        args.save_path,
        'res_RFC_%s_%s_%gpercent.pt' % (args.dataset, args.model, args.percentage),
    )
    torch.save({
        'data': image_syn.detach().cpu(),
        'method': args.method,
        'iteration': args.Iteration,
        # A tensor keeps checkpoints compatible with restricted
        # ``torch.load(..., weights_only=True)`` implementations.
        'loss_history': torch.tensor(loss_history, dtype=torch.float64),
        'seed': args.seed,
        'random_feature_distribution': {
            'models': random_models,
            'samples_per_update': args.num_random_networks,
            'trained': False,
        },
        'cross_covariance': {
            'centered': False,
            'augmentation_pairs': args.num_aug_pairs,
            'strategy': args.distill_aug_strategy,
            'mode': args.distill_aug_mode,
            'real_batch_size': args.batch_real,
            'synthetic_batch_size': args.batch_syn,
        },
    }, result_path)
    print('Saved synthetic data to %s' % result_path, flush=True)

    # Evaluation happens only after the teacher-free objective has finished and
    # the result is safely stored. Evaluation networks are not objective inputs.
    if args.num_eval:
        model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)
        for model_index, model_eval in enumerate(model_eval_pool):
            accuracies = []
            for eval_index in range(args.num_eval):
                eval_seed = args.seed + 1_000_000 + model_index * args.num_eval + eval_index
                set_random_seed(eval_seed)
                network = get_network(
                    model_eval, channel, num_classes, im_size, seed=eval_seed
                ).to(args.device)
                _, _, accuracy = evaluate_synset_SSL(
                    eval_index, network, image_syn.detach().clone(),
                    dst_train, testloader, args,
                )
                accuracies.append(float(accuracy))
            print('Evaluate %d random %s, mean = %.4f std = %.4f' % (
                len(accuracies), model_eval, np.mean(accuracies), np.std(accuracies)
            ), flush=True)


if __name__ == '__main__':
    main()
