"""Match clean/view scattering moments and paired-view cross-covariances.

Additional dependency: python -m pip install kymatio==0.3.0
SSL networks are trained only for downstream evaluation, never for distillation.
"""

import argparse
import math
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torchvision.utils import save_image

from dm_dsa import DMDSA
from eval import ROOT, evaluate_synset_SSL, reference_modules
from utils import get_dataset, get_time, ParamDiffAug, DiffAugment, AUGMENT_FNS

sys.path.append(str(ROOT / '_extra'))
from scattering_moments import (
    ScatteringFeatures,
    backward_scattering_moments,
    backward_scattering_views,
    compute_real_moments,
)

#  random ConvNet, mean = 0.4066 std = 0.0104

# Evaluate 6 random ConvNet, mean = 0.3763 std = 0.0113

def set_random_seed(seed):
    """Seed every RNG used by distillation and SSL evaluation."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_determinism(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    set_random_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


class AugmentedScattering(torch.nn.Module):
    """One sampled Siamese DSA transform, replayed through the pixel gradient."""

    def __init__(self, scattering, strategy, mode, replay_seed, dsa=None):
        super().__init__()
        self.scattering = scattering
        self.strategy = strategy
        self.replay_seed = replay_seed
        self.param = ParamDiffAug()
        self.param.aug_mode = mode
        self.dsa = dsa
        self.cuda_devices = list(range(torch.cuda.device_count()))

    def forward(self, images):
        # The replay seed is freshly sampled for each view, not fixed by --seed.
        # DSA's Siamese mode shares its parameters across real/synthetic images
        # and across chunks. Preserve the surrounding training/evaluation RNGs.
        with torch.random.fork_rng(devices=self.cuda_devices):
            # CPU augmentation permits scale/rotate pixel backpropagation under
            # strict deterministic algorithms (CUDA grid_sample backward does
            # not). Device copies retain the gradient to the original pixels.
            if self.dsa is not None:
                augmented = self.dsa.normalized(images.cpu(), self.replay_seed)
            else:
                augmented = DiffAugment(
                    images.cpu(), self.strategy, seed=self.replay_seed, param=self.param
                )
        return self.scattering(augmented.to(images.device))


def main():
    parser = argparse.ArgumentParser(description='Wavelet scattering moment distillation')

    # -------------------- Data --------------------
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
    parser.add_argument('--percentage', type=float, default=0.2, help='percentage of total data size')
    parser.add_argument('--data_path', type=str, default=str(ROOT.parent / 'data'), help='dataset path')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=0, help='seed for distillation initialization and evaluation')

    # -------------------- Distillation --------------------
    parser.add_argument('--Iteration', type=int, default=8000, help='number of synthetic pixel updates')
    parser.add_argument('--lr_img', type=float, default=0.001, help='Adam learning rate for synthetic pixels')
    parser.add_argument('--save_every', type=int, default=500)
    parser.add_argument('--resume', type=str, default=None, help='continue synthetic images from this checkpoint')
    parser.add_argument('--batch_real', type=int, default=4096, help='batch size for caching full real moments')
    parser.add_argument('--batch_syn', type=int, default=512, help='synthetic scattering batch size; moments use all images')
    parser.add_argument('--scattering_J', type=int, default=2, help='log2 scattering scale (at least 2)')
    parser.add_argument('--scattering_L', type=int, default=8, help='number of wavelet orientations')
    parser.add_argument('--gamma', type=float, default=5.0, help='weight of squared mean distance; covariance weight is 1')
    parser.add_argument('--covariance_block_size', type=int, default=1024, help='rows per exact covariance-loss block')
    parser.add_argument('--distill_aug_views', type=int, default=2, help='fresh views per update; clean-only mode requires 0 together with --pair_weight 0')
    parser.add_argument('--distill_aug_strategy', '--dsa_strategy', '--dsa-strategy', type=str, default='color_crop_cutout_flip_scale_rotate', help='DSA operations for distillation views')
    parser.add_argument('--distill_aug_mode', choices=['S', 'M'], default='M', help='S: sample one operation group per view; M: compose all listed groups')
    parser.add_argument('--dsa', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--pair_weight', type=float, default=1.0, help='weight of summed paired-view cross-covariance losses; 0 disables')

    # -------------------- SSL evaluation --------------------
    parser.add_argument('--model', choices=['ConvNet', 'ConvNetBN', 'ResNet18', 'ResNet18BN', 'VGG11', 'VGG11BN'], default='ResNet18BN')
    parser.add_argument('--ssl_method', '--ssl-method', choices=['simclr', 'barlowtwins'], default='simclr')
    parser.add_argument('--num_eval', '--runs', type=int, default=1, help='number of evaluation seeds; 0 skips evaluation')
    parser.add_argument('--epoch_eval_train', '--ssl-epochs', type=int, default=None)
    parser.add_argument('--ssl-log-every', type=int, default=100)
    parser.add_argument('--label_percentage', '--label-percentage', type=int, choices=[1, 5], default=1)
    parser.add_argument('--epoch_linear_train', '--probe-epochs', type=int, default=100)
    parser.add_argument('--lr_linear', '--lr-linear', type=float, default=0.1)
    parser.add_argument('--batch_linear', '--batch-linear', type=int, default=256)

    # -------------------- Output --------------------
    parser.add_argument('--save_path', type=str, default=str(ROOT / 'OT-SSL-DD/result_Scattering'), help='path to save results')

    args = parser.parse_args()
    if not 0 < args.percentage <= 100:
        parser.error('--percentage must be in (0, 100].')
    if args.Iteration < 0 or args.num_eval < 0:
        parser.error('--Iteration and --num_eval must be nonnegative.')
    if min(args.batch_real, args.batch_syn, args.covariance_block_size, args.save_every) < 1:
        parser.error('Scattering and covariance batch sizes must be positive.')
    if not math.isfinite(args.gamma) or args.gamma < 0:
        parser.error('--gamma must be finite and nonnegative.')
    if not math.isfinite(args.lr_img) or args.lr_img <= 0:
        parser.error('--lr_img must be finite and positive.')
    if args.seed < 0:
        parser.error('--seed must be nonnegative.')
    if args.distill_aug_views < 0:
        parser.error('--distill_aug_views must be nonnegative.')
    if not math.isfinite(args.pair_weight) or args.pair_weight < 0:
        parser.error('--pair_weight must be finite and nonnegative.')
    if args.pair_weight > 0 and args.distill_aug_views < 2:
        parser.error('Paired-view matching needs --distill_aug_views >= 2; use --pair_weight 0 to disable it.')
    if args.distill_aug_views:
        if args.batch_real < 2:
            parser.error('Augmented covariance needs --batch_real >= 2.')
        if set(args.distill_aug_strategy.split('_')) - set(AUGMENT_FNS):
            parser.error('--distill_aug_strategy must contain groups from: ' + ', '.join(AUGMENT_FNS))
    args.method = 'Scattering'
    if args.dsa:
        args.distill_aug_mode = 'S'
    configure_determinism(args.seed)

    os.makedirs(args.save_path, exist_ok=True)

    channel, im_size, num_classes, _, mean, std, dst_train, _, testloader = get_dataset(args.dataset, args.data_path)
    num_syn = int(len(dst_train) * args.percentage / 100)
    if len(dst_train) < 2 or num_syn < 2:
        parser.error('Unbiased covariance requires at least two real and two synthetic images; increase --percentage.')
    scattering = ScatteringFeatures(im_size, J=args.scattering_J, L=args.scattering_L).to(args.device).eval()
    augmentation = DMDSA(args.distill_aug_strategy, mean, std) if args.dsa else None
    print('Hyper-parameters: \n', args.__dict__)

    # Keep the real dataset on the host; only scattering batches go to the device.
    indices = np.random.RandomState(args.seed).permutation(len(dst_train))[:num_syn]
    image_syn = torch.stack([dst_train[int(i)][0] for i in indices]).to(args.device).detach().requires_grad_(True)
    image_syn_initial = image_syn.detach().clone()
    start_iteration = 0
    resumed = torch.load(args.resume, map_location='cpu', weights_only=True) if args.resume else None
    if resumed is not None:
        if (any(resumed[key] != getattr(args, key) for key in ('dataset', 'model', 'percentage', 'seed', 'dsa'))
                or resumed['data'].shape != image_syn.shape):
            raise ValueError('Resume checkpoint does not match this run')
        start_iteration = resumed['iteration']
        if not 0 <= start_iteration <= args.Iteration:
            raise ValueError('Resume iteration exceeds the requested stopping iteration')
        with torch.no_grad():
            image_syn.copy_(resumed['data'].to(args.device))
        image_syn_initial = resumed.get('image_syn_init', image_syn_initial).to(args.device)
    diag_mean = torch.tensor(mean, device=args.device).view(1, channel, 1, 1)
    diag_std = torch.tensor(std, device=args.device).view(1, channel, 1, 1)

    def to_uint8(images):
        return ((images * diag_std + diag_mean) * 255 + 0.5).clamp(0, 255).to(torch.uint8)

    image_syn_init_uint8 = to_uint8(image_syn_initial)

    # Phi is fixed, so the full-dataset real statistics are computed only once.
    # Inputs use the same dataset normalization as the downstream SSL evaluator.
    with torch.no_grad():
        feature_dim = scattering(image_syn[:1]).shape[1]
    covariance_mib = feature_dim ** 2 * image_syn.element_size() / 1024 ** 2
    print('%s caching real scattering moments: %d features, %.1f MiB covariance' % (
        get_time(), feature_dim, covariance_mib
    ), flush=True)
    real_loader = torch.utils.data.DataLoader(
        dst_train, batch_size=args.batch_real, shuffle=False, num_workers=0, drop_last=False
    )
    real_mean, real_covariance = compute_real_moments(scattering, real_loader, args.device)

    augmentation_rng = np.random.default_rng(args.seed)
    pair_count = args.distill_aug_views * (args.distill_aug_views - 1) // 2
    print('Loss = clean + sum of %d view moment losses + %.4g * sum of %d cross-view covariance losses (real batch size %d)' % (
        args.distill_aug_views, args.pair_weight, pair_count, min(args.batch_real, len(dst_train))
    ), flush=True)
    print('%s distillation begins' % get_time(), flush=True)
    optimizer_img = torch.optim.Adam([image_syn], lr=args.lr_img)
    if resumed is not None:
        if 'optimizer_state' in resumed:
            optimizer_img.load_state_dict(resumed['optimizer_state'])
        else:
            print('Resume checkpoint has no Adam state; continuing saved images with fresh Adam.', flush=True)
        if 'augmentation_rng' in resumed:
            augmentation_rng.bit_generator.state = resumed['augmentation_rng']
        print('Resuming iteration %d -> %d from %s' % (start_iteration, args.Iteration, args.resume), flush=True)
    common = reference_modules()['common']
    save_name = os.path.join(args.save_path, 'res_Scattering-SSL_%s_%s_%gpercent.pt' % (
        args.dataset, args.model, args.percentage
    ))

    def save(iteration):
        preview = os.path.join(args.save_path, 'vis_%s_%s_%s_%gpercent_iter%d.png' % (
            args.method, args.dataset, args.model, args.percentage, iteration
        ))
        pixels = (image_syn.detach() * diag_std + diag_mean).clamp(0, 1).cpu()
        save_image(pixels, preview, nrow=int(np.ceil(np.sqrt(num_syn))))
        common.atomic_save(save_name, {
            'data': image_syn.detach().cpu(), 'normalization': [mean, std],
            'dataset': args.dataset, 'model': args.model, 'percentage': args.percentage,
            'method': args.method, 'iteration': iteration,
            'scattering': {'J': args.scattering_J, 'L': args.scattering_L, 'max_order': 2},
            'gamma': args.gamma, 'pair_weight': args.pair_weight, 'seed': args.seed,
            'dsa': args.dsa,
            'augmentation': augmentation.configuration() if args.dsa else None,
            'optimizer': 'adam', 'lr_img': args.lr_img,
            'optimizer_state': optimizer_img.state_dict(), 'image_syn_init': image_syn_initial.detach().cpu(),
            'augmentation_rng': augmentation_rng.bit_generator.state, 'resumed_from': args.resume,
            'distill_augmentation': {
                'views': args.distill_aug_views, 'strategy': args.distill_aug_strategy,
                'mode': args.distill_aug_mode, 'batch_real': args.batch_real, 'seed': args.seed,
                'cross_view_pairs': 'all_unordered_pairs_with_corresponding_image_rows',
            },
        })
        print('Saved synthetic data to %s' % save_name, flush=True)


    for it in range(start_iteration + 1, args.Iteration + 1):
        optimizer_img.zero_grad(set_to_none=True)
        loss, mean_loss, covariance_loss = backward_scattering_moments(
            image_syn, scattering, real_mean, real_covariance,
            gamma=args.gamma, batch_size=args.batch_syn,
            covariance_block_size=args.covariance_block_size,
        )
        clean_loss = loss.item()
        view_losses = []
        pair_loss = image_syn.new_zeros(())
        if args.distill_aug_views:
            real_indices = augmentation_rng.choice(
                len(dst_train), size=min(args.batch_real, len(dst_train)), replace=False
            )
            real_batch = torch.stack([dst_train[int(i)][0] for i in real_indices]).to(args.device)
            views = []
            real_view_features = []
            for _ in range(args.distill_aug_views):
                # Reuse this sampled transform only within this view's real,
                # synthetic, and backward passes. Resample on the next view.
                view = AugmentedScattering(
                    scattering, args.distill_aug_strategy, args.distill_aug_mode,
                    replay_seed=int(augmentation_rng.integers(0, 100000)), dsa=augmentation,
                )
                views.append(view)
                with torch.no_grad():
                    # Every view uses the SAME image rows, preserving pairs.
                    real_view_features.append(torch.cat([
                        view(batch) for batch in real_batch.split(args.batch_syn)
                    ], dim=0))
            augmented_loss, augmented_mean, augmented_covariance, pair_loss, view_terms = backward_scattering_views(
                image_syn, views, real_view_features,
                gamma=args.gamma, pair_weight=args.pair_weight,
                batch_size=args.batch_syn, covariance_block_size=args.covariance_block_size,
            )
            loss = loss + augmented_loss
            mean_loss = mean_loss + augmented_mean
            covariance_loss = covariance_loss + augmented_covariance
            view_losses = [term.item() for term in view_terms]
            del real_batch, real_view_features, views, view
        optimizer_img.step()
        with torch.no_grad():
            if not torch.isfinite(image_syn).all().item():
                raise FloatingPointError('Synthetic pixel update is non-finite; reduce --lr_img.')
            image_syn.copy_(torch.maximum(torch.minimum(image_syn, (1 - diag_mean) / diag_std), -diag_mean / diag_std))
            changed = (to_uint8(image_syn) != image_syn_init_uint8).float().mean().item() * 100
            pixel_change = ((image_syn - image_syn_initial) * diag_std * 255).square().mean().sqrt().item()
        if it%5 == 0 or it ==1:
            print('%s iter = %05d, loss = %.10f, covariance = %.10f, mean = %.10f, pair = %.10f, weighted pair = %.10f, clean = %.10f, augmented views = %s, PNG values changed = %.5f%%, RMS = %.2f/255' % (
                get_time(), it, loss.item(), covariance_loss.item(), mean_loss.item(),
                pair_loss.item(), args.pair_weight * pair_loss.item(),
                clean_loss, ', '.join('%.10f' % value for value in view_losses), changed, pixel_change
            ), flush=True)

        if it % args.save_every == 0 or it == args.Iteration:
            save(it)
    if args.Iteration == 0:
        save(0)

    # Release the dense target before training the final evaluation networks.
    del real_mean, real_covariance, scattering
    torch.use_deterministic_algorithms(False)
    saved = torch.load(save_name, map_location='cpu', weights_only=True)
    output = Path(args.save_path) / 'evaluation' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    accuracies = []
    for run in range(args.num_eval):
        _, _, accuracy = evaluate_synset_SSL(run, args.model, saved['data'], args,
                                             output / ('seed_%03d' % (args.seed + run)), saved['normalization'])
        accuracies.append(accuracy * 100)
        common.atomic_json(output / 'summary.json', {
            'config': dict(vars(args), synthetic_checkpoint=str(Path(save_name).resolve())),
            'complete': len(accuracies) == args.num_eval, 'accuracies_percent': accuracies,
            'mean_percent': float(np.mean(accuracies)),
        })
        print('RESULT seed=%d accuracy=%.2f%%' % (args.seed + run, accuracy * 100), flush=True)


if __name__ == '__main__':
    main()
