"""Match clean/view scattering moments and paired-view cross-covariances.

Additional dependency: python -m pip install kymatio==0.3.0
SSL networks are trained only for downstream evaluation, never for distillation.
"""

import argparse
import math
import os
import random

import numpy as np
import torch
from torchvision.utils import save_image

from scattering_moments import (
    ScatteringFeatures,
    backward_scattering_moments,
    backward_scattering_views,
    compute_real_moments,
)
from utils import (
    get_dataset, get_network, get_eval_pool, evaluate_synset_SSL, get_time,
    ParamDiffAug, DiffAugment, AUGMENT_FNS,
)



# Evaluate 6 random ConvNet, mean = 0.3763 std = 0.0113

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

    def __init__(self, scattering, strategy, mode, replay_seed):
        super().__init__()
        self.scattering = scattering
        self.strategy = strategy
        self.replay_seed = replay_seed
        self.param = ParamDiffAug()
        self.param.aug_mode = mode
        self.cuda_devices = list(range(torch.cuda.device_count()))

    def forward(self, images):
        # The replay seed is freshly sampled for each view, not fixed by --seed.
        # DSA's Siamese mode shares its parameters across real/synthetic images
        # and across chunks. Preserve the surrounding training/evaluation RNGs.
        with torch.random.fork_rng(devices=self.cuda_devices):
            # CPU augmentation permits scale/rotate pixel backpropagation under
            # strict deterministic algorithms (CUDA grid_sample backward does
            # not). Device copies retain the gradient to the original pixels.
            augmented = DiffAugment(
                images.cpu(), self.strategy, seed=self.replay_seed, param=self.param
            )
        return self.scattering(augmented.to(images.device))


def main():
    parser = argparse.ArgumentParser(description='Wavelet scattering moment distillation')

    # -------------------- Data --------------------
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
    parser.add_argument('--percentage', type=int, default=1, help='percentage of total data size')
    parser.add_argument('--data_path', type=str, default='/home/mmoslem3/scratch/data', help='dataset path')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=0, help='seed for distillation initialization and evaluation')

    # -------------------- Distillation --------------------
    parser.add_argument('--Iteration', type=int, default=4000, help='number of synthetic pixel updates')
    parser.add_argument('--lr_img', type=float, default=0.1, help='Adam learning rate for synthetic pixels')
    parser.add_argument('--batch_real', type=int, default=4096, help='batch size for caching full real moments')
    parser.add_argument('--batch_syn', type=int, default=512, help='synthetic scattering batch size; moments use all images')
    parser.add_argument('--scattering_J', type=int, default=2, help='log2 scattering scale (at least 2)')
    parser.add_argument('--scattering_L', type=int, default=8, help='number of wavelet orientations')
    parser.add_argument('--gamma', type=float, default=5.0, help='weight of squared mean distance; covariance weight is 1')
    parser.add_argument('--covariance_block_size', type=int, default=1024, help='rows per exact covariance-loss block')
    parser.add_argument('--distill_aug_views', type=int, default=2, help='fresh views per update; clean-only mode requires 0 together with --pair_weight 0')
    parser.add_argument('--distill_aug_strategy', type=str, default='color_crop_cutout_flip_scale_rotate', help='DSA operations for distillation views')
    parser.add_argument('--distill_aug_mode', choices=['S', 'M'], default='M', help='S: sample one operation group per view; M: compose all listed groups')
    parser.add_argument('--pair_weight', type=float, default=1.0, help='weight of summed paired-view cross-covariance losses; 0 disables')

    # -------------------- SSL evaluation --------------------
    parser.add_argument('--model', type=str, default='ConvNet', help='evaluation model')
    parser.add_argument('--lr_net', type=float, default=0.01, help='SSL evaluation learning rate')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for SSL evaluation')
    parser.add_argument('--ssl_method', type=str, default='simclr', help='simclr/barlowtwins')
    parser.add_argument('--ssl_aug_strategy', type=str, default=None, help='defaults to --distill_aug_strategy; must match it when augmented distillation is enabled')
    parser.add_argument('--projection_dim', type=int, default=128, help='projection dimension for SSL evaluation')
    parser.add_argument('--temperature', type=float, default=0.5, help='temperature for SimCLR')
    parser.add_argument('--barlow_lambda', type=float, default=0.005, help='off-diagonal weight for Barlow Twins')
    parser.add_argument('--eval_mode', type=str, default='S', help='evaluation architecture pool')
    parser.add_argument('--num_eval', type=int, default=4, help='number of random evaluation networks; 0 skips evaluation')
    parser.add_argument('--epoch_eval_train', type=int, default=1000, help='SSL evaluation epochs')
    parser.add_argument('--label_percentage', type=float, default=1.0, help='percentage of labeled data for linear probing')
    parser.add_argument('--epoch_linear_train', type=int, default=100, help='epochs to train the linear probe')
    parser.add_argument('--lr_linear', type=float, default=0.1, help='learning rate for the linear probe')
    parser.add_argument('--batch_linear', type=int, default=256, help='batch size for the linear probe')

    # -------------------- Output --------------------
    parser.add_argument('--save_path', type=str, default='result', help='path to save results')

    args = parser.parse_args()
    if not 0 < args.percentage <= 100:
        parser.error('--percentage must be in (0, 100].')
    if args.Iteration < 0 or args.num_eval < 0:
        parser.error('--Iteration and --num_eval must be nonnegative.')
    if min(args.batch_real, args.batch_syn, args.covariance_block_size) < 1:
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
    if args.ssl_aug_strategy is None:
        args.ssl_aug_strategy = args.distill_aug_strategy
    if args.distill_aug_views and args.ssl_aug_strategy != args.distill_aug_strategy:
        parser.error('SSL evaluation and distillation must use the same augmentation strategy.')
    args.method = 'Scattering'
    args.dsa_param = ParamDiffAug()
    args.dsa_param.aug_mode = args.distill_aug_mode
    args.ssl_aug_mode = args.distill_aug_mode
    args.dsa = False
    configure_determinism(args.seed)

    os.makedirs(args.save_path, exist_ok=True)
    clear_directory(args.save_path)
    print('Cleared all previous files from %s' % args.save_path, flush=True)
    os.makedirs(args.data_path, exist_ok=True)

    channel, im_size, num_classes, _, mean, std, dst_train, _, testloader = get_dataset(args.dataset, args.data_path)
    num_syn = int(len(dst_train) * args.percentage / 100)
    if len(dst_train) < 2 or num_syn < 2:
        parser.error('Unbiased covariance requires at least two real and two synthetic images; increase --percentage.')
    scattering = ScatteringFeatures(im_size, J=args.scattering_J, L=args.scattering_L).to(args.device).eval()
    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model) if args.num_eval else []
    accs_all_exps = {model: [] for model in model_eval_pool}
    baseline_accs = {}
    print('Hyper-parameters: \n', args.__dict__)

    # Keep the real dataset on the host; only scattering batches go to the device.
    indices = np.random.RandomState(args.seed).permutation(len(dst_train))[:num_syn]
    image_syn = torch.stack([dst_train[int(i)][0] for i in indices]).to(args.device).detach().requires_grad_(True)
    diag_mean = torch.tensor(mean, device=args.device).view(1, channel, 1, 1)
    diag_std = torch.tensor(std, device=args.device).view(1, channel, 1, 1)

    def to_uint8(images):
        return ((images * diag_std + diag_mean) * 255 + 0.5).clamp(0, 255).to(torch.uint8)

    image_syn_init_uint8 = to_uint8(image_syn.detach())

    def evaluate(iteration, record=False):
        for model_index, model_eval in enumerate(model_eval_pool):
            print('-------------------------\nEvaluation\nmodel_eval = %s, iteration = %d' % (model_eval, iteration))
            accs = []
            for it_eval in range(args.num_eval):
                eval_index = model_index * args.num_eval + it_eval
                network_seed = args.seed + eval_index
                training_seed = args.seed + 1_000_000 + eval_index
                net_eval = get_network(
                    model_eval, channel, num_classes, im_size, seed=network_seed
                ).to(args.device)
                set_random_seed(training_seed)
                _, _, acc_test = evaluate_synset_SSL(
                    it_eval, net_eval, image_syn.detach().clone(), dst_train, testloader, args
                )
                accs.append(float(acc_test))
            print('Evaluate %d random %s, mean = %.4f std = %.4f\n-------------------------' % (
                len(accs), model_eval, np.mean(accs), np.std(accs)
            ))
            if iteration == 0:
                baseline_accs[model_eval] = list(accs)
            else:
                differences = 100 * (np.asarray(accs) - np.asarray(baseline_accs[model_eval]))
                print('Gain over initial subset with matched evaluation seeds: mean = %+.3f pp, std = %.3f pp, per seed = %s' % (
                    np.mean(differences), np.std(differences),
                    ', '.join('%+.3f' % difference for difference in differences),
                ), flush=True)
            if record:
                accs_all_exps[model_eval].extend(accs)

    evaluate(0, record=args.Iteration == 0)

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

    # This generator intentionally has no fixed seed: augmented batches and
    # transforms are freshly sampled independently of initialization/evaluation.
    augmentation_rng = np.random.default_rng()
    pair_count = args.distill_aug_views * (args.distill_aug_views - 1) // 2
    print('Loss = clean + sum of %d view moment losses + %.4g * sum of %d cross-view covariance losses (real batch size %d)' % (
        args.distill_aug_views, args.pair_weight, pair_count, min(args.batch_real, len(dst_train))
    ), flush=True)
    print('%s distillation begins' % get_time(), flush=True)
    optimizer_img = torch.optim.Adam([image_syn], lr=args.lr_img)


    for it in range(1, args.Iteration + 1):
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
                    replay_seed=int(augmentation_rng.integers(0, 2 ** 31)),
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
            changed = (to_uint8(image_syn) != image_syn_init_uint8).float().mean().item() * 100
        if it%5 == 0 or it ==1:
            print('%s iter = %05d, loss = %.10f, covariance = %.10f, mean = %.10f, pair = %.10f, weighted pair = %.10f, clean = %.10f, augmented views = %s, PNG values changed = %.5f%%' % (
                get_time(), it, loss.item(), covariance_loss.item(), mean_loss.item(),
                pair_loss.item(), args.pair_weight * pair_loss.item(),
                clean_loss, ', '.join('%.10f' % value for value in view_losses), changed
            ), flush=True)

        if it % 400 == 0 and it != args.Iteration:
            evaluate(it)
        if it % 50 == 0 or it == args.Iteration:
            save_name = os.path.join(args.save_path, 'vis_%s_%s_%s_%dpercent_iter%d.png' % (
                args.method, args.dataset, args.model, args.percentage, it
            ))
            image_syn_vis = (image_syn.detach() * diag_std + diag_mean).clamp(0, 1).cpu()
            save_image(image_syn_vis, save_name, nrow=int(np.ceil(np.sqrt(num_syn))))

    # Save before final evaluation so distilled images survive an evaluation failure.
    save_name = os.path.join(args.save_path, 'res_Scattering-SSL_%s_%s_%dpercent.pt' % (
        args.dataset, args.model, args.percentage
    ))
    torch.save({
        'data': image_syn.detach().cpu(),
        'method': args.method,
        'iteration': args.Iteration,
        'scattering': {'J': args.scattering_J, 'L': args.scattering_L, 'max_order': 2},
        'gamma': args.gamma,
        'pair_weight': args.pair_weight,
        'seed': args.seed,
        'baseline_accs': baseline_accs,
        'evaluation_augmentation': {'strategy': args.ssl_aug_strategy, 'mode': args.ssl_aug_mode},
        'distill_augmentation': {
            'views': args.distill_aug_views,
            'strategy': args.distill_aug_strategy,
            'mode': args.distill_aug_mode,
            'batch_real': args.batch_real,
            'fixed_seed': False,
            'cross_view_pairs': 'all_unordered_pairs_with_corresponding_image_rows',
        },
    }, save_name)
    print('Saved synthetic data to %s' % save_name)

    # Release the dense target before training the final evaluation networks.
    del real_mean, real_covariance, scattering
    if args.Iteration > 0:
        evaluate(args.Iteration, record=True)
    for model_eval, accs in accs_all_exps.items():
        print('Scattering distillation, evaluate %d random %s, mean = %.2f%% std = %.2f%%' % (
            len(accs), model_eval, np.mean(accs) * 100, np.std(accs) * 100
        ))


if __name__ == '__main__':
    main()
