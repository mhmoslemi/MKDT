import argparse
import json
import math
import os
import statistics
from datetime import datetime
from pathlib import Path

import kornia.augmentation as K
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
from dm_dsa import DMDSA

from eval import (PROBE_EPOCHS, ROOT, SSL_EPOCHS, evaluate_synset_SSL,
                  reference_modules)
from utils import get_dataset, get_time


def ssl_augmentation(protocol, dataset, method):
    size = protocol['input_size'][dataset]
    common, recipe = protocol['ssl'], protocol['methods'][method]
    blur_kernel = 2 * math.ceil(3 * common['blur_sigma'][1]) + 1
    views = []
    for view in range(2):
        views.append(nn.Sequential(
            K.RandomResizedCrop(
                (size, size), scale=tuple(common['crop_scale']),
                ratio=tuple(common['crop_ratio']), resample='BICUBIC',
                align_corners=False, cropping_mode='resample'),
            nn.Hardtanh(0, 1),
            K.ColorJitter(*recipe['color_jitter'], p=common['color_probability']),
            K.RandomGrayscale(p=common['grayscale_probability']),
            K.RandomGaussianBlur(blur_kernel, tuple(common['blur_sigma']),
                                 p=recipe['blur_probabilities'][view]),
            K.RandomSolarize(thresholds=(128 / 255, 128 / 255), additions=0,
                             p=recipe['solarize_probabilities'][view]),
            K.RandomHorizontalFlip(p=common['flip_probability'])))
    return nn.ModuleList(views)


def similarity_distributions(first_images, second_images, encoder, normalization, negatives=True):
    mean, std = normalization
    first = F.normalize(encoder((first_images - mean) / std), dim=1)
    second = F.normalize(encoder((second_images - mean) / std), dim=1)
    positive = (first * second).sum(1)
    negative = positive.new_empty(0)
    if negatives:
        similarities = first @ second.t()
        different_images = ~torch.eye(len(first), device=first.device, dtype=torch.bool)
        negative = similarities[different_images]
    return positive, negative


def quantile_target(real_sorted, count):
    """Integrate the empirical real quantile over each synthetic probability bin."""
    real = real_sorted.double()
    positions = torch.arange(count + 1, device=real.device) * len(real)
    indices = positions.div(count, rounding_mode='floor')
    fractions = positions.remainder(count).double() / count
    prefix = torch.cat((real.new_zeros(1), real.cumsum(0)))
    integral = prefix[indices] + fractions * real[indices.clamp_max(len(real) - 1)]
    means = integral.diff() * (count / len(real))
    variance = (real.square().mean() - means.square().mean()).clamp_min(0)
    if not torch.isfinite(means).all() or not torch.isfinite(variance):
        raise FloatingPointError('Invalid real quantile target')
    return means.float(), variance.float()


def wasserstein_loss(synthetic, target):
    # The constant retains exact empirical W2 squared when sample counts differ.
    means, variance = target
    assert synthetic.ndim == 1 and synthetic.shape == means.shape
    return (synthetic.sort().values - means).square().mean() + variance


def dataset_pixels(dataset, indices, mean, std):
    indices = indices.cpu()
    if hasattr(dataset, 'data'):
        pixels = torch.as_tensor(dataset.data[indices.numpy()])
        if pixels.ndim != 4:
            raise ValueError('Unsupported raw dataset shape')
        if pixels.shape[-1] in (1, 3):
            pixels = pixels.permute(0, 3, 1, 2)
        return pixels.float().div_(255)
    if hasattr(dataset, 'images'):
        normalized = dataset.images[indices]
    else:
        normalized = torch.stack([dataset[int(index)][0] for index in indices])
    return normalized.float() * std + mean


@torch.no_grad()
def prepare_random_teacher(encoder, dataset, views, normalization, mean, std, args):
    batch_norms = [module for module in encoder.modules() if isinstance(module, nn.BatchNorm2d)]
    momentums = [module.momentum for module in batch_norms]
    for module in batch_norms:
        # Zero residual BN scales are useful for SSL training, not a frozen random teacher.
        nn.init.ones_(module.weight)
        module.reset_running_stats()
        module.momentum = None
        module.train()
    teacher_mean, teacher_std = normalization
    batch_size = min(args.batch_real, len(dataset))
    for start in range(0, args.real_pairs, batch_size):
        count = min(batch_size, args.real_pairs - start)
        indices = torch.randperm(len(dataset))[:count]
        pixels = dataset_pixels(dataset, indices, mean, std).to(args.device)
        for view in views:
            encoder((view(pixels) - teacher_mean) / teacher_std)
    for module, momentum in zip(batch_norms, momentums):
        module.momentum = momentum
    encoder.eval()


def real_distributions(args, dataset, teachers, normalizations, teacher_identity,
                       views, mean, std, negatives, common):
    identity = {
        'dataset': args.dataset, 'data_path': os.path.abspath(args.data_path),
        'dataset_count': len(dataset), 'teachers': teacher_identity,
        'real_pairs': args.real_pairs,
        'negative_pairs': args.negative_pairs if negatives else 0,
        'batch_real': args.batch_real, 'seed': args.seed,
        'augmentation_method': args.ssl_method,
        'protocol': json.loads((ROOT / 'Evaluate_benchamrk/protocol.json').read_text()),
        'implementation_sha256': common.file_hash(__file__)}
    target_path = Path(args.save_path) / ('real_relations_%s.pt' % common.digest_json(identity))
    if target_path.exists():
        saved = torch.load(target_path, map_location='cpu', weights_only=True)
        if saved['identity'] != identity:
            raise ValueError('Cached real similarity distributions do not match this run')
        print('Loaded real similarity distributions: %s' % target_path, flush=True)
        return saved['distributions'], str(target_path)

    distributions = []
    batch_size = min(args.batch_real, len(dataset))
    for teacher_index, (teacher, normalization) in enumerate(zip(teachers, normalizations)):
        common.seed_all(args.seed + 10000 + teacher_index)
        negative_rng = torch.Generator(device=args.device).manual_seed(args.seed + 20000 + teacher_index)
        positive_samples, negative_samples = [], []
        with torch.no_grad():
            for start in range(0, args.real_pairs, batch_size):
                count = min(batch_size, args.real_pairs - start)
                indices = torch.randperm(len(dataset))[:count]
                pixels = dataset_pixels(dataset, indices, mean, std).to(args.device)
                positive, negative = similarity_distributions(
                    views[0](pixels), views[1](pixels), teacher, normalization,
                    negatives and count > 1)
                positive_samples.append(positive.cpu())
                if len(negative):
                    keep = min(len(negative), math.ceil(args.negative_pairs * count / args.real_pairs))
                    order = torch.randperm(len(negative), device=args.device, generator=negative_rng)[:keep]
                    negative_samples.append(negative[order].cpu())
        positive = torch.cat(positive_samples).sort().values
        negative = torch.cat(negative_samples) if negative_samples else torch.empty(0)
        if len(negative) > args.negative_pairs:
            negative = negative[torch.randperm(len(negative))[:args.negative_pairs]]
        if not torch.isfinite(positive).all() or not torch.isfinite(negative).all():
            raise FloatingPointError('Non-finite real similarity distribution')
        if negatives and len(negative) == 0:
            raise ValueError('Real target needs pairs of different images')
        distributions.append({'positive': positive, 'negative': negative.sort().values})
        print('Teacher %d real targets: %d positives, %d negatives'
              % (teacher_index, len(positive), len(negative)), flush=True)
    common.atomic_save(target_path, {'identity': identity, 'distributions': distributions})
    print('Saved real similarity distributions: %s' % target_path, flush=True)
    return distributions, str(target_path)


def resume_distillation(path, images, optimizer, expected, common):
    if not os.path.exists(path):
        return 0, []
    saved = torch.load(path, map_location='cpu', weights_only=True)
    for key, value in expected.items():
        if saved.get(key) != value:
            raise ValueError('Cannot resume: checkpoint %s differs from this run' % key)
    pixels, iteration = saved.get('pixels'), saved.get('iteration')
    history, optimizer_state = saved.get('loss_history'), saved.get('optimizer_state')
    if (not isinstance(pixels, torch.Tensor) or pixels.shape != (len(images), *images[0].shape)
            or not torch.isfinite(pixels).all() or not isinstance(optimizer_state, dict)
            or not isinstance(iteration, int) or iteration < 0
            or not isinstance(history, torch.Tensor) or history.shape != (iteration, 3)):
        raise ValueError('Cannot resume: invalid pixels, optimizer, iteration or loss history')
    with torch.no_grad():
        for image, saved_image in zip(images, pixels):
            image.copy_(saved_image.to(image.device))
    optimizer.load_state_dict(optimizer_state)
    common.restore_rng(saved['rng_state'])
    print('Resumed relational matching at iteration %d: %s' % (iteration, path))
    return iteration, history.tolist()


def distill(args, experiment, exp, dataset, im_size, mean, std, teachers, normalizations,
            targets, target_path, teacher_identity, views, common):
    exp_seed = args.seed + exp
    common.seed_all(exp_seed)
    num_syn = int(len(dataset) * args.percentage / 100)
    if args.init == 'real':
        initial_indices = torch.randperm(len(dataset))[:num_syn]
        initial_pixels = dataset_pixels(dataset, initial_indices, mean, std)
    else:
        initial_pixels = torch.rand(num_syn, 3, *im_size)
    initial_pixels.clamp_(0, 1)
    initial_pixels = initial_pixels.to(args.device)
    images = nn.ParameterList(nn.Parameter(image.clone()) for image in initial_pixels)
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(images, lr=args.lr_img, momentum=args.momentum)
    else:
        optimizer = torch.optim.Adam(images, lr=args.lr_img)
    negative_weight = args.lambda_neg if experiment == 'positive-negative' else 0
    identity = {
        'method': 'RelationalWassersteinDD', 'experiment': experiment,
        'dataset': args.dataset, 'model': args.model, 'percentage': args.percentage,
        'seed': exp_seed, 'initialization': args.init,
        'teacher_identity': teacher_identity if experiment != 'random' else [],
        'target_path': target_path if experiment != 'random' else None,
        'lambda_neg': negative_weight, 'optimizer': args.optimizer,
        'lr_img': args.lr_img, 'momentum': args.momentum,
        'batch_syn': args.batch_syn, 'feature_layer': 'encoder_output_before_projector',
        'dis_metric': 'exact_empirical_1d_wasserstein_squared',
        'augmentation_method': args.ssl_method,
        'augmentation_backend': 'kornia_tensor_benchmark_recipe',
        'dsa': args.dsa,
        'dsa_settings': views[0].configuration() if args.dsa else None,
        'teacher_reduction': 'mean_all_teachers'}
    if args.dsa:
        identity['augmentation_backend'] = 'main_DM_realreal_DSA'
    result_path = Path(args.save_path) / ('res_WAS_%s_%s_%gpercent_%s_exp%d.pt'
                                        % (args.dataset, args.model, args.percentage, experiment, exp))
    start, history = 0, []
    if args.resume:
        start, history = resume_distillation(result_path, images, optimizer, identity, common)
    final_iteration = 0 if experiment == 'random' else args.Iteration

    def save(iteration):
        pixels = torch.stack([image.detach().cpu() for image in images])
        preview_path = result_path.with_name(result_path.stem.replace('res_', 'vis_', 1)
                                             + '_iter%d.png' % iteration)
        save_image(pixels[:256], str(preview_path),
                   nrow=math.ceil(math.sqrt(min(256, num_syn))))
        common.atomic_save(result_path, {
            **identity, 'data': (pixels - mean) / std, 'pixels': pixels,
            'normalization': [mean.flatten().tolist(), std.flatten().tolist()],
            'input_size': list(pixels.shape[2:]), 'iteration': iteration,
            'loss_history': torch.tensor(history).reshape(-1, 3),
            'optimizer_state': optimizer.state_dict(), 'rng_state': common.rng_state()})
        print('Saved synthetic data to %s' % result_path, flush=True)

    print('%s %s begins: %d images, %d iterations'
          % (get_time(), experiment, num_syn, final_iteration), flush=True)
    for iteration in range(start + 1, final_iteration + 1):
        indices = torch.randperm(num_syn, device=args.device)[:min(args.batch_syn, num_syn)]
        selected = [images[index] for index in indices.tolist()]
        synthetic = torch.stack(selected)
        optimizer.zero_grad(set_to_none=True)
        if args.dsa:
            seeds = DMDSA.seeds(2)
            first_images, second_images = views[0](synthetic, seeds[0]), views[1](synthetic, seeds[1])
            indices_real = torch.randperm(len(dataset))[:min(args.batch_real, len(dataset))]
            real_pixels = dataset_pixels(dataset, indices_real, mean, std).to(args.device)
            with torch.no_grad():
                real_first, real_second = views[0](real_pixels, seeds[0]), views[1](real_pixels, seeds[1])
        else:
            first_images, second_images = views[0](synthetic), views[1](synthetic)
        positive_loss, negative_loss = synthetic.new_zeros(()), synthetic.new_zeros(())
        for teacher, normalization, target in zip(teachers, normalizations, targets):
            if args.dsa:
                with torch.no_grad():
                    real_positive, real_negative = similarity_distributions(
                        real_first, real_second, teacher, normalization, negative_weight > 0)
                    target = {'positive': quantile_target(real_positive.sort().values, len(synthetic))}
                    if negative_weight > 0:
                        target['negative'] = quantile_target(real_negative.sort().values, len(synthetic) * (len(synthetic) - 1))
            positive, negative = similarity_distributions(
                first_images, second_images, teacher, normalization, negative_weight > 0)
            positive_loss = positive_loss + wasserstein_loss(positive, target['positive']) / len(teachers)
            if negative_weight > 0:
                negative_loss = negative_loss + wasserstein_loss(negative, target['negative']) / len(teachers)
        loss = positive_loss + negative_weight * negative_loss
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite Wasserstein loss')
        loss.backward()
        if (any(image.grad is None for image in selected)
                or not torch.isfinite(torch.stack([image.grad for image in selected])).all()):
            raise FloatingPointError('Non-finite or missing synthetic pixel gradient')
        optimizer.step()
        with torch.no_grad():
            for image in selected:
                image.clamp_(0, 1)
        history.append([loss.item(), positive_loss.item(), negative_loss.item()])
        if iteration % 50 == 0 or iteration == 1:
            with torch.no_grad():
                pixel_change = (torch.stack(list(images)) - initial_pixels).square().mean().sqrt().mul(255).item()
            print('%s iter = %05d, loss = %.6f, positive = %.6f, negative = %.6f, RMS = %.2f/255'
                  % (get_time(), iteration, loss.item(), positive_loss.item(), negative_loss.item(), pixel_change),
                  flush=True)
        if iteration % args.save_every == 0 or iteration == final_iteration:
            save(iteration)
    if final_iteration == 0 or start >= final_iteration:
        save(start)
    return result_path, exp_seed


def main():
    parser = argparse.ArgumentParser(description='SSL relational distribution matching with 1D Wasserstein loss')
    parser.add_argument('--dataset', choices=['CIFAR10', 'CIFAR100', 'TinyImageNet'], default='CIFAR10')
    parser.add_argument('--model', choices=['ConvNetBN', 'ResNet18BN', 'VGG11BN',
                                          'ConvNet', 'ResNet18', 'VGG11'], default='ConvNetBN')
    parser.add_argument('--percentage', type=float, default=1)
    parser.add_argument('--experiment', choices=['random', 'positive', 'positive-negative', 'all'],
                        default='positive-negative')
    parser.add_argument('--init', choices=['noise', 'real'], default='real')
    parser.add_argument('--lambda_neg', '--lambda-neg', type=float, default=1)
    parser.add_argument('--teacher_source', choices=['pretrained', 'random'], default='random')
    parser.add_argument('--teacher_ssl', choices=['simclr', 'barlowtwins'], default='simclr')
    parser.add_argument('--teacher_seeds', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--teacher_path', default=str(
        ROOT / 'Evaluate_benchamrk/full_ssl_checkpoints_1000ep_3seeds'))
    parser.add_argument('--real_pairs', type=int, default=8192)
    parser.add_argument('--negative_pairs', type=int, default=65536)
    parser.add_argument('--Iteration', type=int, default=8000)
    parser.add_argument('--lr_img', type=float, default=0.002)
    parser.add_argument('--optimizer', choices=['adam', 'sgd'], default='adam')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum with --optimizer sgd')
    parser.add_argument('--batch_real', type=int, default=512)
    parser.add_argument('--batch_syn', type=int, default=512)
    parser.add_argument('--dsa', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--dsa_strategy', '--dsa-strategy', default='color_crop_cutout_flip_scale_rotate')
    parser.add_argument('--save_every', type=int, default=500)
    parser.add_argument('--num_exp', type=int, default=1)
    parser.add_argument('--label-percentage', type=int, choices=list(PROBE_EPOCHS), default=1)
    parser.add_argument('--ssl-method', choices=['simclr', 'barlowtwins'], default='simclr')
    parser.add_argument('--ssl-epochs', dest='epoch_eval_train', type=int, default=None)
    parser.add_argument('--ssl-log-every', type=int, default=100)
    parser.add_argument('--probe-epochs', dest='epoch_linear_train', type=int, default=None)
    parser.add_argument('--lr-linear', type=float, default=0.1)
    parser.add_argument('--batch-linear', type=int, default=256)
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--data_path', default=str(ROOT.parent / 'data'))
    parser.add_argument('--save_path', default=str(ROOT / 'OT-SSL-DD/result_WAS'))
    args = parser.parse_args()
    if args.epoch_eval_train is None:
        args.epoch_eval_train = SSL_EPOCHS[args.dataset].get(args.percentage)
    if args.epoch_linear_train is None:
        args.epoch_linear_train = PROBE_EPOCHS[args.label_percentage]
    if args.epoch_eval_train is None:
        parser.error('No SSL epoch schedule for this size; set --ssl-epochs explicitly')
    if (not 0 < args.percentage <= 100 or args.Iteration < 1 or args.lr_img <= 0
            or min(args.real_pairs, args.batch_real, args.batch_syn) < 2
            or min(args.negative_pairs, args.save_every, args.num_exp) < 1
            or not 0 <= args.momentum < 1 or args.lambda_neg < 0
            or min(args.epoch_eval_train, args.epoch_linear_train, args.runs,
                   args.batch_linear, args.ssl_log_every) < 1 or args.lr_linear <= 0):
        parser.error('Invalid percentage, iteration, batch, optimization, loss or save setting')
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA is unavailable; run this program inside a GPU allocation')
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', '1')))
    reference = reference_modules()
    common = reference['common']
    common.seed_all(args.seed)
    os.makedirs(args.save_path, exist_ok=True)
    print('Hyper-parameters: \n', args.__dict__)

    channel, im_size, _, _, mean, std, dataset, _, _ = get_dataset(args.dataset, args.data_path)
    num_syn = int(len(dataset) * args.percentage / 100)
    if channel != 3 or num_syn < 2:
        raise ValueError('Relational matching needs at least two three-channel images')
    mean = torch.tensor(mean).view(1, channel, 1, 1)
    std = torch.tensor(std).view(1, channel, 1, 1)
    protocol = json.loads((ROOT / 'Evaluate_benchamrk/protocol.json').read_text())
    views = (nn.ModuleList([DMDSA(args.dsa_strategy, mean, std) for _ in range(2)]) if args.dsa
             else ssl_augmentation(protocol, args.dataset, args.ssl_method)).to(args.device)
    experiments = ['random', 'positive', 'positive-negative'] if args.experiment == 'all' else [args.experiment]
    negatives = 'positive-negative' in experiments and args.lambda_neg > 0
    teachers, normalizations, teacher_identity, targets = [], [], [], []
    target_path = None
    if any(experiment != 'random' for experiment in experiments):
        for seed in args.teacher_seeds:
            if args.teacher_source == 'pretrained':
                checkpoint = Path(args.teacher_path) / args.dataset / args.model.removesuffix('BN') / 'full' / 'pt100' / args.teacher_ssl / ('seed_%03d' % seed) / 'encoder.pt'
                encoder, _, saved = common.load_encoder(checkpoint, device=args.device)
                if saved['input_size'] != im_size[0] or saved['model'] != args.model.removesuffix('BN'):
                    raise ValueError('Teacher architecture or input size differs from this run')
                normalization = saved['normalization']
                identity = {'path': str(checkpoint.resolve()), 'sha256': saved['encoder_sha256']}
                print('Loaded teacher seed %d: %s' % (seed, checkpoint), flush=True)
            else:
                common.seed_all(args.seed + seed)
                encoder, _ = common.build_encoder(args.model.removesuffix('BN'), im_size[0])
                encoder = encoder.to(args.device).eval().requires_grad_(False)
                normalization = protocol['normalization'][args.dataset]
                prepare_random_teacher(
                    encoder, dataset, views,
                    tuple(torch.tensor(value, device=args.device).view(1, channel, 1, 1)
                          for value in normalization), mean, std, args)
                identity = {'source': 'fixed_random', 'seed': args.seed + seed,
                            'batchnorm_statistics': 'calibrated_on_real_augmented_images',
                            'sha256': common.state_hash(encoder)}
            teachers.append(encoder)
            normalizations.append(tuple(torch.tensor(value, device=args.device).view(1, channel, 1, 1)
                                        for value in normalization))
            teacher_identity.append({**identity, 'normalization': normalization})
        if args.dsa:
            distributions = [None] * len(teachers)
        else:
            distributions, target_path = real_distributions(
                args, dataset, teachers, normalizations, teacher_identity,
                views, mean, std, negatives, common)
        batch = min(args.batch_syn, num_syn)
        for distribution in distributions:
            if args.dsa:
                targets.append(None)
                continue
            target = {'positive': tuple(value.to(args.device)
                                       for value in quantile_target(distribution['positive'], batch))}
            if negatives:
                target['negative'] = tuple(value.to(args.device) for value in quantile_target(
                    distribution['negative'], batch * (batch - 1)))
            targets.append(target)

    summaries = []
    for exp in range(args.num_exp):
        for experiment in experiments:
            print('\n================== Exp %d / %s ==================\n' % (exp, experiment))
            result_path, exp_seed = distill(
                args, experiment, exp, dataset, im_size, mean, std, teachers, normalizations,
                targets, target_path, teacher_identity, views, common)
            saved = torch.load(result_path, map_location='cpu', weights_only=True)
            required_iteration = 0 if experiment == 'random' else args.Iteration
            if saved['iteration'] < required_iteration:
                raise ValueError('Final synthetic checkpoint is incomplete: %s' % result_path)
            stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            output = Path(args.save_path) / 'evaluation' / experiment / ('exp_%03d_%s' % (exp, stamp))
            output.mkdir(parents=True, exist_ok=False)
            config = dict(vars(args), experiment=experiment,
                          synthetic_checkpoint=str(result_path.resolve()),
                          save_path=str(output.resolve()), generation_experiment=exp, std_ddof=1)
            common.atomic_json(output / 'config.json', config)
            accuracies = []
            for run in range(args.runs):
                seed = exp_seed + run
                _, _, accuracy = evaluate_synset_SSL(
                    run, args.model, saved['data'], args, output / ('seed_%03d' % seed),
                    saved['normalization'], seed=seed)
                accuracies.append(accuracy * 100)
                summary = {
                    'config': config, 'completed_runs': len(accuracies),
                    'expected_runs': args.runs, 'complete': len(accuracies) == args.runs,
                    'accuracies_percent': accuracies, 'mean_percent': statistics.mean(accuracies),
                    'std_percent': statistics.stdev(accuracies) if len(accuracies) > 1 else None}
                common.atomic_json(output / 'summary.json', summary)
                print('%s seed=%d accuracy=%.2f%%' % (experiment, seed, accuracy * 100), flush=True)
            summaries.append(summary)
            comparison_path = Path(args.save_path) / ('summary_WAS_%s_%s_%gpercent.json'
                                                     % (args.dataset, args.model, args.percentage))
            common.atomic_json(comparison_path, {'results': summaries,
                               'complete': len(summaries) == args.num_exp * len(experiments)})
            print('Saved evaluation: %s' % output, flush=True)
    for summary in summaries:
        std_value = '%.2f' % summary['std_percent'] if summary['std_percent'] is not None else 'N/A'
        print('%s exp=%d: %.2f ± %s%%' % (
            summary['config']['experiment'], summary['config']['generation_experiment'],
            summary['mean_percent'], std_value))


if __name__ == '__main__':
    main()
