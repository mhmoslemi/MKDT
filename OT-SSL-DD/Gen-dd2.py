import argparse
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image

from eval import ROOT, reference_modules
from utils import DiffAugment, ParamDiffAug, get_dataset, get_time


KERNEL_SCALES = (0.25, 0.5, 1.0, 2.0, 4.0)


class ResNetSpatialFeatures(nn.Module):
    """Return a ResNet feature map before global average pooling."""

    def __init__(self, encoder, layer):
        super().__init__()
        required = ('conv1', 'bn1', 'layer1', 'layer2', 'layer3', 'layer4')
        if any(not hasattr(encoder, name) for name in required):
            raise ValueError('SADM currently requires the repository ResNet18 encoder')
        self.encoder = encoder
        self.layer = layer

    def forward(self, images):
        features = F.relu(self.encoder.bn1(self.encoder.conv1(images)))
        for name in ('layer1', 'layer2', 'layer3', 'layer4'):
            features = getattr(self.encoder, name)(features)
            if name == self.layer:
                return features
        raise ValueError(self.layer)


def sorted_descriptors(features, topk):
    if features.ndim != 4:
        raise ValueError('Spatial features must have shape [batch, channels, height, width]')
    sorted_features = features.flatten(2).sort(dim=2, descending=True).values
    if not 0 < topk <= sorted_features.shape[2]:
        raise ValueError('topk must not exceed the number of spatial features')
    # Rank-major layout matches the official SADM implementation: rank 1 for
    # every channel, then rank 2 for every channel, and so on.
    rank_major = sorted_features.transpose(1, 2).reshape(features.shape[0], -1)
    top = rank_major[:, :features.shape[1] * topk]
    return rank_major, top


def rbf_mmd(real, synthetic):
    if real.ndim != 2 or synthetic.ndim != 2 or real.shape[1] != synthetic.shape[1]:
        raise ValueError('MMD inputs must be two matrices with the same feature dimension')
    samples = torch.cat((real, synthetic))
    if len(samples) < 2:
        raise ValueError('MMD requires at least two samples')
    squared_distances = torch.cdist(samples, samples).square()
    count = len(samples)
    bandwidth = (squared_distances.detach().sum() / (count * count - count)).clamp_min(
        torch.finfo(samples.dtype).eps)
    scales = samples.new_tensor(KERNEL_SCALES)
    kernel = torch.exp(
        -squared_distances.unsqueeze(0) / (bandwidth * scales[:, None, None])).sum(dim=0)
    real_count = len(real)
    return (kernel[:real_count, :real_count].mean()
            + kernel[real_count:, real_count:].mean()
            - 2 * kernel[:real_count, real_count:].mean())


def sadm_loss(real_features, synthetic_features, topk, lambda_all, lambda_top):
    real_all, real_top = sorted_descriptors(real_features, topk)
    synthetic_all, synthetic_top = sorted_descriptors(synthetic_features, topk)
    loss_all = rbf_mmd(real_all.detach(), synthetic_all)
    loss_top = rbf_mmd(real_top.detach(), synthetic_top)
    return lambda_all * loss_all + lambda_top * loss_top, loss_all, loss_top


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


def resume_distillation(path, images, momentum, optimizer, expected, common):
    if not os.path.exists(path):
        return 0, []
    saved = torch.load(path, map_location='cpu', weights_only=True)
    for key, value in expected.items():
        if saved.get(key) != value:
            raise ValueError('Cannot resume: checkpoint %s differs from this run' % key)
    saved_pixels = saved.get('pixels')
    saved_momentum = saved.get('pixel_momentum')
    optimizer_state = saved.get('optimizer_state')
    iteration = saved.get('iteration')
    history = saved.get('loss_history')
    expected_shape = (len(images), *images[0].shape)
    valid_state = ((isinstance(saved_momentum, torch.Tensor)
                    and saved_momentum.shape == expected_shape
                    and torch.isfinite(saved_momentum).all())
                   if optimizer is None else isinstance(optimizer_state, dict))
    if (not isinstance(saved_pixels, torch.Tensor)
            or saved_pixels.shape != expected_shape
            or not torch.isfinite(saved_pixels).all()
            or not valid_state
            or not isinstance(iteration, int) or iteration < 0
            or not isinstance(history, torch.Tensor) or history.shape != (iteration, 3)):
        raise ValueError('Cannot resume: invalid pixels, optimizer state, iteration or loss history')
    with torch.no_grad():
        for image, saved_image in zip(images, saved_pixels):
            image.copy_(saved_image.to(image.device))
    if optimizer is None:
        momentum.copy_(saved_momentum)
    else:
        optimizer.load_state_dict(optimizer_state)
    common.restore_rng(saved['rng_state'])
    print('Resumed SADM at iteration %d: %s' % (iteration, path))
    return iteration, history.tolist()


# Hyper-parameters:  42%
#  'feature_layer': 'layer3', 'topk': 3, 'lambda_all': 0.008, 'lambda_top': 0.005, 'Iteration': 8000, 'lr_img': 1,
#   'image_optimizer': 'sgd', 'momentum': 0.5, 'batch_real': 256, 'batch_syn': 256, 'aug_strategy': 'color_crop_cutout_flip_scale_rotate',
   

# Hyper-parameters:  41.7
# 'feature_layer': 'layer1', 'topk': 3, 'lambda_all': 0.008, 'lambda_top': 0.005, 'Iteration': 8000, 'lr_img': 1,
#  'image_optimizer': 'sgd', 'momentum': 0.5, 'batch_real': 256, 'batch_syn': 256, 'aug_strategy': 'color_crop_cutout_flip_scale_rotate', 


def main():
    parser = argparse.ArgumentParser(description='2D Semantically Aligned Distribution Matching')
    parser.add_argument('--dataset', choices=['CIFAR10', 'CIFAR100', 'TinyImageNet'], default='CIFAR10')
    parser.add_argument('--model', choices=['ResNet18'], default='ResNet18BN')
    parser.add_argument('--percentage', type=float, default=0.2)
    parser.add_argument('--teacher_source', choices=['pretrained', 'random'], default='random')
    parser.add_argument('--teacher_ssl', choices=['simclr', 'barlowtwins'], default='simclr')
    parser.add_argument('--teacher_seeds', type=int, nargs='+', default=[0])
    parser.add_argument('--teacher_path', default=str(ROOT / 'Evaluate_benchamrk/full_ssl_checkpoints_1000ep_3seeds'))
    parser.add_argument('--feature_layer', choices=['layer1', 'layer2', 'layer3', 'layer4'], default='layer3')
    parser.add_argument('--topk', type=int, default=3)
    parser.add_argument('--lambda_all', type=float, default=1)
    parser.add_argument('--lambda_top', type=float, default=2)
    parser.add_argument('--Iteration', type=int, default=5000)
    parser.add_argument('--lr_img', type=float, default=1)
    parser.add_argument('--image_optimizer', choices=['manual', 'sgd', 'adam'], default='sgd')
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--batch_real', type=int, default=512)
    parser.add_argument('--batch_syn', type=int, default=1024)
    parser.add_argument('--aug_strategy', default='color_crop_cutout_flip_scale_rotate')
    parser.add_argument('--save_every', type=int, default=500)
    parser.add_argument('--num_exp', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--data_path', default=str(ROOT.parent / 'data'))
    parser.add_argument('--save_path', default=str(ROOT / 'OT-SSL-DD/result_SADM2'))
    args = parser.parse_args()
    benchmark_model = args.model.removesuffix('BN')

    if (not 0 < args.percentage <= 100 or args.Iteration < 1 or args.lr_img <= 0
            or min(args.batch_real, args.batch_syn, args.topk, args.save_every, args.num_exp) < 1
            or not 0 <= args.momentum < 1 or min(args.lambda_all, args.lambda_top) < 0
            or args.lambda_all + args.lambda_top == 0):
        parser.error('Invalid percentage, iteration, batch, optimization, loss or save setting')
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA is unavailable; run this program inside a GPU allocation')

    common = reference_modules()['common']
    common.seed_all(args.seed)
    os.makedirs(args.save_path, exist_ok=True)
    print('Hyper-parameters: \n', args.__dict__)

    channel, im_size, _, _, mean, std, train_set, _, _ = get_dataset(args.dataset, args.data_path)
    if channel != 3:
        raise ValueError('SADM evaluation expects three-channel images')
    num_syn = int(len(train_set) * args.percentage / 100)
    if num_syn < 2:
        raise ValueError('The requested percentage produces fewer than two synthetic images')
    mean_cpu = torch.tensor(mean).view(1, channel, 1, 1)
    std_cpu = torch.tensor(std).view(1, channel, 1, 1)

    augmentation = ParamDiffAug()
    augmentation.aug_mode = 'M'
    expected = {
        'dataset': args.dataset,
        'model': args.model,
        'percentage': args.percentage,
        'method': 'SADM2D',
        'teacher_source': args.teacher_source,
        'feature_layer': args.feature_layer,
        'topk': args.topk,
        'lambda_all': args.lambda_all,
        'lambda_top': args.lambda_top,
        'augmentation': args.aug_strategy,
        'lr_img': args.lr_img,
        'batch_real': args.batch_real,
        'batch_syn': args.batch_syn,
    }
    if args.image_optimizer == 'manual':
        expected['momentum'] = args.momentum
    else:
        expected.update(image_optimizer=args.image_optimizer,
                        **({'momentum': args.momentum} if args.image_optimizer == 'sgd' else {}))

    for exp in range(args.num_exp):
        print('\n================== Exp %d ==================\n' % exp)
        exp_seed = args.seed + exp
        common.seed_all(exp_seed)
        teachers = []
        normalizations = []
        teacher_checkpoints = []
        if args.teacher_source == 'pretrained':
            for teacher_seed in args.teacher_seeds:
                checkpoint = os.path.join(
                    args.teacher_path, args.dataset, benchmark_model, 'full', 'pt100', args.teacher_ssl,
                    'seed_%03d' % teacher_seed, 'encoder.pt')
                encoder, _, saved = common.load_encoder(checkpoint, device=args.device)
                teacher_mean, teacher_std = (
                    torch.tensor(value, device=args.device).view(1, channel, 1, 1)
                    for value in saved['normalization'])
                teachers.append(ResNetSpatialFeatures(
                    encoder, args.feature_layer).eval().requires_grad_(False))
                normalizations.append((teacher_mean, teacher_std))
                teacher_checkpoints.append(checkpoint)
                print('Loaded teacher seed %d: %s' % (teacher_seed, checkpoint))
            teacher_identity = {
                'teacher_ssl': args.teacher_ssl,
                'teacher_checkpoints': teacher_checkpoints,
            }
        else:
            random_normalization = (
                torch.tensor(mean, device=args.device).view(1, channel, 1, 1),
                torch.tensor(std, device=args.device).view(1, channel, 1, 1))
            teacher_identity = {
                'teacher_initialization': 'random_per_iteration',
                'teacher_seed_base': exp_seed,
            }
            print('Using a fresh random teacher every iteration')

        assert args.teacher_source == 'random' or len(teachers) == len(normalizations) > 0
        common.seed_all(exp_seed)
        initial_indices = torch.randperm(len(train_set))[:num_syn]
        initial_pixels = dataset_pixels(train_set, initial_indices, mean_cpu, std_cpu).to(args.device).clamp_(0, 1)
        images = nn.ParameterList(nn.Parameter(image) for image in initial_pixels)
        pixel_momentum = torch.zeros_like(initial_pixels) if args.image_optimizer == 'manual' else None
        optimizer_img = None
        if args.image_optimizer == 'sgd':
            optimizer_img = torch.optim.SGD(images, lr=args.lr_img, momentum=args.momentum)
        elif args.image_optimizer == 'adam':
            optimizer_img = torch.optim.Adam(images, lr=args.lr_img)
        assert (optimizer_img is None) == (args.image_optimizer == 'manual')
        result_path = os.path.join(
            args.save_path, 'res_SADM_%s_%s_%gpercent_exp%d.pt'
            % (args.dataset, args.model, args.percentage, exp))
        start_iteration = 0
        loss_history = []
        run_identity = dict(expected, **teacher_identity, seed=exp_seed)
        if args.resume:
            start_iteration, loss_history = resume_distillation(
                result_path, images, pixel_momentum, optimizer_img, run_identity, common)
        if start_iteration >= args.Iteration:
            print('Already complete: %s (%d iterations)' % (result_path, start_iteration))
            continue

        print('%s SADM begins, synthetic images = %d' % (get_time(), num_syn))
        for iteration in range(start_iteration + 1, args.Iteration + 1):
            if args.teacher_source == 'random':
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(exp_seed + iteration)
                    encoder, _ = common.build_encoder(benchmark_model, im_size[0])
                encoder = encoder.to(args.device).eval().requires_grad_(False)
                teachers = [ResNetSpatialFeatures(encoder, args.feature_layer)]
                normalizations = [random_normalization]
            real_indices = torch.randint(len(train_set), (min(args.batch_real, len(train_set)),))
            synthetic_indices = torch.randperm(num_syn, device=args.device)[:min(args.batch_syn, num_syn)]
            real_pixels = dataset_pixels(train_set, real_indices, mean_cpu, std_cpu).to(args.device)
            selected_images = [images[index] for index in synthetic_indices.tolist()]
            if optimizer_img is None:
                synthetic_batch = torch.stack([image.detach() for image in selected_images]).requires_grad_(True)
            else:
                optimizer_img.zero_grad(set_to_none=True)
                synthetic_batch = torch.stack(selected_images)
            real_view = DiffAugment(real_pixels, args.aug_strategy, param=augmentation)
            synthetic_view = DiffAugment(synthetic_batch, args.aug_strategy, param=augmentation)

            loss = synthetic_batch.new_zeros(())
            loss_all = synthetic_batch.new_zeros(())
            loss_top = synthetic_batch.new_zeros(())
            for teacher, (teacher_mean, teacher_std) in zip(teachers, normalizations):
                with torch.no_grad():
                    real_features = teacher((real_view - teacher_mean) / teacher_std)
                synthetic_features = teacher((synthetic_view - teacher_mean) / teacher_std)
                teacher_loss, teacher_all, teacher_top = sadm_loss(
                    real_features, synthetic_features, args.topk, args.lambda_all, args.lambda_top)
                loss = 100*(loss + teacher_loss / len(teachers))
                loss_all = loss_all + teacher_all.detach() / len(teachers)
                loss_top = loss_top + teacher_top.detach() / len(teachers)

            loss.backward()
            if optimizer_img is None:
                if synthetic_batch.grad is None or not torch.isfinite(synthetic_batch.grad).all():
                    raise FloatingPointError('Non-finite or missing synthetic pixel gradient')
                update = args.momentum * pixel_momentum[synthetic_indices] + synthetic_batch.grad
                pixel_momentum[synthetic_indices] = update
                with torch.no_grad():
                    for image, gradient in zip(selected_images, update):
                        image.copy_((image - args.lr_img * gradient).clamp(0, 1))
            else:
                if any(image.grad is None or not torch.isfinite(image.grad).all() for image in selected_images):
                    raise FloatingPointError('Non-finite or missing synthetic pixel gradient')
                optimizer_img.step()
                with torch.no_grad():
                    for image in selected_images:
                        image.clamp_(0, 1)
            loss_history.append([loss.item(), loss_all.item(), loss_top.item()])

            if iteration % 100 == 0:
                print('%s iter = %05d, loss = %.6f, all = %.6f, top = %.6f'
                      % (get_time(), iteration, loss.item(), loss_all.item(), loss_top.item()))

            if iteration % args.save_every == 0 or iteration == args.Iteration:
                pixels_cpu = torch.stack([image.detach().cpu() for image in images])
                normalized = (pixels_cpu - mean_cpu) / std_cpu
                preview_path = os.path.join(
                    args.save_path, 'vis_SADM_%s_%s_%gpercent_exp%d_iter%d.png'
                    % (args.dataset, args.model, args.percentage, exp, iteration))
                save_image(pixels_cpu[:min(256, num_syn)], preview_path,
                           nrow=int(math.ceil(math.sqrt(min(256, num_syn)))))
                result = {
                    **run_identity,
                    'data': normalized,
                    'pixels': pixels_cpu,
                    'normalization': [mean, std],
                    'iteration': iteration,
                    'loss_history': torch.tensor(loss_history),
                    'rng_state': common.rng_state(),
                    'dis_metric': 'rank_sorted_spatial_dual_rbf_mmd',
                    'kernel': 'official_sadm_five_scale_rbf',
                    'kernel_scales': list(KERNEL_SCALES),
                    'initialization': 'random_real_images',
                    'input_size': list(im_size),
                }
                if optimizer_img is None:
                    result['pixel_momentum'] = pixel_momentum.detach().cpu()
                else:
                    result['optimizer_state'] = optimizer_img.state_dict()
                common.atomic_save(result_path, result)
                print('Saved synthetic data to %s' % result_path)


if __name__ == '__main__':
    main()
