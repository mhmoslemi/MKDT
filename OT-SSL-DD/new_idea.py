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

from eval import (PROBE_EPOCHS, ROOT, SSL_EPOCHS, evaluate_synset_SSL,
                  reference_modules)
from utils import get_dataset, get_time


class SpatialFeatures(nn.Module):
    """Return the final spatial feature map before flattening/classification."""

    def __init__(self, encoder):
        super().__init__()
        required = ('conv1', 'bn1', 'layer1', 'layer2', 'layer3', 'layer4')
        self.encoder = encoder
        if all(hasattr(encoder, name) for name in required):
            self.kind = 'resnet'
        elif hasattr(encoder, 'features'):
            self.kind = 'convnet'
        else:
            raise ValueError('This method requires the repository ConvNetBN or ResNet18BN encoder')

    def forward(self, images):
        if self.kind == 'convnet':
            return self.encoder.features(images)
        features = F.relu(self.encoder.bn1(self.encoder.conv1(images)))
        for name in ('layer1', 'layer2', 'layer3', 'layer4'):
            features = getattr(self.encoder, name)(features)
        return features


def ssl_augmentation(size):
    return K.AugmentationSequential(
        K.RandomResizedCrop((size, size), scale=(0.2, 1.0), ratio=(0.75, 4 / 3),
                            align_corners=False, cropping_mode='resample'),
        K.RandomHorizontalFlip(p=0.5),
        K.ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8),
        K.RandomGrayscale(p=0.2),
        same_on_batch=False,
    )


def sorted_orbit_features(images, encoder, augmentation, views, mean, std):
    """Return the channel-sorted augmentation orbit [B, K*H*W, C] and rank 1."""
    batch = len(images)
    augmented = augmentation(images.repeat_interleave(views, dim=0))
    features = encoder((augmented - mean) / std)
    if features.ndim != 4 or features.shape[0] != batch * views:
        raise ValueError('Pre-pooling features must have shape [B*K, C, H, W]')
    channels, height, width = features.shape[1:]
    orbit = features.reshape(batch, views, channels, height * width)
    orbit = orbit.permute(0, 1, 3, 2).reshape(batch, views * height * width, channels)
    sorted_features = orbit.sort(dim=1, descending=True).values
    return sorted_features, sorted_features[:, 0]


def rbf_mmd(real, synthetic):
    if real.ndim < 2 or synthetic.ndim != real.ndim or real.shape[1:] != synthetic.shape[1:]:
        raise ValueError('MMD inputs must have matching non-batch dimensions')
    real = real.flatten(1)
    synthetic = synthetic.flatten(1)
    samples = torch.cat((real, synthetic))
    if len(real) < 2 or len(synthetic) < 2:
        raise ValueError('MMD requires at least two real and two synthetic samples')
    squared_distances = torch.cdist(samples, samples).square()
    off_diagonal = ~torch.eye(len(samples), dtype=torch.bool, device=samples.device)
    sigma = squared_distances.detach()[off_diagonal].mean().clamp_min(
        torch.finfo(samples.dtype).eps)
    kernel = torch.exp(-squared_distances / (2 * sigma))
    real_count = len(real)
    return (kernel[:real_count, :real_count].mean()
            + kernel[real_count:, real_count:].mean()
            - 2 * kernel[:real_count, real_count:].mean())


def orbit_loss(real_sorted, real_top, synthetic_sorted, synthetic_top,
               lambda_all, lambda_top):
    loss_all = rbf_mmd(real_sorted.detach(), synthetic_sorted)
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


def resume_distillation(path, images, optimizer, expected, common):
    if not os.path.exists(path):
        return 0, []
    saved = torch.load(path, map_location='cpu', weights_only=True)
    for key, value in expected.items():
        if saved.get(key) != value:
            raise ValueError('Cannot resume: checkpoint %s differs from this run' % key)
    saved_pixels = saved.get('pixels')
    optimizer_state = saved.get('optimizer_state')
    iteration = saved.get('iteration')
    history = saved.get('loss_history')
    expected_shape = (len(images), *images[0].shape)
    if (not isinstance(saved_pixels, torch.Tensor)
            or saved_pixels.shape != expected_shape
            or not torch.isfinite(saved_pixels).all()
            or not isinstance(optimizer_state, dict)
            or not isinstance(iteration, int) or iteration < 0
            or not isinstance(history, torch.Tensor) or history.shape != (iteration, 3)):
        raise ValueError('Cannot resume: invalid pixels, optimizer, iteration or loss history')
    with torch.no_grad():
        for image, saved_image in zip(images, saved_pixels):
            image.copy_(saved_image.to(image.device))
    optimizer.load_state_dict(optimizer_state)
    common.restore_rng(saved['rng_state'])
    print('Resumed orbit matching at iteration %d: %s' % (iteration, path))
    return iteration, history.tolist()



# using saved teacher, 
# (ENV) mmoslem3@rack15-04:~/scratch/MKDT/OT-SSL-DD$ python new_idea.py 
# Hyper-parameters: 
# Hyper-parameters: 
#  {'dataset': 'CIFAR10', 'model': 'ResNet18BN', 'percentage': 0.2, 'teacher_source': 'pretrained', 'teacher_ssl': 'simclr', 'teacher_seeds': [0], 'teacher_path': '/scratch/mmoslem3/MKDT/Evaluate_benchamrk/full_ssl_checkpoints_1000ep_3seeds', 'K_views': 8, 'lambda_all': 1, 'lambda_top': 2, 'Iteration': 8000, 'lr_img': 0.5, 'momentum': 0.9, 'batch_real': 2048, 'batch_syn': 1024, 'save_every': 500, 'num_exp': 1, 'label_percentage': 1, 'ssl_method': 'simclr', 'epoch_eval_train': 2000, 'ssl_log_every': 100, 'epoch_linear_train': 100, 'lr_linear': 0.1, 'batch_linear': 256, 'runs': 1, 'seed': 0, 'resume': False, 'device': 'cuda', 'data_path': '/scratch/mmoslem3/data', 'save_path': '/scratch/mmoslem3/MKDT/OT-SSL-DD/result_new_idea'}
# for 500 iter: 41.51

def main():
    parser = argparse.ArgumentParser(description='Augmentation-Aligned Distribution Matching')
    parser.add_argument('--dataset', choices=['CIFAR10', 'CIFAR100', 'TinyImageNet'], default='CIFAR10')
    parser.add_argument('--model', choices=['ConvNetBN', 'ResNet18BN'], default='ConvNetBN')
    parser.add_argument('--percentage', type=float, default=1)
    parser.add_argument('--teacher_source', choices=['pretrained', 'random'], default='pretrained')
    parser.add_argument('--teacher_ssl', choices=['simclr', 'barlowtwins'], default='simclr')
    parser.add_argument('--teacher_seeds', type=int, nargs='+', default=[0,1,2])
    parser.add_argument('--teacher_path', default=str(
        ROOT / 'Evaluate_benchamrk/full_ssl_checkpoints_1000ep_3seeds'))
    parser.add_argument('--K_views', type=int, default=8)
    parser.add_argument('--lambda_all', type=float, default=0.8)
    parser.add_argument('--lambda_top', type=float, default=0.5)
    parser.add_argument('--Iteration', type=int, default=8000)
    parser.add_argument('--lr_img', type=float, default=0.002)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--batch_real', type=int, default=512)
    parser.add_argument('--batch_syn', type=int, default=512)
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
    parser.add_argument('--save_path', default=str(ROOT / 'OT-SSL-DD/result_new_idea2'))
    args = parser.parse_args()
    benchmark_model = args.model.removesuffix('BN')

    if args.epoch_eval_train is None:
        args.epoch_eval_train = SSL_EPOCHS[args.dataset].get(args.percentage)
    if args.epoch_linear_train is None:
        args.epoch_linear_train = PROBE_EPOCHS[args.label_percentage]
    if args.epoch_eval_train is None:
        parser.error('No SSL epoch schedule for this size; set --ssl-epochs explicitly')

    if (not 0 < args.percentage <= 100 or args.Iteration < 1 or args.lr_img <= 0
            or min(args.K_views, args.save_every, args.num_exp) < 1
            or min(args.batch_real, args.batch_syn) < 2
            or not 0 <= args.momentum < 1 or min(args.lambda_all, args.lambda_top) < 0
            or args.lambda_all + args.lambda_top == 0
            or min(args.epoch_eval_train, args.epoch_linear_train, args.runs,
                   args.batch_linear, args.ssl_log_every) < 1
            or args.lr_linear <= 0):
        parser.error('Invalid percentage, iteration, batch, optimization, loss or save setting')
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA is unavailable; run this program inside a GPU allocation')

    common = reference_modules()['common']
    common.seed_all(args.seed)
    os.makedirs(args.save_path, exist_ok=True)
    print('Hyper-parameters: \n', args.__dict__)

    channel, im_size, _, _, mean, std, train_set, _, _ = get_dataset(args.dataset, args.data_path)
    if channel != 3:
        raise ValueError('Orbit matching expects three-channel images')
    num_syn = int(len(train_set) * args.percentage / 100)
    if num_syn < 2:
        raise ValueError('The requested percentage produces fewer than two synthetic images')
    mean_cpu = torch.tensor(mean).view(1, channel, 1, 1)
    std_cpu = torch.tensor(std).view(1, channel, 1, 1)
    random_normalization = (mean_cpu.to(args.device), std_cpu.to(args.device))
    augmentation = ssl_augmentation(im_size[0]).to(args.device)
    expected = {
        'dataset': args.dataset,
        'model': args.model,
        'percentage': args.percentage,
        'method': 'AugmentationAlignedDM',
        'teacher_source': args.teacher_source,
        'K_views': args.K_views,
        'feature_layer': ('pre_global_average_pool' if benchmark_model == 'ResNet18'
                          else 'pre_classifier_spatial_map'),
        'lambda_all': args.lambda_all,
        'lambda_top': args.lambda_top,
        'augmentation': 'random_resized_crop_flip_color_jitter_grayscale',
        'kernel': 'single_rbf_joint_mean_squared_distance',
        'lr_img': args.lr_img,
        'momentum': args.momentum,
        'batch_real': args.batch_real,
        'batch_syn': args.batch_syn,
    }

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
                    args.teacher_path, args.dataset, benchmark_model, 'full', 'pt100',
                    args.teacher_ssl, 'seed_%03d' % teacher_seed, 'encoder.pt')
                encoder, _, teacher_saved = common.load_encoder(
                    checkpoint, device=args.device)
                teacher_mean, teacher_std = (
                    torch.tensor(value, device=args.device).view(1, channel, 1, 1)
                    for value in teacher_saved['normalization'])
                teachers.append(SpatialFeatures(encoder).eval().requires_grad_(False))
                normalizations.append((teacher_mean, teacher_std))
                teacher_checkpoints.append(checkpoint)
                print('Loaded teacher seed %d: %s' % (teacher_seed, checkpoint))
            teacher_identity = {
                'teacher_ssl': args.teacher_ssl,
                'teacher_checkpoints': teacher_checkpoints,
                'teacher_sampling': 'uniform_one_per_iteration',
            }
        else:
            teacher_identity = {
                'teacher_initialization': 'random_per_iteration',
                'teacher_seed_base': exp_seed,
            }
            print('Using a fresh random teacher every iteration')

        common.seed_all(exp_seed)
        initial_indices = torch.randperm(len(train_set))[:num_syn]
        initial_pixels = dataset_pixels(train_set, initial_indices, mean_cpu, std_cpu)
        initial_pixels = initial_pixels.to(args.device).clamp_(0, 1)
        initial_pixels_reference = initial_pixels.detach().clone()
        images = nn.ParameterList(nn.Parameter(image) for image in initial_pixels)
        # optimizer_img = torch.optim.SGD(images, lr=args.lr_img, momentum=args.momentum)
        optimizer_img = torch.optim.Adam(images, lr=args.lr_img) # optimizer_img for synthetic data
        optimizer_img.zero_grad()

        result_path = os.path.join(
            args.save_path, 'res_AADM_%s_%s_%gpercent_exp%d.pt'
            % (args.dataset, args.model, args.percentage, exp))
        start_iteration = 0
        loss_history = []
        run_identity = dict(expected, **teacher_identity, seed=exp_seed)
        if args.resume:
            start_iteration, loss_history = resume_distillation(
                result_path, images, optimizer_img, run_identity, common)
        if start_iteration >= args.Iteration:
            print('Already complete: %s (%d iterations)' % (result_path, start_iteration))
        else:
            print('%s orbit matching begins, synthetic images = %d' % (get_time(), num_syn))
        for iteration in range(start_iteration + 1, args.Iteration + 1):
            if args.teacher_source == 'random':
                devices = [torch.cuda.current_device()] if args.device == 'cuda' else []
                with torch.random.fork_rng(devices=devices):
                    torch.manual_seed(exp_seed + iteration)
                    encoder, _ = common.build_encoder(benchmark_model, im_size[0])
                teachers = [SpatialFeatures(
                    encoder).to(args.device).eval().requires_grad_(False)]
                normalizations = [random_normalization]
            assert len(teachers) == len(normalizations) > 0
            teacher_index = (torch.randint(len(teachers), (1,)).item()
                             if args.teacher_source == 'pretrained' and len(teachers) > 1 else 0)
            teacher = teachers[teacher_index]
            teacher_mean, teacher_std = normalizations[teacher_index]

            real_indices = torch.randint(len(train_set), (min(args.batch_real, len(train_set)),))
            synthetic_indices = torch.randperm(num_syn, device=args.device)[:min(args.batch_syn, num_syn)]
            real_pixels = dataset_pixels(train_set, real_indices, mean_cpu, std_cpu).to(args.device)
            selected_images = [images[index] for index in synthetic_indices.tolist()]
            optimizer_img.zero_grad(set_to_none=True)
            synthetic_batch = torch.stack(selected_images)

            with torch.no_grad():
                real_sorted, real_top = sorted_orbit_features(
                    real_pixels, teacher, augmentation, args.K_views,
                    teacher_mean, teacher_std)
            synthetic_sorted, synthetic_top = sorted_orbit_features(
                synthetic_batch, teacher, augmentation, args.K_views,
                teacher_mean, teacher_std)
            loss, loss_all, loss_top = orbit_loss(
                real_sorted, real_top, synthetic_sorted, synthetic_top,
                args.lambda_all, args.lambda_top)
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite orbit matching loss')
            loss.backward()
            if any(image.grad is None or not torch.isfinite(image.grad).all() for image in selected_images):
                raise FloatingPointError('Non-finite or missing synthetic pixel gradient')
            optimizer_img.step()
            with torch.no_grad():
                for image in selected_images:
                    image.clamp_(0, 1)
            loss_history.append([loss.item(), loss_all.item(), loss_top.item()])

            if iteration % 50 == 0:
                with torch.no_grad():
                    current_pixels = torch.stack([image.detach() for image in images])
                    pixel_drift = ((current_pixels - initial_pixels_reference) * 255).square().mean().sqrt().item()
                print('%s iter = %05d, loss = %.6f, all = %.6f, top = %.6f, pixel drift RMS = %.5f/255'
                      % (get_time(), iteration, loss.item(), loss_all.item(),
                         loss_top.item(), pixel_drift), flush=True)

            if iteration % args.save_every == 0 or iteration == args.Iteration:
                pixels_cpu = torch.stack([image.detach().cpu() for image in images])
                normalized = (pixels_cpu - mean_cpu) / std_cpu
                preview_path = os.path.join(
                    args.save_path, 'vis_AADM_%s_%s_%gpercent_exp%d_iter%d.png'
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
                    'dis_metric': 'channel_sorted_augmentation_orbit_dual_rbf_mmd',
                    'initialization': 'random_real_images',
                    'input_size': list(im_size),
                    'optimizer_state': optimizer_img.state_dict(),
                }
                common.atomic_save(result_path, result)
                print('Saved synthetic data to %s' % result_path)

        if start_iteration < args.Iteration:
            del real_pixels, selected_images, synthetic_batch
            del real_sorted, real_top, synthetic_sorted, synthetic_top
            del loss, loss_all, loss_top, result
        teachers = []
        normalizations = []
        teacher = encoder = teacher_mean = teacher_std = teacher_saved = None
        del images, optimizer_img, initial_pixels, initial_pixels_reference
        if args.device == 'cuda':
            torch.cuda.empty_cache()
        saved = torch.load(result_path, map_location='cpu', weights_only=True)
        if saved.get('iteration', -1) < args.Iteration:
            raise ValueError('Final synthetic checkpoint is incomplete: %s' % result_path)
        torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', '1')))
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        output = Path(args.save_path) / 'evaluation' / ('exp_%03d_%s' % (exp, stamp))
        output.mkdir(parents=True, exist_ok=False)
        config = {key: str(value) if isinstance(value, Path) else value
                  for key, value in vars(args).items()}
        config.update(synthetic_checkpoint=str(Path(result_path).resolve()),
                      save_path=str(output.resolve()), generation_experiment=exp,
                      std_ddof=1)
        common.atomic_json(output / 'config.json', config)
        print(json.dumps(config, indent=2), flush=True)
        accuracies = []

        for run in range(args.runs):
            seed = args.seed + run
            _, _, accuracy = evaluate_synset_SSL(
                run, args.model, saved['data'], args, output / ('seed_%03d' % seed),
                saved.get('normalization'))
            accuracies.append(accuracy * 100)
            summary = {
                'config': config,
                'completed_runs': len(accuracies),
                'expected_runs': args.runs,
                'complete': len(accuracies) == args.runs,
                'accuracies_percent': accuracies,
                'mean_percent': statistics.mean(accuracies),
                'std_percent': statistics.stdev(accuracies) if len(accuracies) > 1 else None,
            }
            common.atomic_json(output / 'summary.json', summary)
            print('seed=%d accuracy=%.2f%%' % (seed, accuracy * 100), flush=True)

        std_value = ('%.2f' % summary['std_percent']
                     if summary['std_percent'] is not None else 'N/A')
        print('Mean ± sample std: %.2f ± %s%%' % (summary['mean_percent'], std_value))
        print('Saved evaluation: %s' % output)


if __name__ == '__main__':
    main()
