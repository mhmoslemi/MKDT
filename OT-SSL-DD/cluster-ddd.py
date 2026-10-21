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


# Every cluster is represented by the same number of synthetic images.
# Match three scalar similarity distributions: same-image positives,
# different-image negatives within a cluster, and negatives across clusters.


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


def encode(encoder, pixels, normalization):
    mean, std = normalization
    return F.normalize(encoder((pixels - mean) / std), dim=1)


@torch.no_grad()
def prepare_random_teacher(encoder, dataset, views, normalization, mean, std, args):
    batch_norms = [module for module in encoder.modules() if isinstance(module, nn.BatchNorm2d)]
    momentums = [module.momentum for module in batch_norms]
    for module in batch_norms:
        nn.init.ones_(module.weight)
        module.reset_running_stats()
        module.momentum = None
        module.train()
    batch_size = min(args.batch_real, len(dataset))
    for start in range(0, args.bn_calibration_pairs, batch_size):
        count = min(batch_size, args.bn_calibration_pairs - start)
        indices = torch.randperm(len(dataset))[:count]
        pixels = dataset_pixels(dataset, indices, mean, std).to(args.device)
        for view in views:
            encode(encoder, view(pixels), normalization)
    for module, momentum in zip(batch_norms, momentums):
        module.momentum = momentum
    encoder.eval()


@torch.no_grad()
def spherical_kmeans(features, clusters, iterations, seed):
    """Cosine k-means on unit-norm SSL embeddings."""
    generator = torch.Generator(device='cpu').manual_seed(seed)
    first = torch.randint(len(features), (1,), generator=generator).item()
    centers = [features[first]]
    nearest_distance = features.new_full((len(features),), float('inf'))
    for _ in range(1, clusters):
        distance = (1 - features @ centers[-1]).clamp_min(0)
        nearest_distance = torch.minimum(nearest_distance, distance)
        probabilities = nearest_distance.square().cpu()
        if probabilities.sum().item() == 0:
            index = torch.randint(len(features), (1,), generator=generator).item()
        else:
            index = torch.multinomial(probabilities, 1, generator=generator).item()
        centers.append(features[index])
    centers = F.normalize(torch.stack(centers), dim=1)

    for _ in range(iterations):
        membership = (features @ centers.t()).argmax(dim=1)
        counts = torch.bincount(membership, minlength=clusters)
        sums = torch.zeros_like(centers)
        sums.index_add_(0, membership, features)
        updated = F.normalize(sums, dim=1)
        if (counts == 0).any():
            # Reseed empty clusters using points poorly explained by current centers.
            similarity = (features * centers[membership]).sum(dim=1)
            farthest = similarity.argsort()
            empty = torch.where(counts == 0)[0]
            for offset, cluster in enumerate(empty.tolist()):
                updated[cluster] = features[farthest[offset]]
        if torch.allclose(updated, centers, atol=1e-6):
            centers = updated
            break
        centers = updated

    membership = (features @ centers.t()).argmax(dim=1)
    counts = torch.bincount(membership, minlength=clusters)
    if counts.min().item() < 2:
        raise ValueError('K-means produced a cluster with fewer than two real images; '
                         'reduce --clusters or change --seed')
    return membership.cpu(), counts.cpu()


@torch.no_grad()
def cluster_real_data(args, dataset, teacher, normalization, teacher_identity,
                      mean, std, common, protocol):
    identity = {
        'dataset': args.dataset, 'data_path': os.path.abspath(args.data_path),
        'dataset_count': len(dataset), 'teacher': teacher_identity,
        'clusters': args.clusters, 'kmeans_iterations': args.kmeans_iterations,
        'seed': args.seed, 'feature': 'unaugmented_encoder_output_cosine',
        'implementation_sha256': common.file_hash(__file__)}
    path = Path(args.save_path) / ('clusters_%s.pt' % common.digest_json(identity))
    if path.exists():
        saved = torch.load(path, map_location='cpu', weights_only=True)
        if saved['identity'] != identity:
            raise ValueError('Cached clustering does not match this run')
        print('Loaded real clustering: %s' % path, flush=True)
        return saved['membership'], str(path)

    features = []
    for start in range(0, len(dataset), args.cluster_batch):
        indices = torch.arange(start, min(start + args.cluster_batch, len(dataset)))
        pixels = dataset_pixels(dataset, indices, mean, std).to(args.device)
        features.append(encode(teacher, pixels, normalization).cpu())
    features = F.normalize(torch.cat(features).to(args.device), dim=1)
    membership, counts = spherical_kmeans(
        features, args.clusters, args.kmeans_iterations, args.seed)
    common.atomic_save(path, {'identity': identity, 'membership': membership, 'counts': counts})
    print('Saved clustering: %s; cluster sizes: %s' % (path, counts.tolist()), flush=True)
    return membership, str(path)


def cluster_index(membership, clusters):
    members = [torch.where(membership == k)[0] for k in range(clusters)]
    counts = torch.tensor([len(indices) for indices in members], dtype=torch.long)
    offsets = torch.cat((counts.new_zeros(1), counts.cumsum(dim=0)[:-1]))
    flat = torch.cat(members)
    return members, counts, offsets, flat


@torch.no_grad()
def real_distributions(args, dataset, teachers, normalizations, teacher_identity,
                       membership, cluster_path, views, mean, std, protocol, common):
    identity = {
        'dataset': args.dataset, 'data_path': os.path.abspath(args.data_path),
        'dataset_count': len(dataset), 'clustering': cluster_path,
        'teachers': teacher_identity, 'real_pairs_per_cluster': args.real_pairs_per_cluster,
        'batch_real': args.batch_real, 'seed': args.seed,
        'augmentation_method': args.ssl_method, 'protocol': protocol,
        'implementation_sha256': common.file_hash(__file__)}
    path = Path(args.save_path) / ('real_cluster_relations_%s.pt' % common.digest_json(identity))
    if path.exists():
        saved = torch.load(path, map_location='cpu', weights_only=True)
        if saved['identity'] != identity:
            raise ValueError('Cached real relation distributions do not match this run')
        print('Loaded real cluster relations: %s' % path, flush=True)
        return saved['distributions'], str(path)

    members, counts, offsets, flat = cluster_index(membership, args.clusters)
    distributions = []
    for teacher_index, (teacher, normalization) in enumerate(zip(teachers, normalizations)):
        common.seed_all(args.seed + 10000 + teacher_index)
        teacher_distributions = []
        for cluster in range(args.clusters):
            samples = {'positive': [], 'intra': [], 'inter': []}
            for start in range(0, args.real_pairs_per_cluster, args.batch_real):
                count = min(args.batch_real, args.real_pairs_per_cluster - start)
                # Positive: two augmentations of the identical real image.
                positions = torch.randint(counts[cluster].item(), (count,))
                first_indices = members[cluster][positions]

                # Intra negative: a different real image inside the same cluster.
                shifts = torch.randint(1, counts[cluster].item(), (count,))
                second_indices = members[cluster][(positions + shifts) % counts[cluster]]

                # Inter negative: sample an outside cluster uniformly, then an image.
                other_clusters = torch.randint(args.clusters - 1, (count,))
                other_clusters += (other_clusters >= cluster).long()
                positions_outside = (torch.rand(count) * counts[other_clusters]).long()
                third_indices = flat[offsets[other_clusters] + positions_outside]

                anchors = dataset_pixels(dataset, first_indices, mean, std).to(args.device)
                inside = dataset_pixels(dataset, second_indices, mean, std).to(args.device)
                outside = dataset_pixels(dataset, third_indices, mean, std).to(args.device)
                augmented = torch.cat((views[0](anchors), views[1](anchors),
                                       views[1](inside), views[1](outside)))
                first, positive, intra, inter = encode(
                    teacher, augmented, normalization).split(count)
                samples['positive'].append((first * positive).sum(dim=1).cpu())
                samples['intra'].append((first * intra).sum(dim=1).cpu())
                samples['inter'].append((first * inter).sum(dim=1).cpu())

            teacher_distributions.append({name: torch.cat(values)
                                          for name, values in samples.items()})
            for name, values in teacher_distributions[-1].items():
                if not torch.isfinite(values).all():
                    raise FloatingPointError('Non-finite real %s similarities' % name)
        distributions.append(teacher_distributions)
        print('Teacher %d: cached %d real relations per type, per cluster'
              % (teacher_index, args.real_pairs_per_cluster), flush=True)
    common.atomic_save(path, {'identity': identity, 'distributions': distributions})
    print('Saved real cluster relations: %s' % path, flush=True)
    return distributions, str(path)


def gaussian_kernel(first, second, sigmas):
    differences = (first[:, None] - second[None, :]).square()
    result = torch.zeros_like(differences)
    for sigma in sigmas:
        result = result + torch.exp(-differences / (2 * sigma ** 2))
    return result / len(sigmas)


def mmd_loss(real, synthetic, sigmas, samples):
    """Biased (nonnegative) Gaussian-kernel MMD squared on scalar similarities."""
    if len(real) > samples:
        real = real[torch.randperm(len(real), device=real.device)[:samples]]
    if len(synthetic) > samples:
        synthetic = synthetic[torch.randperm(len(synthetic), device=synthetic.device)[:samples]]
    return (gaussian_kernel(real, real, sigmas).mean()
            + gaussian_kernel(synthetic, synthetic, sigmas).mean()
            - 2 * gaussian_kernel(real, synthetic, sigmas).mean()).clamp_min(0)


def synthetic_distributions(similarities, clusters, images_per_cluster):
    """Relations across two independently augmented views of the entire synset."""
    distributions = []
    diagonal = similarities.diagonal()
    for cluster in range(clusters):
        start = cluster * images_per_cluster
        end = start + images_per_cluster
        within = similarities[start:end, start:end]
        mask = ~torch.eye(images_per_cluster, device=similarities.device, dtype=torch.bool)
        inter = torch.cat((similarities[start:end, :start],
                           similarities[start:end, end:]), dim=1)
        distributions.append({
            'positive': diagonal[start:end],
            'intra': within[mask],
            'inter': inter.reshape(-1)})
    return distributions


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
            or not isinstance(history, torch.Tensor) or history.shape != (iteration, 4)):
        raise ValueError('Cannot resume: invalid pixels, optimizer, iteration or loss history')
    with torch.no_grad():
        for image, saved_image in zip(images, pixels):
            image.copy_(saved_image.to(image.device))
    optimizer.load_state_dict(optimizer_state)
    common.restore_rng(saved['rng_state'])
    print('Resumed cluster MMD at iteration %d: %s' % (iteration, path), flush=True)
    return iteration, history.tolist()


def distill(args, experiment, exp, dataset, im_size, mean, std, membership,
            teachers, normalizations, targets, cluster_path, target_path,
            teacher_identity, views, common):
    exp_seed = args.seed + exp
    common.seed_all(exp_seed)
    num_syn = args.num_syn
    images_per_cluster = num_syn // args.clusters
    if args.init == 'real':
        if experiment == 'random':
            initial_indices = torch.randperm(len(dataset))[:num_syn]
        else:
            members, _, _, _ = cluster_index(membership, args.clusters)
            # The first m synthetic images belong to C_1, the next m to C_2, etc.
            initial_indices = torch.cat([
                indices[torch.randperm(len(indices))[:images_per_cluster]]
                for indices in members])
        initial_pixels = dataset_pixels(dataset, initial_indices, mean, std)
    else:
        initial_pixels = torch.rand(num_syn, 3, *im_size)
    initial_pixels.clamp_(0, 1)
    images = nn.ParameterList(nn.Parameter(image.to(args.device)) for image in initial_pixels)
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(images, lr=args.lr_img, momentum=args.momentum)
    else:
        optimizer = torch.optim.Adam(images, lr=args.lr_img)

    identity = {
        'method': 'ClusterRelationalMMD', 'experiment': experiment,
        'dataset': args.dataset, 'model': args.model, 'percentage': args.percentage,
        'num_syn': num_syn, 'clusters': args.clusters, 'images_per_cluster': images_per_cluster,
        'seed': exp_seed, 'initialization': args.init,
        'cluster_path': cluster_path if experiment != 'random' else None,
        'teacher_identity': teacher_identity if experiment != 'random' else [],
        'target_path': target_path if experiment != 'random' else None,
        'lambda_intra': args.lambda_intra, 'lambda_inter': args.lambda_inter,
        'mmd_sigmas': args.mmd_sigmas, 'mmd_samples': args.mmd_samples,
        'view_repeats': args.view_repeats, 'optimizer': args.optimizer,
        'lr_img': args.lr_img, 'momentum': args.momentum,
        'feature_layer': 'encoder_output_before_projector',
        'augmentation_method': args.ssl_method,
        'augmentation_backend': 'kornia_tensor_benchmark_recipe',
        'teacher_reduction': 'mean_all_teachers'}
    result_path = Path(args.save_path) / ('res_CRMMD_%s_%s_%dimages_%s_exp%d.pt'
                                         % (args.dataset, args.model, num_syn, experiment, exp))
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
            'cluster_ids': (torch.arange(num_syn) // images_per_cluster
                            if experiment != 'random' else None),
            'normalization': [mean.flatten().tolist(), std.flatten().tolist()],
            'input_size': list(pixels.shape[2:]), 'iteration': iteration,
            'loss_history': torch.tensor(history).reshape(-1, 4),
            'optimizer_state': optimizer.state_dict(), 'rng_state': common.rng_state()})
        print('Saved synthetic data to %s' % result_path, flush=True)

    print('%s %s begins: %d images, %d clusters, %d images/cluster, %d iterations'
          % (get_time(), experiment, num_syn, args.clusters,
             images_per_cluster, final_iteration), flush=True)
    for iteration in range(start + 1, final_iteration + 1):
        optimizer.zero_grad(set_to_none=True)
        synthetic = torch.stack(list(images))
        # Resample the downstream SSL augmentation pipeline every iteration.
        augmented = [(views[0](synthetic), views[1](synthetic))
                     for _ in range(args.view_repeats)]
        positive_loss = synthetic.new_zeros(())
        intra_loss = synthetic.new_zeros(())
        inter_loss = synthetic.new_zeros(())

        for teacher, normalization, teacher_targets in zip(teachers, normalizations, targets):
            relations = [{'positive': [], 'intra': [], 'inter': []}
                         for _ in range(args.clusters)]
            for first_images, second_images in augmented:
                first = encode(teacher, first_images, normalization)
                second = encode(teacher, second_images, normalization)
                similarities = first @ second.t()
                current = synthetic_distributions(similarities, args.clusters, images_per_cluster)
                for cluster, distribution in enumerate(current):
                    for name in ('positive', 'intra', 'inter'):
                        relations[cluster][name].append(distribution[name])

            for cluster in range(args.clusters):
                for name, accumulator in (('positive', 'positive_loss'),
                                          ('intra', 'intra_loss'), ('inter', 'inter_loss')):
                    if name == 'intra' and args.lambda_intra == 0:
                        continue
                    if name == 'inter' and args.lambda_inter == 0:
                        continue
                    synthetic_values = torch.cat(relations[cluster][name])
                    loss = mmd_loss(teacher_targets[cluster][name], synthetic_values,
                                    args.mmd_sigmas, args.mmd_samples)
                    if accumulator == 'positive_loss':
                        positive_loss = positive_loss + loss
                    elif accumulator == 'intra_loss':
                        intra_loss = intra_loss + loss
                    else:
                        inter_loss = inter_loss + loss

        reduction = len(teachers) * args.clusters
        positive_loss = positive_loss / reduction
        intra_loss = intra_loss / reduction
        inter_loss = inter_loss / reduction
        loss = 5*positive_loss + args.lambda_intra * intra_loss + args.lambda_inter * inter_loss
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite cluster relational MMD loss')
        loss.backward()
        if (any(image.grad is None for image in images)
                or not torch.isfinite(torch.stack([image.grad for image in images])).all()):
            raise FloatingPointError('Non-finite or missing synthetic pixel gradient')
        optimizer.step()
        with torch.no_grad():
            for image in images:
                image.clamp_(0, 1)
        history.append([loss.item(), positive_loss.item(), intra_loss.item(), inter_loss.item()])
        if iteration % 50 == 0 or iteration == 1:
            print('%s iter = %05d, loss = %.6f, pos = %.6f, intra = %.6f, inter = %.6f'
                  % (get_time(), iteration, loss.item(), positive_loss.item(),
                     intra_loss.item(), inter_loss.item()), flush=True)
        if iteration % args.save_every == 0 or iteration == final_iteration:
            save(iteration)
    if final_iteration == 0 or start >= final_iteration:
        save(start)
    return result_path, exp_seed


def main():
    parser = argparse.ArgumentParser(description='Cluster-conditioned SSL relational MMD distillation')
    parser.add_argument('--dataset', choices=['CIFAR10', 'CIFAR100', 'TinyImageNet'], default='CIFAR10')
    parser.add_argument('--model', choices=['ConvNetBN', 'ResNet18BN', 'VGG11BN',
                                          'ConvNet', 'ResNet18', 'VGG11'], default='ResNet18BN')
    parser.add_argument('--percentage', type=float, default=0.2)
    parser.add_argument('--num_syn', type=int, default=None,
                        help='Override percentage with an exact synthetic-image budget')
    parser.add_argument('--clusters', type=int, default=20)
    parser.add_argument('--kmeans_iterations', type=int, default=50)
    parser.add_argument('--cluster_batch', type=int, default=512)
    parser.add_argument('--experiment', choices=['random', 'full', 'all'], default='full')
    parser.add_argument('--init', choices=['noise', 'real'], default='real')
    parser.add_argument('--lambda_intra', type=float, default=1)
    parser.add_argument('--lambda_inter', type=float, default=0)
    parser.add_argument('--mmd_sigmas', type=float, nargs='+', default=[0.05, 0.1, 0.2, 0.4])
    parser.add_argument('--mmd_samples', type=int, default=256)
    parser.add_argument('--view_repeats', type=int, default=2)
    parser.add_argument('--teacher_source', choices=['pretrained', 'random'], default='pretrained')
    parser.add_argument('--teacher_ssl', choices=['simclr', 'barlowtwins'], default='simclr')
    parser.add_argument('--teacher_seeds', type=int, nargs='+', default=[0])
    # parser.add_argument('--teacher_seeds', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--teacher_path', default=str(
        ROOT / 'Evaluate_benchamrk/full_ssl_checkpoints_1000ep_3seeds'))
    parser.add_argument('--bn_calibration_pairs', type=int, default=8192)
    parser.add_argument('--real_pairs_per_cluster', type=int, default=512)
    parser.add_argument('--batch_real', type=int, default=128)
    parser.add_argument('--Iteration', type=int, default=8000)
    parser.add_argument('--lr_img', type=float, default=0.005)
    parser.add_argument('--optimizer', choices=['adam', 'sgd'], default='adam')
    parser.add_argument('--momentum', type=float, default=0.9)
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
    parser.add_argument('--skip_eval', action='store_true', default=False,
                        help='Disable final eval.py SSL training and linear probing; enabled by default')
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--data_path', default=str(ROOT.parent / 'data'))
    parser.add_argument('--save_path', default=str(ROOT / 'OT-SSL-DD/result_CRMMD2'))
    args = parser.parse_args()

    if (not 0 < args.percentage <= 100 or args.clusters < 2
            or min(args.kmeans_iterations, args.cluster_batch, args.real_pairs_per_cluster,
                   args.batch_real, args.mmd_samples, args.view_repeats,
                   args.save_every, args.num_exp) < 1
            or args.Iteration < 1 or args.lr_img <= 0 or not 0 <= args.momentum < 1
            or min(args.lambda_intra, args.lambda_inter) < 0
            or not args.mmd_sigmas or min(args.mmd_sigmas) <= 0
            or not all(math.isfinite(sigma) for sigma in args.mmd_sigmas)
            or args.bn_calibration_pairs < 1 or (args.num_syn is not None and args.num_syn < 1)):
        parser.error('Invalid clustering, budget, iteration, optimization or MMD setting')
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA is unavailable; run this program inside a GPU allocation')
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', '1')))
    reference = reference_modules()
    common = reference['common']
    common.seed_all(args.seed)
    os.makedirs(args.save_path, exist_ok=True)

    channel, im_size, _, _, mean, std, dataset, _, _ = get_dataset(args.dataset, args.data_path)
    num_syn = (int(len(dataset) * args.percentage / 100)
               if args.num_syn is None else args.num_syn)
    if channel != 3 or num_syn > len(dataset) or num_syn < 2 * args.clusters:
        parser.error('Need a three-channel dataset and at least two synthetic images per cluster')
    if num_syn % args.clusters:
        parser.error('Synthetic budget must be divisible by --clusters (equal images per cluster)')
    args.num_syn = num_syn
    args.percentage = 100 * num_syn / len(dataset)
    if not args.skip_eval:
        if args.epoch_eval_train is None:
            args.epoch_eval_train = SSL_EPOCHS[args.dataset].get(args.percentage)
        if args.epoch_linear_train is None:
            args.epoch_linear_train = PROBE_EPOCHS[args.label_percentage]
        if args.epoch_eval_train is None:
            parser.error('No SSL epoch schedule for this size; set --ssl-epochs explicitly')
        if (min(args.epoch_eval_train, args.epoch_linear_train, args.runs,
                args.batch_linear, args.ssl_log_every) < 1 or args.lr_linear <= 0):
            parser.error('Invalid downstream evaluation setting')
    print('Hyper-parameters: \n', args.__dict__)
    mean = torch.tensor(mean).view(1, channel, 1, 1)
    std = torch.tensor(std).view(1, channel, 1, 1)
    protocol = json.loads((ROOT / 'Evaluate_benchamrk/protocol.json').read_text())
    views = ssl_augmentation(protocol, args.dataset, args.ssl_method).to(args.device)
    experiments = ['random', 'full'] if args.experiment == 'all' else [args.experiment]
    teachers, normalizations, teacher_identity, targets = [], [], [], []
    membership, cluster_path, target_path = None, None, None

    if 'full' in experiments:
        if args.teacher_source == 'random':
            print('Warning: random teacher clustering need not form semantic groups.', flush=True)
        for seed in args.teacher_seeds:
            if args.teacher_source == 'pretrained':
                checkpoint = (Path(args.teacher_path) / args.dataset /
                              args.model.removesuffix('BN') / 'full' / 'pt100' /
                              args.teacher_ssl / ('seed_%03d' % seed) / 'encoder.pt')
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
            encoder = encoder.to(args.device).eval().requires_grad_(False)
            teachers.append(encoder)
            normalizations.append(tuple(torch.tensor(value, device=args.device).view(1, channel, 1, 1)
                                        for value in normalization))
            teacher_identity.append({**identity, 'normalization': normalization})
        if not teachers:
            parser.error('Specify at least one --teacher_seeds value')
        membership, cluster_path = cluster_real_data(
            args, dataset, teachers[0], normalizations[0], teacher_identity[0],
            mean, std, common, protocol)
        if (torch.bincount(membership, minlength=args.clusters).min().item()
                < args.num_syn // args.clusters and args.init == 'real'):
            parser.error('Some clusters have fewer real images than synthetic initialization slots')
        distributions, target_path = real_distributions(
            args, dataset, teachers, normalizations, teacher_identity,
            membership, cluster_path, views, mean, std, protocol, common)
        targets = [[{name: values.to(args.device) for name, values in cluster.items()}
                    for cluster in teacher_distributions]
                   for teacher_distributions in distributions]

    summaries = []
    for exp in range(args.num_exp):
        for experiment in experiments:
            print('\n================== Exp %d / %s ==================\n' % (exp, experiment))
            result_path, exp_seed = distill(
                args, experiment, exp, dataset, im_size, mean, std, membership,
                teachers, normalizations, targets, cluster_path, target_path,
                teacher_identity, views, common)
            if args.skip_eval:
                continue
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
            comparison_path = Path(args.save_path) / ('summary_CRMMD_%s_%s_%dimages.json'
                                                     % (args.dataset, args.model, num_syn))
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
