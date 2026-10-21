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


def ssl_augmentation(protocol, dataset, method):
    size = protocol['input_size'][dataset]
    common, recipe = protocol['ssl'], protocol['methods'][method]
    kernel = 2 * math.ceil(3 * common['blur_sigma'][1]) + 1
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
            K.RandomGaussianBlur(kernel, tuple(common['blur_sigma']),
                                 p=recipe['blur_probabilities'][view]),
            K.RandomSolarize(thresholds=(128 / 255, 128 / 255), additions=0,
                             p=recipe['solarize_probabilities'][view]),
            K.RandomHorizontalFlip(p=common['flip_probability'])))
    return nn.ModuleList(views)


def dataset_pixels(dataset, indices, mean, std):
    indices = indices.cpu()
    if hasattr(dataset, 'data'):
        pixels = torch.as_tensor(dataset.data[indices.numpy()])
        if pixels.shape[-1] in (1, 3):
            pixels = pixels.permute(0, 3, 1, 2)
        return pixels.float().div_(255)
    if hasattr(dataset, 'images'):
        normalized = dataset.images[indices]
    else:
        normalized = torch.stack([dataset[int(index)][0] for index in indices])
    return normalized.float() * std + mean


def augmented_features(pixels, net, augmentation, views, normalization):
    mean, std = normalization
    features = [net((augmentation[view % 2](pixels) - mean) / std)
                for view in range(views)]
    return torch.stack(features).mean(dim=0) # one augmentation-averaged embedding per image


def matching_losses(real, synthetic):
    if (real.ndim != 2 or synthetic.ndim != 2 or not len(real)
            or not len(synthetic) or real.shape[1] != synthetic.shape[1]):
        raise ValueError('Expected nonempty real/synthetic embedding matrices of equal width')
    loss_dm = (real.detach().mean(0) - synthetic.mean(0)).square().sum()
    count = len(synthetic)
    if count < 2:
        return loss_dm, synthetic.sum() * 0
    neighbors = max(1, min(count - 1, round(0.2 * count)))
    with torch.no_grad():
        distances = torch.cdist(synthetic, synthetic).square()
        distances.fill_diagonal_(torch.inf)
        indices = distances.topk(neighbors, largest=False).indices
    neighbor_mean = synthetic[indices].mean(dim=1)
    log_p = F.log_softmax(synthetic, dim=1)
    log_q = F.log_softmax(neighbor_mean, dim=1)
    # Eq. 8-9: maximize KL to the nearest-neighbor mean, hence the negative sign.
    loss_icd = -(log_p.exp() * (log_p - log_q)).sum()
    return loss_dm, loss_icd


@torch.no_grad()
def kmeans(features, clusters, iterations):
    centers = features[torch.randperm(len(features), device=features.device)[:clusters]].clone()
    for _ in range(iterations):
        distances = torch.cdist(features, centers).square()
        membership = distances.argmin(dim=1)
        counts = torch.bincount(membership, minlength=clusters)
        sums = torch.zeros_like(centers)
        sums.index_add_(0, membership, features)
        updated = sums / counts.clamp_min(1)[:, None]
        empty = torch.where(counts == 0)[0]
        if len(empty):
            farthest = distances.min(dim=1).values.topk(len(empty)).indices
            updated[empty] = features[farthest]
        if torch.allclose(updated, centers, atol=1e-5, rtol=0):
            centers = updated
            break
        centers = updated
    return torch.cdist(features, centers).argmin(dim=1)


@torch.no_grad()
def diverse_representatives(features, count):
    if len(features) < count:
        raise ValueError('A real cluster is too small; reduce --clusters or change --seed')
    first = (features - features.mean(0)).square().sum(1).argmin().item()
    selected = []
    nearest = features.new_full((len(features),), torch.inf)
    index = first
    for _ in range(count):
        selected.append(index)
        distances = (features - features[index]).square().sum(1)
        nearest = torch.minimum(nearest, distances)
        nearest[selected] = -1 # also keep representatives unique when embeddings tie
        index = nearest.argmax().item()
    return torch.tensor(selected, device=features.device)


@torch.no_grad()
def cluster_real_data(dataset, net, normalization, mean, std, args):
    features = []
    teacher_mean, teacher_std = normalization
    for start in range(0, len(dataset), args.cluster_batch):
        indices = torch.arange(start, min(start + args.cluster_batch, len(dataset)))
        pixels = dataset_pixels(dataset, indices, mean, std).to(args.device)
        features.append(F.normalize(net((pixels - teacher_mean) / teacher_std), dim=1))
    features = torch.cat(features)
    if not torch.isfinite(features).all():
        raise FloatingPointError('Non-finite clustering features')
    membership = kmeans(features, args.clusters, args.kmeans_iterations)
    initial_indices = []
    for cluster in range(args.clusters):
        members = torch.where(membership == cluster)[0]
        chosen = diverse_representatives(features[members], args.images_per_cluster)
        initial_indices.append(members[chosen])
        print('cluster %d: %d real images -> %d synthetic images'
              % (cluster, len(members), len(chosen)), flush=True)
    return membership.cpu(), torch.cat(initial_indices).cpu()


def self_test():
    torch.manual_seed(7)
    real = torch.randn(9, 4, dtype=torch.double, requires_grad=True)
    synthetic = torch.randn(5, 4, dtype=torch.double, requires_grad=True)
    dm, icd = matching_losses(real, synthetic)
    assert dm >= 0 and icd < 0
    (dm + icd).backward()
    assert real.grad is None and torch.isfinite(synthetic.grad).all()
    assert synthetic.grad.abs().sum() > 0
    assert torch.autograd.gradcheck(lambda x: sum(matching_losses(real, x)), (synthetic,))
    assert matching_losses(real, synthetic[:1])[1] == 0
    assert matching_losses(real, torch.ones_like(synthetic))[1].abs() < 1e-12
    pair = torch.tensor([[2., 0.], [0., 1.]], dtype=torch.double)
    log_p = pair.log_softmax(1)
    expected = -(log_p.exp() * (log_p - log_p.flip(0))).sum()
    assert torch.allclose(matching_losses(pair, pair)[1], expected)
    assert matching_losses(pair, pair)[0] == 0

    features = torch.cat((torch.randn(20, 3) * 0.01 - 3,
                          torch.randn(20, 3) * 0.01 + 3))
    membership = kmeans(features, 2, 30)
    assert membership[:20].unique().numel() == membership[20:].unique().numel() == 1
    assert membership[0] != membership[-1]
    assert diverse_representatives(torch.ones(8, 3), 5).unique().numel() == 5

    net = nn.Sequential(nn.Flatten(), nn.Linear(12, 4)).double().requires_grad_(False)
    pixels = torch.randn(5, 3, 2, 2, dtype=torch.double, requires_grad=True)
    views = [lambda x: x, lambda x: 2 * x]
    output = augmented_features(pixels, net, views, 4, (0, 1))
    expected = (net(pixels) + net(2 * pixels)) / 2
    assert output.shape == (5, 4) and torch.allclose(output, expected)
    sum(matching_losses(real.detach(), output)).backward()
    assert torch.isfinite(pixels.grad).all() and pixels.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in net.parameters())
    print('Self-test passed: clustering, unique initialization, DM, KL sign/direction and pixel gradients')


def main():
    parser = argparse.ArgumentParser(description='Cluster-Conditioned DM + Intra-Cluster Diversity')
    parser.add_argument('--dataset', choices=list(SSL_EPOCHS), default='CIFAR10')
    parser.add_argument('--model', choices=['ConvNetBN', 'ResNet18BN'], default='ConvNetBN')
    parser.add_argument('--clusters', type=int, default=20)
    parser.add_argument('--images_per_cluster', type=int, default=5)
    parser.add_argument('--teacher_source', choices=['pretrained', 'random'], default='pretrained')
    parser.add_argument('--teacher_ssl', choices=['simclr', 'barlowtwins'], default='simclr')
    parser.add_argument('--teacher_seeds', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--teacher_path', default=str(
        ROOT / 'Evaluate_benchamrk/full_ssl_checkpoints_1000ep_3seeds'))
    parser.add_argument('--K_views', type=int, default=2)
    parser.add_argument('--lambda_div', type=float, default=10.0)
    parser.add_argument('--Iteration', type=int, default=20000)
    parser.add_argument('--lr_img', type=float, default=1.0)
    parser.add_argument('--momentum', type=float, default=0.5)
    parser.add_argument('--batch_real', type=int, default=256, help='real images per cluster')
    parser.add_argument('--cluster_batch', type=int, default=256)
    parser.add_argument('--kmeans_iterations', type=int, default=50)
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
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--data_path', default=str(ROOT.parent / 'data'))
    parser.add_argument('--save_path', default=str(ROOT / 'OT-SSL-DD/result_hier'))
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if (min(args.clusters, args.images_per_cluster, args.K_views, args.Iteration,
            args.batch_real, args.cluster_batch, args.kmeans_iterations, args.save_every,
            args.num_exp, args.runs, args.batch_linear, args.ssl_log_every) < 1
            or args.K_views % 2 or args.lr_img <= 0 or args.lr_linear <= 0
            or not 0 <= args.momentum < 1 or args.lambda_div < 0
            or any(seed < 0 for seed in args.teacher_seeds) or args.seed < 0):
        parser.error('Counts/rates must be positive, K_views even, momentum in [0,1), weights/seeds nonnegative')
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA is unavailable; run inside a GPU allocation')
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', '1')))
    common = reference_modules()['common']
    protocol = json.loads((ROOT / 'Evaluate_benchamrk/protocol.json').read_text())
    channel, im_size, _, _, mean, std, train_set, _, _ = get_dataset(args.dataset, args.data_path)
    num_syn = args.clusters * args.images_per_cluster
    if num_syn < 2 or num_syn > len(train_set):
        parser.error('clusters * images_per_cluster must be between 2 and the real dataset size')
    args.percentage = 100 * num_syn / len(train_set)
    if int(len(train_set) * args.percentage / 100) < num_syn:
        args.percentage = math.nextafter(args.percentage, math.inf) # avoid losing one image to float rounding in eval.py
    if args.epoch_eval_train is None:
        args.epoch_eval_train = SSL_EPOCHS[args.dataset].get(args.percentage)
    if args.epoch_linear_train is None:
        args.epoch_linear_train = PROBE_EPOCHS[args.label_percentage]
    if args.epoch_eval_train is None:
        parser.error('No SSL schedule for %g%% (%d images); set --ssl-epochs'
                     % (args.percentage, num_syn))
    if min(args.epoch_eval_train, args.epoch_linear_train) < 1:
        parser.error('SSL/probe epochs must be positive')
    print('Hyper-parameters: \n', args.__dict__)
    mean_cpu, std_cpu = (torch.tensor(value).view(1, channel, 1, 1) for value in (mean, std))
    mean_gpu, std_gpu = mean_cpu.to(args.device), std_cpu.to(args.device)
    augmentation = ssl_augmentation(protocol, args.dataset, args.ssl_method).to(args.device)
    benchmark_model = args.model.removesuffix('BN')
    os.makedirs(args.save_path, exist_ok=True)

    for exp in range(args.num_exp):
        print('\n================== Exp %d ==================\n' % exp)
        common.seed_all(args.seed + exp)
        teachers, normalizations, teacher_hashes = [], [], []
        seeds = args.teacher_seeds if args.teacher_source == 'pretrained' else args.teacher_seeds[:1]
        for teacher_seed in seeds:
            path = Path(args.teacher_path) / args.dataset / benchmark_model / 'full' / 'pt100'
            path = path / args.teacher_ssl / ('seed_%03d' % teacher_seed) / 'encoder.pt'
            encoder, _, metadata = common.load_encoder(path, device=args.device)
            if (metadata['source'], metadata['model'], metadata['method'], metadata['seed'],
                    metadata['input_size']) != (args.dataset, benchmark_model, args.teacher_ssl,
                                               teacher_seed, im_size[0]):
                raise ValueError('Wrong dataset, model, SSL method, seed or resolution in %s' % path)
            teachers.append(encoder)
            normalizations.append(tuple(torch.tensor(value, device=args.device).view(1, channel, 1, 1)
                                        for value in metadata['normalization']))
            teacher_hashes.append(metadata['encoder_sha256'])
            print('Loaded teacher seed %d: %s' % (teacher_seed, path), flush=True)
        print('Clustering uses pretrained teacher seed %d; matching uses %s teachers'
              % (seeds[0], args.teacher_source), flush=True)
        identity = {key: getattr(args, key) for key in (
            'dataset', 'model', 'clusters', 'images_per_cluster', 'percentage', 'teacher_source',
            'teacher_ssl', 'K_views', 'lambda_div', 'lr_img', 'momentum', 'batch_real',
            'kmeans_iterations', 'cluster_batch', 'ssl_method')}
        identity.update(method='ClusterConditionedDM_ICD', seed=args.seed + exp,
                        teacher_seeds=seeds, teacher_hashes=teacher_hashes,
                        augmentation_protocol={'ssl': protocol['ssl'],
                                               'recipe': protocol['methods'][args.ssl_method]},
                        teacher_sampling=('uniform_one_per_iteration' if args.teacher_source == 'pretrained'
                                          else 'fresh_random_per_iteration'),
                        clustering_feature='l2_normalized_encoder_embedding')
        result_path = Path(args.save_path) / ('res_CCDM_%s_%s_%gpercent_exp%d.pt'
                      % (args.dataset, args.model, args.percentage, exp))
        common.seed_all(args.seed + exp)
        saved = None
        if result_path.exists():
            if not args.resume:
                raise FileExistsError('Use --resume or a different --save_path: %s' % result_path)
            saved = torch.load(result_path, map_location='cpu', weights_only=True)
            if any(saved.get(key) != value for key, value in identity.items()):
                raise ValueError('Cannot resume a checkpoint with different generation settings')
            membership, initial_indices = saved['membership'], saved['initial_indices']
        else:
            membership, initial_indices = cluster_real_data(
                train_set, teachers[0], normalizations[0], mean_cpu, std_cpu, args)
        if (membership.shape != (len(train_set),) or membership.min() < 0
                or membership.max() >= args.clusters
                or torch.bincount(membership, minlength=args.clusters).min() < args.images_per_cluster
                or initial_indices.shape != (num_syn,) or initial_indices.unique().numel() != num_syn
                or initial_indices.min() < 0 or initial_indices.max() >= len(train_set)):
            raise ValueError('Invalid clustering or initialization indices')
        members = [torch.where(membership == cluster)[0] for cluster in range(args.clusters)]
        initial_pixels = dataset_pixels(train_set, initial_indices, mean_cpu, std_cpu).to(args.device)
        image_syn = nn.Parameter((initial_pixels - mean_gpu) / std_gpu)
        optimizer_img = torch.optim.SGD([image_syn], lr=args.lr_img, momentum=args.momentum)
        start_iteration, loss_history = 0, []
        if saved is not None:
            if saved['data'].shape != image_syn.shape or not torch.isfinite(saved['data']).all():
                raise ValueError('Invalid saved synthetic images')
            with torch.no_grad():
                image_syn.copy_(saved['data'].to(args.device))
            optimizer_img.load_state_dict(saved['optimizer_state'])
            start_iteration, loss_history = saved['iteration'], saved['loss_history'].tolist()
            if start_iteration < 0 or len(loss_history) != start_iteration:
                raise ValueError('Invalid saved iteration/loss history')
            common.restore_rng(saved['rng_state'])
            print('Resumed at iteration %d' % start_iteration)
        if args.teacher_source == 'random':
            teachers.clear() # keep only the clustering, not a pretrained matching teacher
            normalizations.clear()
        encoder = None

        print('%s cluster matching begins, synthetic images = %d' % (get_time(), num_syn))
        for it in range(start_iteration + 1, args.Iteration + 1):
            if args.teacher_source == 'random':
                net, _ = common.build_encoder(benchmark_model, im_size[0])
                net = net.to(args.device).eval().requires_grad_(False)
                normalization = (mean_gpu, std_gpu)
            else:
                index = torch.randint(len(teachers), (1,)).item()
                net, normalization = teachers[index], normalizations[index]
            optimizer_img.zero_grad(set_to_none=True)
            loss_dm_total, loss_icd_total = 0.0, 0.0
            for cluster in range(args.clusters):
                indices = members[cluster]
                indices = indices[torch.randperm(len(indices))[:args.batch_real]]
                real_pixels = dataset_pixels(train_set, indices, mean_cpu, std_cpu).to(args.device)
                first = cluster * args.images_per_cluster
                synthetic_pixels = image_syn[first:first + args.images_per_cluster] * std_gpu + mean_gpu
                with torch.no_grad():
                    output_real = augmented_features(
                        real_pixels, net, augmentation, args.K_views, normalization)
                output_syn = augmented_features(
                    synthetic_pixels, net, augmentation, args.K_views, normalization)
                loss_dm, loss_icd = matching_losses(output_real, output_syn)
                loss = loss_dm + args.lambda_div * loss_icd
                if not torch.isfinite(loss):
                    raise FloatingPointError('Non-finite loss at iteration %d, cluster %d' % (it, cluster))
                loss.backward() # accumulate the cluster sum without retaining all cluster graphs
                loss_dm_total += loss_dm.item()
                loss_icd_total += loss_icd.item()
            if image_syn.grad is None or not torch.isfinite(image_syn.grad).all():
                raise FloatingPointError('Non-finite or missing synthetic pixel gradients')
            optimizer_img.step()
            with torch.no_grad():
                image_syn.copy_((image_syn * std_gpu + mean_gpu).clamp(0, 1).sub(mean_gpu).div(std_gpu))
            loss_total = loss_dm_total + args.lambda_div * loss_icd_total
            loss_history.append([loss_total, loss_dm_total, loss_icd_total])
            if it % 50 == 0 or it == 1 or it == args.Iteration:
                with torch.no_grad():
                    drift = ((image_syn * std_gpu + mean_gpu - initial_pixels) * 255).square().mean().sqrt()
                print('%s iter = %05d, loss = %.6f, DM = %.6f, ICD = %.6f, pixel drift RMS = %.5f/255'
                      % (get_time(), it, loss_total, loss_dm_total, loss_icd_total, drift), flush=True)
            if it % args.save_every == 0 or it == args.Iteration:
                data_cpu = image_syn.detach().cpu()
                pixels_cpu = (data_cpu * std_cpu + mean_cpu).clamp(0, 1)
                preview = Path(args.save_path) / ('vis_CCDM_%s_%s_%gpercent_exp%d_iter%d.png'
                          % (args.dataset, args.model, args.percentage, exp, it))
                save_image(pixels_cpu[:256], preview, nrow=args.images_per_cluster)
                common.atomic_save(result_path, {
                    **identity, 'data': data_cpu, 'pixels': pixels_cpu,
                    'normalization': [mean, std], 'iteration': it,
                    'membership': membership, 'initial_indices': initial_indices,
                    'synthetic_clusters': torch.arange(args.clusters).repeat_interleave(args.images_per_cluster),
                    'initialization': 'cluster_center_then_farthest_point_representatives',
                    'loss_history': torch.tensor(loss_history), 'rng_state': common.rng_state(),
                    'optimizer_state': optimizer_img.state_dict(),
                })
                print('Saved synthetic data to %s' % result_path, flush=True)

        teachers.clear()
        normalizations.clear()
        net = normalization = output_real = output_syn = loss = loss_dm = loss_icd = None
        del image_syn, optimizer_img, initial_pixels
        if args.device == 'cuda':
            torch.cuda.empty_cache()
        saved = torch.load(result_path, map_location='cpu', weights_only=True)
        if saved['iteration'] < args.Iteration:
            raise ValueError('Final synthetic checkpoint is incomplete: %s' % result_path)
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        output = Path(args.save_path) / 'evaluation' / ('exp_%03d_%s' % (exp, stamp))
        output.mkdir(parents=True, exist_ok=False)
        config = dict(vars(args), synthetic_checkpoint=str(result_path.resolve()),
                      generation_experiment=exp, std_ddof=1)
        config['save_path'] = str(output.resolve())
        common.atomic_json(output / 'config.json', config)
        accuracies = []
        for run in range(args.runs):
            _, _, accuracy = evaluate_synset_SSL(
                run, args.model, saved['data'], args, output / ('seed_%03d' % (args.seed + run)),
                saved.get('normalization'))
            accuracies.append(accuracy * 100)
            summary = dict(config=config, completed_runs=len(accuracies), expected_runs=args.runs,
                           complete=len(accuracies) == args.runs, accuracies_percent=accuracies,
                           mean_percent=statistics.mean(accuracies),
                           std_percent=statistics.stdev(accuracies) if len(accuracies) > 1 else None)
            common.atomic_json(output / 'summary.json', summary)
            print('seed=%d accuracy=%.2f%%' % (args.seed + run, accuracy * 100), flush=True)
        std_value = '%.2f' % summary['std_percent'] if summary['std_percent'] is not None else 'N/A'
        print('Mean ± sample std: %.2f ± %s%%' % (summary['mean_percent'], std_value))
        print('Saved evaluation: %s' % output)


if __name__ == '__main__':
    main()
