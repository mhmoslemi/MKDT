import argparse
import math
import os
import statistics
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image

from eval import (PROBE_EPOCHS, ROOT, SSL_EPOCHS, evaluate_synset_SSL, reference_modules)
from utils import DiffAugment, ParamDiffAug, get_dataset, get_time


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


def matching_losses(real, synthetic):
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
    first = (features - features.mean(0)).square().sum(1).argmin().item()
    selected = [first]
    nearest = features.new_full((len(features),), torch.inf)
    for _ in range(1, count):
        distances = (features - features[selected[-1]]).square().sum(1)
        nearest = torch.minimum(nearest, distances)
        nearest[selected] = -1
        candidates = torch.where(nearest >= 0)[0]
        selected.append(candidates[nearest[candidates].argmax()].item())
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
    membership = kmeans(features, args.clusters, args.kmeans_iterations)
    initial_indices = []
    for cluster in range(args.clusters):
        members = torch.where(membership == cluster)[0]
        chosen = diverse_representatives(features[members], args.images_per_cluster)
        initial_indices.append(members[chosen])
        print('cluster %d: %d real images -> %d synthetic images' % (cluster, len(members), len(chosen)), flush=True)
    return membership.cpu(), torch.cat(initial_indices).cpu()


def main():
    parser = argparse.ArgumentParser(description='Cluster-Conditioned DM + Intra-Cluster Diversity')
    parser.add_argument('--dataset', choices=list(SSL_EPOCHS), default='CIFAR10')
    parser.add_argument('--model', choices=['ConvNetBN', 'ResNet18BN'], default='ResNet18BN')
    parser.add_argument('--clusters', type=int, default=20)
    parser.add_argument('--images_per_cluster', type=int, default=5)
    parser.add_argument('--teacher_ssl', choices=['simclr', 'barlowtwins'], default='simclr')
    # parser.add_argument('--teacher_seeds', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--teacher_seeds', type=int, nargs='+', default=[0])
    parser.add_argument('--teacher_path', default=str(ROOT / 'Evaluate_benchamrk/full_ssl_checkpoints_1000ep_3seeds'))
    parser.add_argument('--dsa_strategy', default='color_crop_cutout_flip_scale_rotate')
    parser.add_argument('--lambda_div', type=float, default=1)
    parser.add_argument('--Iteration', type=int, default=1000)
    parser.add_argument('--lr_img', type=float, default=0.01)
    parser.add_argument('--optimizer', choices=['sgd', 'adam'], default='adam')
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--batch_real', type=int, default=256, help='real images per cluster')
    parser.add_argument('--cluster_batch', type=int, default=256)
    parser.add_argument('--kmeans_iterations', type=int, default=100)
    parser.add_argument('--save_png_every', type=int, default=100)
    parser.add_argument('--save_pt_every', type=int, default=500)
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
    parser.add_argument('--save_path', default=str(ROOT / 'OT-SSL-DD/result_hier2'))
    args = parser.parse_args()
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', '1')))
    common = reference_modules()['common']
    channel, im_size, _, _, mean, std, train_set, _, _ = get_dataset(args.dataset, args.data_path)
    num_syn = args.clusters * args.images_per_cluster
    args.percentage = 100 * num_syn / len(train_set)
    if int(len(train_set) * args.percentage / 100) < num_syn:
        args.percentage = math.nextafter(args.percentage, math.inf) # avoid losing one image to float rounding in eval.py
    if args.epoch_eval_train is None:
        args.epoch_eval_train = SSL_EPOCHS[args.dataset][args.percentage]
    if args.epoch_linear_train is None:
        args.epoch_linear_train = PROBE_EPOCHS[args.label_percentage]
    print('Hyper-parameters: \n', args.__dict__)
    mean_cpu, std_cpu = (torch.tensor(value).view(1, channel, 1, 1) for value in (mean, std))
    mean_gpu, std_gpu = mean_cpu.to(args.device), std_cpu.to(args.device)
    dsa_param = ParamDiffAug()
    benchmark_model = args.model.removesuffix('BN')
    os.makedirs(args.save_path, exist_ok=True)

    for exp in range(args.num_exp):
        print('\n================== Exp %d ==================\n' % exp)
        common.seed_all(args.seed + exp)
        teacher_seed = args.teacher_seeds[0]
        path = Path(args.teacher_path) / args.dataset / benchmark_model / 'full' / 'pt100'
        path = path / args.teacher_ssl / ('seed_%03d' % teacher_seed) / 'encoder.pt'
        teacher, _, metadata = common.load_encoder(path, device=args.device)
        normalization = tuple(torch.tensor(value, device=args.device).view(1, channel, 1, 1) for value in metadata['normalization'])
        print('Clustering uses pretrained teacher seed %d; DM uses a fresh random network every iteration' % teacher_seed, flush=True)
        run_config = dict(vars(args), method='ClusterConditionedDM_ICD', seed=args.seed + exp)
        result_path = Path(args.save_path) / ('res_CCDM_%s_%s_%gpercent_exp%d.pt' % (args.dataset, args.model, args.percentage, exp))
        common.seed_all(args.seed + exp)
        saved = None


        if args.resume and result_path.exists():
            saved = torch.load(result_path, map_location='cpu', weights_only=True)
            membership, initial_indices = saved['membership'], saved['initial_indices']
        else:
            membership, initial_indices = cluster_real_data(train_set, teacher, normalization, mean_cpu, std_cpu, args)

        members = [torch.where(membership == cluster)[0] for cluster in range(args.clusters)]
        initial_pixels = dataset_pixels(train_set, initial_indices, mean_cpu, std_cpu).to(args.device)
        image_syn = nn.Parameter((initial_pixels - mean_gpu) / std_gpu)
        optimizer_img = (torch.optim.SGD([image_syn], lr=args.lr_img, momentum=args.momentum) if args.optimizer == 'sgd'
                         else torch.optim.Adam([image_syn], lr=args.lr_img))
        start_iteration, loss_history = 0, []
        if saved is not None:
            with torch.no_grad():
                image_syn.copy_(saved['data'].to(args.device))
            optimizer_img.load_state_dict(saved['optimizer_state'])
            start_iteration, loss_history = saved['iteration'], saved['loss_history'].tolist()
            common.restore_rng(saved['rng_state'])
            print('Resumed at iteration %d' % start_iteration)
        teacher = normalization = None

        print('%s cluster matching begins, synthetic images = %d' % (get_time(), num_syn))
        for it in range(start_iteration + 1, args.Iteration + 1):
            net, _ = common.build_encoder(benchmark_model, im_size[0])
            net = net.to(args.device).eval().requires_grad_(False)
            optimizer_img.zero_grad(set_to_none=True)
            loss_dm_total, loss_icd_total = 0.0, 0.0
            for cluster in range(args.clusters):
                indices = members[cluster]
                indices = indices[torch.randperm(len(indices))[:args.batch_real]]
                real_images = (dataset_pixels(train_set, indices, mean_cpu, std_cpu).to(args.device) - mean_gpu) / std_gpu
                first = cluster * args.images_per_cluster
                synthetic_images = image_syn[first:first + args.images_per_cluster]
                seed = int(time.time() * 1000) % 100000
                real_images = DiffAugment(real_images, args.dsa_strategy, seed=seed, param=dsa_param)
                synthetic_images = DiffAugment(synthetic_images, args.dsa_strategy, seed=seed, param=dsa_param)
                with torch.no_grad():
                    output_real = net(real_images)
                output_syn = net(synthetic_images)
                loss_dm, loss_icd = matching_losses(output_real, output_syn)
                loss = loss_dm + args.lambda_div * loss_icd
                loss.backward()
                loss_dm_total += loss_dm.item()
                loss_icd_total += loss_icd.item()
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
            if it % args.save_png_every == 0 or it == args.Iteration:
                pixels_cpu = (image_syn.detach().cpu() * std_cpu + mean_cpu).clamp(0, 1)
                preview = Path(args.save_path) / ('vis_CCDM_%s_%s_%gpercent_exp%d_iter%d.png' % (args.dataset, args.model, args.percentage, exp, it))
                save_image(pixels_cpu[:256], preview, nrow=args.images_per_cluster)
            if it % args.save_pt_every == 0 or it == args.Iteration:
                data_cpu = image_syn.detach().cpu()
                pixels_cpu = (data_cpu * std_cpu + mean_cpu).clamp(0, 1)
                common.atomic_save(result_path, {
                    **run_config, 'data': data_cpu, 'pixels': pixels_cpu,
                    'normalization': [mean, std], 'iteration': it,
                    'membership': membership, 'initial_indices': initial_indices,
                    'synthetic_clusters': torch.arange(args.clusters).repeat_interleave(args.images_per_cluster),
                    'initialization': 'cluster_center_then_farthest_point_representatives',
                    'loss_history': torch.tensor(loss_history), 'rng_state': common.rng_state(),
                    'optimizer_state': optimizer_img.state_dict(),
                })
                print('Saved synthetic data to %s' % result_path, flush=True)

        net = output_real = output_syn = loss = loss_dm = loss_icd = None
        del image_syn, optimizer_img, initial_pixels
        if args.device == 'cuda':
            torch.cuda.empty_cache()
        saved = torch.load(result_path, map_location='cpu', weights_only=True)
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        output = Path(args.save_path) / 'evaluation' / ('exp_%03d_%s' % (exp, stamp))
        output.mkdir(parents=True, exist_ok=False)
        config = dict(vars(args), synthetic_checkpoint=str(result_path.resolve()), generation_experiment=exp, std_ddof=1)
        config['save_path'] = str(output.resolve())
        common.atomic_json(output / 'config.json', config)
        accuracies = []
        for run in range(args.runs):
            _, _, accuracy = evaluate_synset_SSL(run, args.model, saved['data'], args, output / ('seed_%03d' % (args.seed + run)), saved.get('normalization'))
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
