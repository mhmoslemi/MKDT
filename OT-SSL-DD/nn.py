"""SSL response distribution distillation (SSRD), with images-only output.

Place this file beside the supplied utils.py and networks.py. Neither those
files nor eval.py needs to be edited. Evaluation remains a separate operation.

Example:
    python ssl_response_distillation.py --dataset CIFAR10 --percentage 0.2

The default latest checkpoint is compatible with the fixed path in the supplied
eval.py: result_both/res_IIC_CIFAR10_ResNet18_0.2percent_final.pt. Its `data` tensor
uses utils.get_dataset normalization, which is also stored in `normalization`.
The IIC filename is a compatibility convention; checkpoint `method` is SSRD.

Method:
  * Sample complete, paired SSL minibatches from real and synthetic images.
  * Measure their SGD/momentum and BatchNorm responses at the SAME learner state.
  * Match equally weighted empirical response distributions by exact assignment
    OT (squared Euclidean W2), retaining every trainable parameter's gradient.
  * Differentiate matched synthetic responses with respect to the image pixels.
    Replay one response at a time, avoiding a bank of second-order GPU graphs.
  * Advance a pool of learners on the current detached synthetic images, with
    staggered initial ages and periodic resets to cover different training stages.

This is a finite-sample, alternating approximation to the formal occupancy
objective. Learner states are held fixed during each image update; we do NOT
differentiate through their previous training history. The finite empirical OT
estimate and this truncated state gradient are not a certificate of the theorem.
There is no teacher, synthetic label, or target embedding in the output.
"""

import argparse
import math
import os
import random
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torchvision.utils import save_image

from utils import DiffAugment, ParamDiffAug, get_dataset, get_network, get_time


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _images_from_batch(batch):
    return batch[0] if isinstance(batch, (tuple, list)) else batch


@contextmanager
def seeded_torch(seed):
    """Replay augmentation/model randomness without consuming the caller's RNG."""
    devices = list(range(torch.cuda.device_count()))
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(int(seed))
        if devices:
            torch.cuda.manual_seed_all(int(seed))
        yield


def make_diff_augmenter(strategy, param):
    def augment(images, seed):
        with seeded_torch(seed):
            # seed=-1 gives independent per-image draws, not Siamese batch draws.
            return DiffAugment(images, strategy, seed=-1, param=param)
    return augment


class ResponseBatchNorm(nn.Module):
    """BatchNorm with differentiable expressions for its running-stat updates.

    Native running-stat buffers are updated without an autograd graph. Matching
    only those detached buffers would miss their pixel derivatives. This local
    module retains the same batch normalization and running-stat equations, and
    separately records their differentiable next values during response capture.
    Only instantiated probe modules are replaced; networks.py is never changed.
    """

    def __init__(self, source):
        super().__init__()
        self.num_features = source.num_features
        self.eps = source.eps
        self.momentum = source.momentum
        self.affine = source.affine
        self.track_running_stats = source.track_running_stats
        self.register_parameter('weight', source.weight)
        self.register_parameter('bias', source.bias)
        for name in ('running_mean', 'running_var', 'num_batches_tracked'):
            value = getattr(source, name)
            self.register_buffer(name, None if value is None else value.detach().clone())
        self.training = source.training
        self.end_response()

    def begin_response(self):
        self.capture_response = True
        if self.track_running_stats:
            self.base_mean = self.running_mean.detach().clone()
            self.base_var = self.running_var.detach().clone()
            self.next_mean = self.base_mean
            self.next_var = self.base_var

    def end_response(self):
        self.capture_response = False
        self.base_mean = None
        self.base_var = None
        self.next_mean = None
        self.next_var = None

    def forward(self, images):
        dimensions = (0,) + tuple(range(2, images.ndim))
        shape = (1, -1) + (1,) * (images.ndim - 2)
        use_batch_stats = self.training or not self.track_running_stats
        if use_batch_stats:
            count = images.numel() // images.shape[1]
            if count <= 1:
                raise ValueError('BatchNorm requires more than one value per channel')
            mean = images.mean(dim=dimensions)
            variance = images.var(dim=dimensions, unbiased=False)
            if self.training and self.track_running_stats:
                with torch.no_grad():
                    self.num_batches_tracked.add_(1)
                factor = self.momentum
                if factor is None:
                    factor = 1.0 / float(self.num_batches_tracked.item())
                previous_mean = self.next_mean if self.capture_response else self.running_mean.detach().clone()
                previous_var = self.next_var if self.capture_response else self.running_var.detach().clone()
                next_mean = (1.0 - factor) * previous_mean + factor * mean
                # PyTorch uses biased variance for normalization, unbiased for running_var.
                next_var = (1.0 - factor) * previous_var + factor * variance * (count / (count - 1))
                if self.capture_response:
                    self.next_mean, self.next_var = next_mean, next_var
                with torch.no_grad():
                    self.running_mean.copy_(next_mean.detach())
                    self.running_var.copy_(next_var.detach())
        else:
            mean, variance = self.running_mean, self.running_var

        output = (images - mean.view(shape)) * torch.rsqrt(variance.view(shape) + self.eps)
        if self.affine:
            output = output * self.weight.view(shape) + self.bias.view(shape)
        return output


def prepare_probe_modules(module):
    """Keep second derivatives and running-stat responses local to this file."""
    for name, child in list(module.named_children()):
        if isinstance(child, nn.modules.batchnorm._BatchNorm):
            setattr(module, name, ResponseBatchNorm(child))
        else:
            prepare_probe_modules(child)
    if isinstance(module, (nn.ReLU, nn.LeakyReLU)):
        module.inplace = False


class SSLProbe(nn.Module):
    def __init__(self, backbone, feature_dim, projection_dim):
        super().__init__()
        self.backbone = backbone
        self.projector = nn.Sequential(
            nn.Linear(feature_dim, projection_dim),
            nn.ReLU(inplace=False),
            nn.Linear(projection_dim, projection_dim),
        )

    def forward(self, images):
        return self.projector(self.backbone.embed(images))


def make_probe(model_name, method, channel, num_classes, im_size, args, seed):
    with seeded_torch(seed):
        network = get_network(model_name, channel, num_classes, im_size, seed=seed)
        if isinstance(network, nn.DataParallel):
            network = network.module
        network = network.to(args.device)
        if not callable(getattr(network, 'embed', None)):
            raise ValueError('%s must provide an embed(images) method' % model_name)
        # Classifier parameters do not participate in SSL and must not enter OT.
        if hasattr(network, 'classifier'):
            network.classifier = nn.Identity()
        network.eval()
        with torch.no_grad():
            example = torch.zeros(2, channel, *im_size, device=args.device)
            feature_dim = network.embed(example).shape[1]
        prepare_probe_modules(network)
        model = SSLProbe(network, feature_dim, args.projection_dim).to(args.device)
        model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr_net,
                                momentum=args.momentum, weight_decay=args.weight_decay)
    return dict(model=model, optimizer=optimizer, name=model_name, method=method,
                seed=int(seed), age=0)


def simclr_loss(features_u, features_v, temperature):
    features = torch.cat((F.normalize(features_u, dim=1, eps=1e-8),
                          F.normalize(features_v, dim=1, eps=1e-8)))
    logits = features @ features.t() / temperature
    diagonal = torch.eye(len(features), device=features.device, dtype=torch.bool)
    logits = logits.masked_fill(diagonal, -torch.inf)
    targets = (torch.arange(len(features), device=features.device) + len(features_u)) % len(features)
    return F.cross_entropy(logits, targets)


def barlow_twins_loss(features_u, features_v, off_diagonal_weight, eps):
    def standardize(features):
        centered = features - features.mean(dim=0)
        # A variance floor also keeps second derivatives finite at constant outputs.
        std = features.var(dim=0, unbiased=False).clamp_min(eps ** 2).sqrt()
        return centered / (std + eps)
    first, second = standardize(features_u), standardize(features_v)
    correlation = first.t() @ second / len(first)
    diagonal = torch.diagonal(correlation)
    off_diagonal = correlation - torch.diag(diagonal)
    return (diagonal - 1).square().sum() + off_diagonal_weight * off_diagonal.square().sum()


def paired_ssl_loss(model, images, draw, augmenter, method, args):
    with seeded_torch(draw['model_seed']):
        first = augmenter(images, draw['view_u_seed'])
        second = augmenter(images, draw['view_v_seed'])
        # Separate view forwards match the supplied SSL training style. All pairs
        # still enter ONE complete batch loss, including every negative/correlation.
        first = model(first)
        second = model(second)
        if method == 'simclr':
            return simclr_loss(first, second, args.temperature)
        return barlow_twins_loss(first, second, args.barlow_lambda, args.barlow_eps)


def make_draw(count, batch_size, objective_rng, replacement):
    indices = objective_rng.choice(count, size=batch_size, replace=replacement)
    seeds = objective_rng.integers(0, 2 ** 31 - 1, size=3)
    return dict(indices=torch.from_numpy(indices.astype(np.int64)),
                view_u_seed=int(seeds[0]), view_v_seed=int(seeds[1]), model_seed=int(seeds[2]))


def real_batch(dataset, draw, device):
    images = [_images_from_batch(dataset[int(index)]) for index in draw['indices'].tolist()]
    return torch.stack(images).to(device)


def synthetic_batch(images, draw):
    return images.index_select(0, draw['indices'].to(images.device))


def snapshot_buffers(model):
    return {name: value.detach().clone() for name, value in model.named_buffers()}


def restore_buffers(model, buffers):
    with torch.no_grad():
        for name, value in model.named_buffers():
            value.copy_(buffers[name])
    for module in model.modules():
        if isinstance(module, ResponseBatchNorm):
            module.end_response()


def batch_response(probe, images, draw, augmenter, args, buffers, create_graph):
    """An isometric representation of conditional SGD/momentum state differences.

    At the SAME current state, weight decay and the previous momentum contribute
    common translations, which cancel in transport distances. For plain SGD the
    parameter difference is -lr * delta_gradient. With momentum, the next momentum
    difference is delta_gradient as well; sqrt(lr**2 + 1) preserves the combined
    parameter/momentum squared distance exactly. This is not gradient normalization.
    Running means/variances add their actual differentiable state increments.
    BN integer counters have identical increments for every batch and zero cost.
    """
    model = probe['model']
    restore_buffers(model, buffers)
    model.train()
    for module in model.modules():
        if isinstance(module, ResponseBatchNorm):
            module.begin_response()
    loss = paired_ssl_loss(model, images, draw, augmenter, probe['method'], args)
    if not torch.isfinite(loss):
        raise RuntimeError('Non-finite SSL loss while constructing a batch response')
    parameters = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
    gradients = torch.autograd.grad(loss, parameters, create_graph=create_graph,
                                    retain_graph=create_graph, allow_unused=False)
    scale = math.sqrt(args.lr_net ** 2 + (1.0 if args.momentum > 0 else 0.0))
    pieces = [gradient.reshape(-1) * scale for gradient in gradients]
    for module in model.modules():
        if isinstance(module, ResponseBatchNorm) and module.track_running_stats:
            pieces.extend(((module.next_mean - module.base_mean).reshape(-1),
                           (module.next_var - module.base_var).reshape(-1)))
    response = torch.cat(pieces)
    if not create_graph:
        response = response.detach()
    if not torch.isfinite(response).all():
        raise RuntimeError('Non-finite SSL state response')
    return response, float(loss.detach().item())


def response_cost_matrix(real_responses, syn_responses, chunk_size=65536):
    """CPU float64 accumulation without a response_count^2 x parameter tensor."""
    cost = torch.zeros(len(real_responses), len(syn_responses), dtype=torch.float64)
    for start in range(0, real_responses.shape[1], chunk_size):
        real = real_responses[:, start:start + chunk_size].double()
        synthetic = syn_responses[:, start:start + chunk_size].double()
        cost += (real.square().sum(dim=1, keepdim=True)
                 + synthetic.square().sum(dim=1).unsqueeze(0) - 2 * real @ synthetic.t())
    return cost.clamp_min_(0)


def match_response_distributions(probe, dataset, image_syn, augmenter, args, objective_rng, gradient_weight):
    """Exact empirical W2 assignment, then memory-bounded differentiable replay.

    Equally weighted response clouds have an optimal permutation coupling.
    Holding its exact minimizing assignment fixed gives an OT subgradient almost
    everywhere. Replaying the same draws recovers each synthetic response graph.
    Only current-state pixel derivatives are taken; the real targets and probe
    history are detached (the alternating occupancy approximation above).
    """
    model = probe['model']
    buffers = snapshot_buffers(model)
    replacement = args.batch_sampling == 'with_replacement'
    real_draws = [make_draw(len(dataset), args.effective_batch_size, objective_rng, replacement)
                  for _ in range(args.response_batches)]
    syn_draws = [make_draw(len(image_syn), args.effective_batch_size, objective_rng, replacement)
                 for _ in range(args.response_batches)]
    real_responses, syn_responses = None, None
    real_ssl, syn_ssl = [], []

    try:
        # No second-order graphs are retained in either cloud during assignment.
        for index, draw in enumerate(real_draws):
            images = real_batch(dataset, draw, args.device)
            response, ssl = batch_response(probe, images, draw, augmenter, args, buffers, False)
            if real_responses is None:
                real_responses = torch.empty(args.response_batches, response.numel(), dtype=torch.float32)
            real_responses[index].copy_(response.cpu())
            real_ssl.append(ssl)
            del response, images
        for index, draw in enumerate(syn_draws):
            images = synthetic_batch(image_syn.detach(), draw)
            response, ssl = batch_response(probe, images, draw, augmenter, args, buffers, False)
            if syn_responses is None:
                syn_responses = torch.empty_like(real_responses)
            syn_responses[index].copy_(response.cpu())
            syn_ssl.append(ssl)
            del response, images

        cost = response_cost_matrix(real_responses, syn_responses)
        if not torch.isfinite(cost).all():
            raise RuntimeError('Non-finite empirical response transport cost')
        real_indices, syn_indices = linear_sum_assignment(cost.numpy())
        target_for_syn = np.empty(args.response_batches, dtype=np.int64)
        target_for_syn[syn_indices] = real_indices
        transport_value = float(cost[real_indices, syn_indices].mean().item())
        real_energy = float(real_responses.double().square().sum(dim=1).mean().item())
        del syn_responses, cost

        replay_value = 0.0
        for index, draw in enumerate(syn_draws):
            images = synthetic_batch(image_syn, draw)
            response, _ = batch_response(probe, images, draw, augmenter, args, buffers, True)
            target = real_responses[int(target_for_syn[index])].to(args.device)
            response_loss = (response - target).square().sum() / args.response_batches
            replay_value += float(response_loss.detach().item())
            pixel_gradient, = torch.autograd.grad(response_loss * gradient_weight, image_syn)
            if not torch.isfinite(pixel_gradient).all():
                raise RuntimeError('Non-finite second-order synthetic pixel gradient')
            with torch.no_grad():
                if image_syn.grad is None:
                    image_syn.grad = pixel_gradient.detach()
                else:
                    image_syn.grad.add_(pixel_gradient.detach())
            del images, response, target, response_loss, pixel_gradient
        return dict(loss=replay_value, assignment_loss=transport_value,
                    real_ssl=float(np.mean(real_ssl)), syn_ssl=float(np.mean(syn_ssl)),
                    real_response_energy=real_energy, probe_age=int(probe['age']))
    finally:
        # Counterfactual real/synthetic batches must never advance the probe state.
        restore_buffers(model, buffers)


def advance_probe(probe, image_syn, augmenter, args, objective_rng, steps):
    """Actual synthetic-only learner steps; no real data or class labels here."""
    losses = []
    model, optimizer = probe['model'], probe['optimizer']
    model.train()
    for module in model.modules():
        if isinstance(module, ResponseBatchNorm):
            module.end_response()
    for _ in range(steps):
        draw = make_draw(len(image_syn), args.effective_batch_size, objective_rng,
                         args.batch_sampling == 'with_replacement')
        images = synthetic_batch(image_syn.detach(), draw)
        optimizer.zero_grad(set_to_none=True)
        loss = paired_ssl_loss(model, images, draw, augmenter, probe['method'], args)
        if not torch.isfinite(loss):
            raise RuntimeError('Non-finite SSL loss while advancing a synthetic learner')
        loss.backward()
        if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all()
               for parameter in model.parameters()):
            raise RuntimeError('Non-finite synthetic learner gradient')
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        probe['age'] += 1
        losses.append(float(loss.detach().item()))
    return float(np.mean(losses)) if losses else 0.0


@torch.no_grad()
def deepcluster_initialize(dst_train, num_syn, device, seed, channel, num_classes, im_size,
                           model_name='ConvNet', batch_size=256, iterations=4):
    """Initialize with real representatives of unlabeled random-feature clusters."""
    print('Initializing %d images with random-feature K-Means on %s...' % (num_syn, model_name), flush=True)
    with seeded_torch(seed):
        network = get_network(model_name, channel, num_classes, im_size, seed=seed)
        if isinstance(network, nn.DataParallel):
            network = network.module
        network = network.to(device).eval()
        features = []
        loader = torch.utils.data.DataLoader(dst_train, batch_size=batch_size, shuffle=False, num_workers=0)
        for batch in loader:
            images = _images_from_batch(batch).to(device)
            features.append(network.embed(images).flatten(1).cpu())
        features = torch.cat(features, dim=0)
    del network, loader
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(features), generator=generator)[:num_syn]
    centers = features[indices].clone()
    assignments = torch.empty(len(features), dtype=torch.long)

    for _ in range(iterations):
        for start in range(0, len(features), 2048):
            distances = torch.cdist(features[start:start + 2048], centers)
            assignments[start:start + len(distances)] = distances.argmin(dim=1)
        sums = torch.zeros_like(centers)
        sums.index_add_(0, assignments, features)
        counts = torch.bincount(assignments, minlength=num_syn)
        populated = counts > 0
        new_centers = centers.clone()
        new_centers[populated] = sums[populated] / counts[populated, None]
        converged = torch.allclose(new_centers, centers, atol=1e-4)
        centers = new_centers
        if converged:
            break

    # Select a nearest member of each cluster, avoiding duplicate representatives.
    for start in range(0, len(features), 2048):
        distances = torch.cdist(features[start:start + 2048], centers)
        assignments[start:start + len(distances)] = distances.argmin(dim=1)
    chosen, used = [], set()
    empty = []
    for cluster in range(num_syn):
        members = torch.nonzero(assignments == cluster, as_tuple=False).flatten()
        if len(members):
            distances = (features[members] - centers[cluster]).square().sum(dim=1)
            index = int(members[distances.argmin()].item())
            chosen.append(index)
            used.add(index)
        else:
            empty.append(cluster)
    if empty:
        for index in torch.randperm(len(dst_train), generator=generator).tolist():
            if index not in used:
                chosen.append(index)
                used.add(index)
                if len(chosen) == num_syn:
                    break
    return torch.stack([_images_from_batch(dst_train[index]) for index in chosen]).to(device)


def save_synthetic(iteration, image_syn, args, mean, std, histories, random_models, methods, final=False):
    """Keep visualization snapshots, numbered .pt files, and eval.py's latest path."""
    images = image_syn.detach().cpu().contiguous()
    if not torch.isfinite(images).all():
        raise RuntimeError('Refusing to save non-finite synthetic images')
    normalization = [[float(value) for value in mean], [float(value) for value in std]]
    mean_tensor = torch.tensor(normalization[0]).view(1, -1, 1, 1)
    std_tensor = torch.tensor(normalization[1]).view(1, -1, 1, 1)
    visible = (images * std_tensor + mean_tensor).clamp(0, 1)
    grid_path = os.path.join(args.save_path, 'vis_SSRD_%s_%s_%gpercent_iter%d.png' %
                             (args.dataset, args.model, args.percentage, iteration))
    save_image(visible, grid_path, nrow=int(np.ceil(np.sqrt(len(images)))))

    saved = dict(
        data=images, method=args.method, dataset=args.dataset, percentage=float(args.percentage),
        normalization=normalization, iteration=int(iteration), complete=bool(final),
        seed=int(args.seed), num_syn=int(len(images)), config=dict(vars(args)),
        response_params=dict(
            distance='empirical_squared_Wasserstein_2', solver='exact_equal_weight_assignment',
            response_batches=args.response_batches, batch_size=args.effective_batch_size,
            sampling=args.batch_sampling, parameter_metric='SGD_parameter_and_momentum_state',
            batchnorm_metric='differentiable_running_mean_and_variance_increments',
            models=random_models, methods=methods, method_reduction='uniform_sampled_recipes',
            optimizer='SGD', lr=args.lr_net, momentum=args.momentum, weight_decay=args.weight_decay,
            augmentation=args.distill_aug_strategy, augmentation_mode=args.distill_aug_mode,
            state_distribution='synthetic_trained_pool_with_periodic_resets',
            occupancy_gradient='truncated_at_current_probe_state',
            teacher_required=False, labels_used=False,
        ),
    )
    for name, values in histories.items():
        saved[name] = torch.tensor(values, dtype=torch.float64)

    base = 'res_IIC_%s_%s_%gpercent' % (args.dataset, args.model, args.percentage)
    snapshot_path = os.path.join(args.save_path, base + '_.pt')
    result_path = os.path.join(args.save_path, base + '_final.pt')
    torch.save(saved, snapshot_path)
    temporary = result_path + '.tmp'
    torch.save(saved, temporary)
    os.replace(temporary, result_path)
    print('Saved synthetic data to %s (iteration %d); visualization: %s' %
          (result_path, iteration, grid_path), flush=True)


def main():
    parser = argparse.ArgumentParser(description='SSL response distribution distillation (SSRD)')

    # Data: labels returned by get_dataset are always ignored.
    parser.add_argument('--dataset', choices=['CIFAR10', 'CIFAR100', 'TinyImageNet'], default='CIFAR10')
    parser.add_argument('--percentage', type=float, default=0.2)
    parser.add_argument('--data_path', default='/home/mmoslem3/scratch/data')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=0)

    # Pixel optimization and complete-batch response distributions.
    parser.add_argument('--Iteration', type=int, default=2000)
    parser.add_argument('--lr_img', type=float, default=0.01)
    parser.add_argument('--lr_img_min', type=float, default=0.0005)
    parser.add_argument('--response_batches', type=int, default=4, help='Independent batch responses per empirical distribution; >= 2')
    parser.add_argument('--batch_train', '--batch_real', dest='batch_train', type=int, default=512,
                        help='Requested batch size shared by real responses, synthetic responses and probe training')
    parser.add_argument('--batch_sampling', choices=['without_replacement', 'with_replacement'], default='without_replacement')

    # Evolving learner states. These settings affect generation, never eval.py.
    parser.add_argument('--random_models', default='ConvNetBN', help='Comma-separated model names accepted by supplied get_network')
    parser.add_argument('--ssl_method', choices=['simclr', 'barlowtwins', 'both'], default='simclr')
    parser.add_argument('--pool_size', type=int, default=6, help='Learners per architecture/SSL-method combination')
    parser.add_argument('--num_random_networks', type=int, default=4, help='Learner states sampled per image update')
    parser.add_argument('--probe_max_steps', type=int, default=100, help='Reset a learner after this many synthetic SGD steps')
    parser.add_argument('--encoder_train_steps', type=int, default=1, help='Actual synthetic SGD steps after a sampled state is matched')
    parser.add_argument('--lr_net', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--projection_dim', type=int, default=128)
    parser.add_argument('--temperature', type=float, default=0.2)
    parser.add_argument('--barlow_lambda', type=float, default=0.005)
    parser.add_argument('--barlow_eps', type=float, default=1e-5)
    parser.add_argument('--distill_aug_strategy', default='color_crop_cutout_flip_scale_rotate')
    parser.add_argument('--distill_aug_mode', choices=['S', 'M'], default='M')

    # Initialization and output follow final.py; evaluation stays external.
    parser.add_argument('--init', choices=['kmeans', 'random'], default='kmeans')
    parser.add_argument('--init_batch_size', type=int, default=256)
    parser.add_argument('--model', default='ResNet18', help='Model label in the existing evaluator-compatible filename')
    parser.add_argument('--save_path', default='agents')
    parser.add_argument('--save_every', type=int, default=50)
    parser.add_argument('--log_every', type=int, default=25)
    args = parser.parse_args()

    random_models = [name.strip() for name in args.random_models.split(',') if name.strip()]
    methods = ['simclr', 'barlowtwins'] if args.ssl_method == 'both' else [args.ssl_method]
    positive = (args.Iteration, args.pool_size, args.num_random_networks, args.probe_max_steps,
                args.encoder_train_steps, args.projection_dim, args.init_batch_size, args.save_every, args.log_every)
    if not random_models or min(positive) < 1 or min(args.response_batches, args.batch_train) < 2:
        parser.error('Models must be nonempty; counts must be positive; response_batches and batch_train must be >= 2')
    if not 0 < args.percentage <= 100 or args.lr_img <= 0 or not 0 <= args.lr_img_min <= args.lr_img:
        parser.error('Invalid percentage or image learning rates')
    if args.lr_net <= 0 or not 0 <= args.momentum < 1 or args.weight_decay < 0:
        parser.error('Invalid probe SGD settings')
    if args.temperature <= 0 or args.barlow_lambda < 0 or args.barlow_eps <= 0:
        parser.error('Invalid SSL loss settings')
    if args.num_random_networks > args.pool_size * len(random_models) * len(methods):
        parser.error('num_random_networks cannot exceed the total learner pool')
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA requested but unavailable')
    args.method = 'SSRD'
    set_random_seed(args.seed)
    # Assignment/replay should use the same numerical response, not TF32 rounding.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(args.data_path, exist_ok=True)
    channel, im_size, num_classes, _, mean, std, dst_train, dst_test, testloader = get_dataset(args.dataset, args.data_path)
    del dst_test, testloader
    num_syn = int(len(dst_train) * args.percentage / 100)
    if num_syn < 2:
        raise ValueError('SSL requires at least two synthetic images')
    args.effective_batch_size = args.batch_train
    if args.batch_sampling == 'without_replacement':
        args.effective_batch_size = min(args.batch_train, num_syn, len(dst_train))
    if args.effective_batch_size != args.batch_train:
        print('Using batch size %d for BOTH real and synthetic updates: only %d distinct synthetic images exist.' %
              (args.effective_batch_size, num_syn), flush=True)

    objective_rng = np.random.default_rng(args.seed + 10_000)
    if args.init == 'kmeans':
        image_syn_init = deepcluster_initialize(dst_train, num_syn, args.device, args.seed,
                                                channel, num_classes, im_size, random_models[0], args.init_batch_size)
    else:
        indices = objective_rng.choice(len(dst_train), size=num_syn, replace=False)
        image_syn_init = torch.stack([_images_from_batch(dst_train[int(index)]) for index in indices]).to(args.device)
    image_syn = image_syn_init.detach().clone().requires_grad_(True)
    image_syn_initial = image_syn.detach().clone()
    del image_syn_init
    diag_mean = torch.tensor(mean, device=args.device).view(1, channel, 1, 1)
    diag_std = torch.tensor(std, device=args.device).view(1, channel, 1, 1)
    pixel_min, pixel_max = -diag_mean / diag_std, (1 - diag_mean) / diag_std

    def to_uint8(images):
        return ((images * diag_std + diag_mean) * 255 + 0.5).clamp(0, 255).to(torch.uint8)

    initial_uint8 = to_uint8(image_syn_initial)
    distill_dsa_param = ParamDiffAug()
    distill_dsa_param.aug_mode = args.distill_aug_mode
    augmenter = make_diff_augmenter(args.distill_aug_strategy, distill_dsa_param)
    optimizer_img = torch.optim.Adam([image_syn], lr=args.lr_img)
    scheduler_img = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_img, T_max=args.Iteration, eta_min=args.lr_img_min)
    histories = {name: [] for name in (
        'loss_history', 'total_loss_history', 'assignment_loss_history', 'real_ssl_history',
        'synthetic_ssl_history', 'encoder_ssl_history', 'real_response_energy_history', 'probe_age_history')}

    print('Hyper-parameters:\n', args.__dict__, flush=True)
    print('Objective: empirical W2 of whole-batch SSL state responses; exact assignment, no labels.', flush=True)
    print('Occupancy approximation: evolving synthetic-only learner pool; previous learner steps are detached.', flush=True)
    print('Evaluation is external and unchanged. Saved images use the original dataset normalization.', flush=True)
    save_synthetic(0, image_syn, args, mean, std, histories, random_models, methods)

    pool = []
    for model_name in random_models:
        for method in methods:
            for member in range(args.pool_size):
                seed = int(objective_rng.integers(0, 2 ** 31 - 1))
                probe = make_probe(model_name, method, channel, num_classes, im_size, args, seed)
                # An initial learner and staggered later states are present immediately.
                warmup = member * args.probe_max_steps // args.pool_size
                advance_probe(probe, image_syn, augmenter, args, objective_rng, warmup)
                pool.append(probe)
                print('Prepared %s/%s learner, seed=%d, age=%d' % (model_name, method, seed, probe['age']), flush=True)

    print('%s distillation begins' % get_time(), flush=True)
    for iteration in range(1, args.Iteration + 1):
        optimizer_img.zero_grad(set_to_none=True)
        selected = objective_rng.choice(len(pool), size=args.num_random_networks, replace=False).tolist()
        sampled = []
        for index in selected:
            probe = pool[index]
            if probe['age'] >= args.probe_max_steps:
                name, method = probe['name'], probe['method']
                pool[index] = None
                del probe
                seed = int(objective_rng.integers(0, 2 ** 31 - 1))
                probe = make_probe(name, method, channel, num_classes, im_size, args, seed)
                pool[index] = probe
            sampled.append(match_response_distributions(
                probe, dst_train, image_syn, augmenter, args, objective_rng,
                gradient_weight=1.0 / args.num_random_networks))

        if image_syn.grad is None or not torch.isfinite(image_syn.grad).all():
            raise RuntimeError('Missing or non-finite synthetic image gradients')
        log_iteration = iteration == 1 or iteration % args.log_every == 0 or iteration == args.Iteration
        if log_iteration:
            pixel_grad_rms = float(image_syn.grad.square().mean().sqrt().item())
            step_lr = optimizer_img.param_groups[0]['lr']
        optimizer_img.step()
        scheduler_img.step()
        with torch.no_grad():
            image_syn.copy_(torch.maximum(torch.minimum(image_syn, pixel_max), pixel_min))
            if not torch.isfinite(image_syn).all():
                raise RuntimeError('Non-finite synthetic pixels after image update')

        encoder_losses = []
        for index in selected:
            steps = min(args.encoder_train_steps, args.probe_max_steps - pool[index]['age'])
            encoder_losses.append(advance_probe(pool[index], image_syn, augmenter, args, objective_rng, steps))
        averaged = {key: float(np.mean([item[key] for item in sampled])) for key in sampled[0]}
        histories['loss_history'].append(averaged['loss'])
        histories['total_loss_history'].append(averaged['loss'])
        histories['assignment_loss_history'].append(averaged['assignment_loss'])
        histories['real_ssl_history'].append(averaged['real_ssl'])
        histories['synthetic_ssl_history'].append(averaged['syn_ssl'])
        histories['encoder_ssl_history'].append(float(np.mean(encoder_losses)))
        histories['real_response_energy_history'].append(averaged['real_response_energy'])
        histories['probe_age_history'].append(averaged['probe_age'])

        if log_iteration:
            with torch.no_grad():
                changed = float((to_uint8(image_syn) != initial_uint8).float().mean().item() * 100)
                drift = float(((image_syn - image_syn_initial) * diag_std * 255).square().mean().sqrt().item())
            print('%s iter = %05d, response_W2_squared = %.6e, assignment_cost = %.6e, real_SSL = %.6f, synthetic_SSL = %.6f, probe_age = %.1f, pixel_grad_RMS = %.3e, pixel drift RMS = %.5f/255, PNG values changed = %.5f%%, lr = %.3e' %
                  (get_time(), iteration, averaged['loss'], averaged['assignment_loss'],
                   averaged['real_ssl'], averaged['syn_ssl'], averaged['probe_age'],
                   pixel_grad_rms, drift, changed, step_lr), flush=True)
        if iteration % args.save_every == 0 or iteration == args.Iteration:
            save_synthetic(iteration, image_syn, args, mean, std, histories, random_models, methods,
                           final=iteration == args.Iteration)


if __name__ == '__main__':
    main()
