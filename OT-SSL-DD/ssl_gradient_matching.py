"""One-step SSL meta-learning (legacy filename retained for existing launch setups).

For a learner state (theta, momentum), take a virtual SGD step on synthetic S:
    theta_plus(S) = SGD(theta, momentum, mean_a grad_theta L_SSL(S, a)).
Optimize J(S) = mean_q L_SSL(real_query_q; theta_plus(S)). No gradient-distance
or covariance proxy enters J. This is a one-step bilevel objective, following the
dataset-distillation formulation: https://arxiv.org/abs/1811.10959, adapted to SSL.

With the starting state fixed, its exact image derivative is
    grad_S J = -eta * (d g_S / d S)^T * grad_theta_plus L_query.
We compute the query gradient at the UPDATED weights, then replay each synthetic
view for this mixed-derivative product. This bounds memory without dropping the
derivative through SGD. Momentum and weight decay are included in the forward
update; they have zero image derivative at the fixed starting state.

A pixel candidate must improve J and pass a single check on disjoint real query
images, independently sampled synthetic batches, and fresh augmentations. This
is an optimization guard, not a downstream-accuracy guarantee. No test images or
class labels enter distillation. Accepted images drive the learners' next steps;
history before the current step is detached (truncated meta-learning).
"""

from contextlib import contextmanager
from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call

from utils import DiffAugment, get_network


def cuda_rng_devices():
    return list(range(torch.cuda.device_count()))


def draw_view_seeds(rng, count):
    # Two augmentation seeds and one seed for stochastic network layers.
    return rng.integers(0, 2**31 - 1, size=(count, 3)).tolist()


def ssl_loss(z1, z2, args):
    """Same projection-space SSL objective as evaluate_synset_SSL in utils.py."""
    method = args.ssl_method.lower()
    if method == 'simclr':
        z = torch.cat((F.normalize(z1, dim=1), F.normalize(z2, dim=1)), dim=0)
        logits = z @ z.T / args.temperature
        diagonal = torch.eye(len(z), dtype=torch.bool, device=z.device)
        logits = logits.masked_fill(diagonal, -1e9)
        targets = (torch.arange(len(z), device=z.device) + len(z1)) % len(z)
        return F.cross_entropy(logits, targets)
    if method in ('barlowtwins', 'barlow_twins'):
        z1 = (z1 - z1.mean(0)) / (z1.std(0, unbiased=False) + 1e-5)
        z2 = (z2 - z2.mean(0)) / (z2.std(0, unbiased=False) + 1e-5)
        correlation = z1.T @ z2 / len(z1)
        diagonal = torch.diagonal(correlation)
        return (diagonal - 1).square().sum() + args.barlow_lambda * (
            correlation - torch.diag(diagonal)
        ).square().sum()
    raise ValueError(f'Unsupported SSL method: {args.ssl_method}')


class _SSLModel(nn.Module):
    """Expose embed + projector through forward for stateless updated weights."""

    def __init__(self, encoder, projector):
        super().__init__()
        self.encoder = encoder
        self.projector = projector

    def forward(self, images):
        return self.projector(self.encoder.embed(images))


class SSLReference:
    """An encoder plus the evaluation architecture's two-layer SSL projector."""

    def __init__(self, args, channel, num_classes, im_size, seed):
        with torch.random.fork_rng(devices=cuda_rng_devices()):
            encoder = get_network(args.model, channel, num_classes, im_size, seed=seed)
            self.encoder = encoder.module if isinstance(encoder, nn.DataParallel) else encoder
            if not hasattr(self.encoder, 'embed'):
                raise ValueError(f'{args.model} does not provide embed() for SSL')
            self.encoder.eval()
            for module in self.encoder.modules():
                if isinstance(module, nn.ReLU):
                    module.inplace = False
            # Identify parameters used by embed(); classification heads are unused
            # by SSL and must not enter the gradient vector or reference optimizer.
            dummy = torch.zeros((2, channel, *im_size), device=args.device)
            features = self.encoder.embed(dummy)
            feature_dim = features.shape[1]
            encoder_parameters = tuple(self.encoder.parameters())
            used = torch.autograd.grad(features.sum(), encoder_parameters, allow_unused=True)
            for parameter, gradient in zip(encoder_parameters, used):
                parameter.requires_grad_(gradient is not None)
            torch.manual_seed(seed + 1)
            self.projector = nn.Sequential(
                nn.Linear(feature_dim, args.projection_dim),
                nn.ReLU(inplace=False),
                nn.Linear(args.projection_dim, args.projection_dim),
            ).to(args.device)
        self.model = _SSLModel(self.encoder, self.projector)
        self.named_parameters = tuple((name, p) for name, p in self.model.named_parameters() if p.requires_grad)
        self.parameters = tuple(p for _, p in self.named_parameters)
        self.optimizer = torch.optim.SGD(
            self.parameters, lr=args.lr_net, momentum=0.9, weight_decay=0.0005
        )
        self.steps = 0

    @contextmanager
    def matching_mode(self):
        # Use training-mode batch statistics, but do not mutate running buffers
        # between real/synthetic forwards or line-search candidates. Dropout is
        # replayed from its stored seed. The default ConvNet uses GroupNorm.
        modules = list(self.encoder.modules()) + list(self.projector.modules())
        training_flags = [module.training for module in modules]
        batch_norms = [module for module in modules if isinstance(module, nn.modules.batchnorm._BatchNorm)]
        tracking_flags = [module.track_running_stats for module in batch_norms]
        self.encoder.train()
        self.projector.train()
        for module in batch_norms:
            module.track_running_stats = False
        try:
            yield
        finally:
            for module, training in zip(modules, training_flags):
                module.training = training
            for module, tracking in zip(batch_norms, tracking_flags):
                module.track_running_stats = tracking

    def advance(self, gradients, args):
        """Commit one SGD step; no graph is retained across distillation iterations."""
        if not all(torch.isfinite(gradient).all().item() for gradient in gradients):
            raise FloatingPointError('Non-finite SSL gradient; learner update aborted')
        self.optimizer.zero_grad(set_to_none=True)
        for parameter, gradient in zip(self.parameters, gradients):
            parameter.grad = gradient.detach().clone()
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.steps += 1
        if self.steps == args.reference_max_steps // 2 + 1:
            for group in self.optimizer.param_groups:
                group['lr'] *= 0.1


def parameter_gradients(reference, images, seeds, args, create_graph=False):
    # enable_grad is required even during candidate evaluation: we still need
    # first derivatives with respect to network parameters in that case.
    if not create_graph:
        images = images.detach()
    with torch.enable_grad(), reference.matching_mode(), torch.random.fork_rng(devices=cuda_rng_devices()):
        torch.manual_seed(seeds[0])
        view1 = DiffAugment(images, args.ssl_aug_strategy, seed=-1, param=args.dsa_param)
        torch.manual_seed(seeds[1])
        view2 = DiffAugment(images, args.ssl_aug_strategy, seed=-1, param=args.dsa_param)
        # seed=-1 above keeps independently sampled transforms for each image,
        # as in evaluation. Replaying the outer seed reproduces the whole batch.
        torch.manual_seed(seeds[2])
        z1 = reference.projector(reference.encoder.embed(view1))
        z2 = reference.projector(reference.encoder.embed(view2))
        loss = ssl_loss(z1, z2, args)
        gradients = torch.autograd.grad(loss, reference.parameters, create_graph=create_graph)
    return gradients, loss.detach()


def mean_parameter_gradients(reference, images, view_pairs, args):
    averaged = None
    loss_avg = torch.zeros((), device=images.device)
    for seeds in view_pairs:
        gradients, loss = parameter_gradients(reference, images, seeds, args)
        if averaged is None:
            averaged = [gradient.detach().clone() for gradient in gradients]
        else:
            for total, gradient in zip(averaged, gradients):
                total.add_(gradient.detach())
        loss_avg.add_(loss / len(view_pairs))
    for gradient in averaged:
        gradient.div_(len(view_pairs))
    return tuple(averaged), loss_avg


@dataclass
class MetaTask:
    reference: SSLReference
    indices: torch.Tensor
    support_views: list
    query_images: torch.Tensor
    query_views: list


def virtual_sgd_weights(reference, gradients, requires_grad=False):
    """The exact next weights of this learner's SGD, without mutating its state."""
    group = reference.optimizer.param_groups[0]
    updated = {}
    for (name, parameter), gradient in zip(reference.named_parameters, gradients):
        direction = gradient.detach() + group['weight_decay'] * parameter.detach()
        momentum = reference.optimizer.state.get(parameter, {}).get('momentum_buffer')
        if momentum is not None:
            direction = direction + group['momentum'] * momentum.detach()
        updated[name] = (parameter.detach() - group['lr'] * direction).requires_grad_(requires_grad)
    return updated


def query_ssl_loss(task, weights, seeds, args):
    reference = task.reference
    # Clone buffers so even a custom module cannot overwrite the learner's state.
    buffers = {name: buffer.detach().clone() for name, buffer in reference.model.named_buffers()}
    with reference.matching_mode(), torch.random.fork_rng(devices=cuda_rng_devices()):
        torch.manual_seed(seeds[0])
        view1 = DiffAugment(task.query_images, args.ssl_aug_strategy, seed=-1, param=args.dsa_param)
        torch.manual_seed(seeds[1])
        view2 = DiffAugment(task.query_images, args.ssl_aug_strategy, seed=-1, param=args.dsa_param)
        torch.manual_seed(seeds[2])
        z1 = functional_call(reference.model, (weights, buffers), (view1,))
        z2 = functional_call(reference.model, (weights, buffers), (view2,))
        return ssl_loss(z1, z2, args)


def mean_query_loss(task, weights, args):
    with torch.no_grad():
        return sum(query_ssl_loss(task, weights, seeds, args).item() for seeds in task.query_views) / len(task.query_views)


def meta_gradient_and_backward(task, images, args):
    """Exact one-step meta-gradient via updated-weight query gradients and replay."""
    reference = task.reference
    support_gradients, support_ssl = mean_parameter_gradients(
        reference, images.detach().index_select(0, task.indices), task.support_views, args
    )
    weights = virtual_sgd_weights(reference, support_gradients, requires_grad=True)
    query_gradients = [torch.zeros_like(weight) for weight in weights.values()]
    loss_avg = 0.0
    with torch.enable_grad():
        for seeds in task.query_views:
            loss = query_ssl_loss(task, weights, seeds, args)
            gradients = torch.autograd.grad(loss, tuple(weights.values()))
            loss_avg += loss.detach().item() / len(task.query_views)
            for total, gradient in zip(query_gradients, gradients):
                total.add_(gradient.detach() / len(task.query_views))
    if not math.isfinite(loss_avg):
        raise FloatingPointError('Non-finite real-query SSL loss after the synthetic SGD step')
    lr = reference.optimizer.param_groups[0]['lr']
    upstream = tuple(-lr * gradient / (args.num_reference_nets * len(task.support_views)) for gradient in query_gradients)
    for seeds in task.support_views:
        gradients, _ = parameter_gradients(
            reference, images.index_select(0, task.indices), seeds, args, create_graph=True
        )
        differentiable = [(gradient, coefficient) for gradient, coefficient in zip(gradients, upstream) if gradient.requires_grad]
        if differentiable:
            image_gradient, = torch.autograd.grad(
                tuple(pair[0] for pair in differentiable), images,
                grad_outputs=tuple(pair[1] for pair in differentiable), allow_unused=True,
            )
            if image_gradient is not None:
                if images.grad is None:
                    images.grad = image_gradient.detach()
                else:
                    images.grad.add_(image_gradient.detach())
    before_sgd = mean_query_loss(task, {name: p.detach() for name, p in reference.named_parameters}, args)
    return {
        'loss': loss_avg, 'query_before_sgd': before_sgd,
        'synthetic_ssl': support_ssl.item(),
    }


def sampled_meta_loss(images, tasks, args, reject_above=math.inf):
    total = 0.0
    for task in tasks:
        gradients, _ = mean_parameter_gradients(
            task.reference, images.index_select(0, task.indices), task.support_views, args
        )
        weights = virtual_sgd_weights(task.reference, gradients)
        total += mean_query_loss(task, weights, args) / len(tasks)
        if not math.isfinite(total) or total > reject_above:
            return math.inf
    return total


@dataclass
class MetaStepResult:
    loss_after: float
    step_size: float = 0.0
    trials: int = 0
    accepted: bool = False
    reason: str = 'no_descent'
    next_step: float = None
    guard_before: float = None
    guard_after: float = None


@torch.no_grad()
def meta_backtracking_step(images, tasks, guard_tasks, args, lower, upper, pixel_std, loss_before, initial_step):
    """Find a descent candidate, then check it ONCE on independent samples.

    The guard never contributes to the image gradient or backtracking search.
    A failed guard discards the candidate; subsequent iterations draw new data.
    """
    result = MetaStepResult(loss_after=loss_before)
    gradient = images.grad.detach()
    original = images.detach()
    gradient_pixel_rms = (gradient.double() * pixel_std * 255).square().mean().sqrt().item()
    if not math.isfinite(gradient_pixel_rms) or gradient_pixel_rms == 0:
        return result
    step_limit = min(
        args.lr_img / max(gradient_pixel_rms, 1e-12), math.sqrt(torch.finfo(images.dtype).max)
    )
    step = step_limit if initial_step is None else min(initial_step, step_limit)
    result.next_step = step
    for trial in range(1, 21):
        result.trials = trial
        candidate = torch.maximum(torch.minimum(original - step * gradient, upper), lower)
        slope = (gradient * (candidate - original)).sum(dtype=torch.float64).item()
        if not math.isfinite(slope):
            step *= 0.5
            result.next_step = step
            continue
        if slope >= 0:
            return result
        bound = loss_before + 1e-4 * slope
        loss_after = sampled_meta_loss(candidate, tasks, args, bound) if bound >= 0 else math.inf
        if math.isfinite(loss_after) and loss_after <= bound and loss_after < loss_before:
            result.guard_before = sampled_meta_loss(original, guard_tasks, args)
            result.guard_after = sampled_meta_loss(candidate, guard_tasks, args)
            if not (math.isfinite(result.guard_before) and math.isfinite(result.guard_after)
                    and result.guard_after < result.guard_before):
                result.reason = 'independent_batch_rejected'
                # A noisy held-out rejection need not indicate excessive step
                # length. Keep the coefficient; repeated guard rejections must
                # not geometrically shrink every future update to zero.
                result.next_step = step
                return result
            images.copy_(candidate)
            result.loss_after = loss_after
            result.step_size = step
            result.accepted = True
            result.reason = 'accepted'
            result.next_step = step
            return result
        step *= 0.5
        result.next_step = step
    return result
