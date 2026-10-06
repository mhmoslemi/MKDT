"""Match SSL training updates, without differentiating through a training trajectory.

For a fixed reference state theta and A augmentation draws, let
    g_D = mean_a grad_theta L_SSL(D, theta, a).
The image objective is ||g_S - g_T||^2 / max(||g_T||^2, 1e-12), averaged
over reference states. There is no division by the number of parameters.
Starting from identical parameters and SGD momentum state, one SGD update obeys
    ||theta_S_next - theta_T_next||^2 = eta^2 * ||g_S - g_T||^2,
where eta is the reference optimizer's current learning rate.
This is a local update-matching statement, not an accuracy guarantee.

References advance on real data only. Their weights remain fixed while computing
the image gradient and accepting a pixel step. Matching uses mixed derivatives
of the actual SSL loss; no reference update is unrolled into the image graph.
Gradient matching: https://arxiv.org/abs/2006.05929 (adapted here to SSL).
"""

from contextlib import contextmanager
from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.parameters = tuple(p for p in self.encoder.parameters() if p.requires_grad) + tuple(self.projector.parameters())
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

    def advance(self, real_gradients, args):
        """One real-data SGD step, outside the synthetic-image computation graph."""
        if not all(torch.isfinite(gradient).all().item() for gradient in real_gradients):
            raise FloatingPointError('Non-finite real SSL gradient; reference update aborted')
        self.optimizer.zero_grad(set_to_none=True)
        for parameter, gradient in zip(self.parameters, real_gradients):
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


def squared_norm(gradients):
    return torch.stack([gradient.square().sum() for gradient in gradients]).sum()


@dataclass
class GradientTarget:
    reference: SSLReference
    indices: torch.Tensor
    view_pairs: list
    gradients: tuple
    norm_squared: torch.Tensor


def gradient_match_and_backward(reference, real_images, images, indices, view_pairs, args):
    """Differentiate the squared difference of MEAN SSL gradients over views.

    The upstream derivative for g_S,a is 2*(g_S-g_T)/(K*A*||g_T||^2).
    Replaying one view pair at a time evaluates the mixed theta/image derivative
    exactly for these sampled batches, without retaining A second-order graphs.
    """
    real_gradients, real_ssl = mean_parameter_gradients(reference, real_images, view_pairs, args)
    synthetic_gradients, synthetic_ssl = mean_parameter_gradients(
        reference, images.detach().index_select(0, indices), view_pairs, args
    )
    normalizer = squared_norm(real_gradients).clamp_min(1e-12)
    residuals = tuple(synthetic - real for synthetic, real in zip(synthetic_gradients, real_gradients))
    error = squared_norm(residuals)
    loss = error / normalizer
    if not torch.isfinite(loss).item():
        raise FloatingPointError('Non-finite SSL gradient-matching loss')
    upstream = tuple(
        2 * residual / (args.num_reference_nets * len(view_pairs) * normalizer)
        for residual in residuals
    )
    for seeds in view_pairs:
        # A fresh index_select graph each time allows independent backward passes.
        gradients, _ = parameter_gradients(
            reference, images.index_select(0, indices), seeds, args, create_graph=True
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
    synthetic_norm = squared_norm(synthetic_gradients)
    dot = torch.stack([(real * synthetic).sum() for real, synthetic in zip(real_gradients, synthetic_gradients)]).sum()
    cosine = dot / (normalizer * synthetic_norm.clamp_min(1e-12)).sqrt()
    target = GradientTarget(reference, indices, view_pairs, real_gradients, normalizer)
    metrics = {
        'loss': loss.item(), 'cosine': cosine.item(),
        'norm_ratio': (synthetic_norm / normalizer).sqrt().item(),
        'real_ssl': real_ssl.item(), 'synthetic_ssl': synthetic_ssl.item(),
    }
    return target, metrics


def sampled_gradient_match_loss(images, targets, args, reject_above):
    total = 0.0
    for target in targets:
        gradients, _ = mean_parameter_gradients(
            target.reference, images.index_select(0, target.indices), target.view_pairs, args
        )
        error = squared_norm(tuple(synthetic - real for synthetic, real in zip(gradients, target.gradients)))
        total += (error / target.norm_squared).item() / len(targets)
        if not math.isfinite(total) or total > reject_above:
            return math.inf
    return total


@torch.no_grad()
def projected_backtracking_step(images, targets, args, lower, upper, pixel_std, loss_before, initial_step):
    """Bound pixel movement and decrease the objective for these sampled targets.

    Armijo descent on one minibatch does not imply expected-loss descent. In
    particular, carrying a growing step coefficient across changing references
    and minibatches can accept increasingly large updates that fit sampling noise.
    Recompute the RMS bound from the CURRENT gradient on EVERY iteration.
    """
    gradient = images.grad.detach()
    original = images.detach()
    gradient_pixel_rms = (gradient.double() * pixel_std * 255).square().mean().sqrt().item()
    if not math.isfinite(gradient_pixel_rms) or gradient_pixel_rms == 0:
        return loss_before, 0.0, 0, False
    step_limit = min(
        args.lr_img / max(gradient_pixel_rms, 1e-12), math.sqrt(torch.finfo(images.dtype).max)
    )
    step = step_limit if initial_step is None else min(initial_step, step_limit)
    for trial in range(1, 21):
        candidate = torch.maximum(torch.minimum(original - step * gradient, upper), lower)
        slope = (gradient * (candidate - original)).sum(dtype=torch.float64).item()
        if not math.isfinite(slope):
            step *= 0.5
            continue
        if slope >= 0:
            return loss_before, 0.0, trial, False
        bound = loss_before + 1e-4 * slope
        loss_after = sampled_gradient_match_loss(candidate, targets, args, bound) if bound >= 0 else math.inf
        if math.isfinite(loss_after) and loss_after <= bound and loss_after < loss_before:
            images.copy_(candidate)
            return loss_after, step, trial, True
        step *= 0.5
    return loss_before, 0.0, 20, False
