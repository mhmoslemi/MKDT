"""Single-GPU SimCLR / Barlow training, with explicit CIFAR reference recipes.

Recipe and equations: solo-learn (MIT) and the original SimCLR / Barlow papers.
This is a local implementation, not a claim of bitwise official reproduction.
"""

import json
import math
from pathlib import Path
import time

import torch
from torch import nn
from torch.nn import functional as F

from common import (atomic_save, build_encoder, load_encoder, restore_rng,
                    rng_state, seed_all, state_hash)
from data import ImageView, loader, ssl_transform


class LARS(torch.optim.Optimizer):
    """Momentum SGD with the clipped LARS rule used by the CIFAR reference.

All bias and normalization parameters are excluded from decay and adaptation.
The denominator uses ||gradient|| + decay*||weight|| + epsilon, before momentum.
"""
    def __init__(self, parameters, lr, cfg):
        super().__init__(parameters, dict(lr=lr, momentum=cfg['momentum'],
                         weight_decay=cfg['weight_decay'], eta=cfg['lars_eta'],
                         eps=cfg['lars_eps'], clip=cfg['lars_clip']))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for parameter in group['params']:
                if parameter.grad is None:
                    continue
                update = parameter.grad
                if parameter.ndim > 1:
                    weight_norm, grad_norm = parameter.norm(), update.norm()
                    trust = group['eta'] * weight_norm / (
                        grad_norm + group['weight_decay'] * weight_norm + group['eps'])
                    trust = torch.where((weight_norm > 0) & (grad_norm > 0), trust,
                                        torch.ones_like(trust))
                    if group['clip']:
                        trust = torch.clamp(trust / max(group['lr'], 1e-16), max=1.0)
                    update = (update + group['weight_decay'] * parameter) * trust
                state = self.state[parameter]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(parameter)
                velocity = state['momentum_buffer']
                velocity.mul_(group['momentum']).add_(update)
                parameter.add_(velocity, alpha=-group['lr'])
        return loss


def cosine_lr(step, total, peak, warmup=0, start=0., minimum=0.):
    if warmup and step < warmup:
        return start + (peak - start) * step / max(1, warmup - 1)
    progress = (step - warmup) / max(1, total - warmup - 1)
    return minimum + (peak - minimum) * 0.5 * (1 + math.cos(math.pi * progress))


def simclr_loss(first, second, temperature):
    if len(first) != len(second) or len(first) < 2:
        raise ValueError('SimCLR requires paired batches with at least two images')
    features = F.normalize(torch.cat((first, second)).float(), dim=1)
    scores = features @ features.T / temperature
    scores.fill_diagonal_(-torch.inf)
    count = len(first)
    positives = (torch.arange(2 * count, device=scores.device) + count) % (2 * count)
    return F.cross_entropy(scores, positives)


def barlow_loss(first, second, lamb=0.0051, scale=0.1, eps=1e-5):
    if first.shape != second.shape or len(first) < 2:
        raise ValueError('Barlow requires paired batches with at least two images')
    # Biased (divide-by-N) variance matches BatchNorm1d(affine=False).
    first, second = first.float(), second.float()
    a = (first - first.mean(0)) * torch.rsqrt(first.var(0, unbiased=False) + eps)
    b = (second - second.mean(0)) * torch.rsqrt(second.var(0, unbiased=False) + eps)
    correlation = a.T @ b / len(a)
    diagonal = torch.diagonal(correlation)
    dimension = correlation.shape[0]
    off = correlation.flatten()[:-1].view(dimension - 1, dimension + 1)[:, 1:].flatten()
    return scale * ((diagonal - 1).square().sum() + lamb * off.square().sum())


class SSLModel(nn.Module):
    def __init__(self, model, size, recipe):
        super().__init__()
        self.encoder, self.dimension = build_encoder(model, size)
        dims = [self.dimension] + recipe['projector']
        layers = []
        for index, (incoming, outgoing) in enumerate(zip(dims[:-1], dims[1:])):
            layers.append(nn.Linear(incoming, outgoing, bias=recipe['projector_bias']))
            if index < len(dims) - 2:
                if recipe['projector_batchnorm']:
                    layers.append(nn.BatchNorm1d(outgoing))
                layers.append(nn.ReLU())
        self.projector = nn.Sequential(*layers)

    def forward(self, first, second):
        return self.projector(self.encoder(first)), self.projector(self.encoder(second))


def train(dataset, protocol, source, model_name, method, seed, output,
          identity, workers, device, full=False, stop_after_epoch=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    destination = output / 'encoder.pt'
    if destination.exists():
        encoder, _, saved = load_encoder(destination, identity, device='cpu')
        del encoder
        return destination, saved
    seed_all(seed + 3000)
    size = protocol['input_size'][source]
    cfg = protocol['ssl']
    if method == 'none':
        encoder, _ = build_encoder(model_name, size)
        epochs, steps, effective_batch = 0, 0, 0
    else:
        recipe = protocol['methods'][method]
        epochs = cfg['full_epochs' if full else 'subset_epochs']
        if stop_after_epoch is not None and (
                type(stop_after_epoch) is not int or not 1 <= stop_after_epoch <= epochs):
            raise ValueError('stop_after_epoch must be within the configured training budget')
        # A scheduler boundary does not change the total LR/warmup schedule.
        end_epoch = epochs if stop_after_epoch is None else stop_after_epoch
        effective_batch = min(len(dataset), cfg['batch_size'])
        if effective_batch < 2:
            raise ValueError('Cannot train SSL on fewer than two distinct images')
        stream = loader(ImageView(dataset, ssl_transform(protocol, source, method)),
                        effective_batch, workers, seed, shuffle=True, drop_last=True)
        model = SSLModel(model_name, size, recipe).to(device)
        peak = recipe['lr'] * (effective_batch / cfg['batch_size'] if cfg['scale_lr_with_batch'] else 1)
        optimizer = LARS(model.parameters(), peak, cfg)
        total_steps = epochs * len(stream)
        warmup_steps = min(cfg['warmup_epochs'], max(0, epochs - 1)) * len(stream)
        resume = output / 'training.pt'
        start_epoch = steps = 0
        if resume.exists():
            saved = torch.load(resume, map_location='cpu', weights_only=True)
            if saved['identity'] != identity or saved['steps_per_epoch'] != len(stream):
                raise ValueError('Training resume protocol or dataset mismatch')
            model.load_state_dict(saved['model'], strict=True)
            optimizer.load_state_dict(saved['optimizer'])
            start_epoch, steps = saved['epoch'], saved['steps']
            restore_rng(saved['rng'])
        with (output / 'ssl.jsonl').open('a', buffering=1) as log:
            for epoch in range(start_epoch, end_epoch):
                # Each epoch has an independent deterministic augmentation/shuffle
                # stream, including after restart and with workers=0.
                epoch_seed = seed * 1000003 + epoch + 10000
                seed_all(epoch_seed)
                stream.generator.manual_seed(epoch_seed)
                model.train()
                start_time = time.monotonic()
                weighted_loss = examples = 0
                for (first, second), _unused_labels in stream:
                    lr = cosine_lr(steps, total_steps, peak, warmup_steps,
                                   cfg['warmup_start_lr'], cfg['minimum_lr'])
                    for group in optimizer.param_groups:
                        group['lr'] = lr
                    first, second = first.to(device, non_blocking=True), second.to(device, non_blocking=True)
                    optimizer.zero_grad(set_to_none=True)
                    z1, z2 = model(first, second)
                    if method == 'simclr':
                        loss = simclr_loss(z1, z2, recipe['temperature'])
                    else:
                        loss = barlow_loss(z1, z2, recipe['lambda'], recipe['loss_scale'], recipe['correlation_eps'])
                    if not torch.isfinite(loss):
                        raise RuntimeError(f'Non-finite SSL loss at epoch {epoch}, step {steps}')
                    loss.backward()
                    finite = torch.stack([torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None]).all()
                    if not finite:
                        raise RuntimeError(f'Non-finite SSL gradient at epoch {epoch}, step {steps}')
                    optimizer.step()
                    steps += 1
                    examples += len(first)
                    weighted_loss += float(loss.detach()) * len(first)
                stats = {'epoch': epoch + 1, 'steps': steps, 'lr': lr,
                         'loss': weighted_loss / examples, 'examples': examples,
                         'projector_std': float(z1.detach().std(0, unbiased=False).mean()),
                         'seconds': time.monotonic() - start_time}
                line = json.dumps(stats, allow_nan=False)
                print(line, file=log)
                print(f'SSL {source}/{model_name}/{method} seed={seed}: {line}', flush=True)
                if (epoch + 1) % cfg['checkpoint_every_epochs'] == 0 or epoch + 1 == end_epoch:
                    atomic_save(resume, {'identity': identity, 'epoch': epoch + 1, 'steps': steps,
                                        'steps_per_epoch': len(stream), 'model': model.state_dict(),
                                        'optimizer': optimizer.state_dict(), 'rng': rng_state()})
        if end_epoch < epochs:
            # Do not present an intermediate stage as a finished SSL encoder.
            return None, {'complete': False, 'epoch': max(start_epoch, end_epoch),
                          'steps': steps, 'effective_batch': effective_batch}
        encoder = model.encoder.cpu()
    saved = {'kind': 'ssl_reference_encoder', 'complete': True, 'identity': identity,
             'model': model_name, 'input_size': size, 'source': source, 'method': method,
             'seed': seed, 'epochs': epochs, 'steps': steps, 'effective_batch': effective_batch,
             'encoder': encoder.state_dict(), 'encoder_sha256': state_hash(encoder),
             'normalization': protocol['normalization'][source]}
    atomic_save(destination, saved)
    # Reload strictly before any probe: no missing keys, classifier mismatch,
    # silent random initialization, or projector features can pass this boundary.
    checked, _, _ = load_encoder(destination, identity, device='cpu')
    del checked
    return destination, {key: value for key, value in saved.items() if key != 'encoder'}
