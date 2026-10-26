"""Native-resolution port of the official A²MIM CNN recipe (Apache-2.0).

Pinned upstream definitions and their license are in reference/. The backbone
is our existing BN ResNet18; its stride is 8 instead of ImageNet's stride 32.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from common import build_encoder


def block_mask(batch, size, patch_size, device):
    grid = size // patch_size
    count = math.ceil(grid * grid * 0.6)
    order = torch.rand(batch, grid * grid, device=device).argsort(dim=1)
    mask = torch.zeros(batch, grid * grid, device=device)
    mask.scatter_(1, order[:, :count], 1)
    return mask.reshape(batch, 1, grid, grid).repeat_interleave(
        patch_size, 2).repeat_interleave(patch_size, 3)


def reconstruction_loss(prediction, target, mask):
    # Official focal_l1_loss(alpha=.2, gamma=1, sigmoid, residual=False).
    error = (prediction - target).abs()
    spatial = (error * (2 * torch.sigmoid(0.2 * error) - 1) * mask).sum()
    spatial = spatial / (mask.sum() + 1e-5) / 3

    # Official A2MIMHead: mixed detached replacement, unmasked FFT weight=.001.
    fft_mask = mask + (1 - mask) * 0.001
    replacement = 0.5 * prediction.detach() + 0.5 * target
    fft_prediction = prediction * fft_mask + replacement * (1 - fft_mask)
    predicted_spectrum = torch.fft.fft2(fft_prediction, norm='ortho').mean(0, keepdim=True)
    target_spectrum = torch.fft.fft2(target, norm='ortho').mean(0, keepdim=True)
    difference = predicted_spectrum - target_spectrum
    distance = difference.real.square() + difference.imag.square()
    weight = torch.log1p(distance.sqrt())
    weight = (weight / weight.max()).nan_to_num(nan=0).clamp(0, 1).detach()
    frequency = (weight * distance).mean()
    return spatial + 0.5 * frequency, spatial, frequency


class A2MIM(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.encoder, dimension = build_encoder('ResNet18', size)
        # Stage-3 output is 256 channels for BasicBlock ResNet18.
        self.mask_token = nn.Parameter(torch.zeros(1, 256, 1, 1))
        self.mask_gamma = nn.Parameter(torch.full((1, 256, 1, 1), 1e-6))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.decoder = nn.Sequential(nn.Conv2d(dimension, 3 * 8**2, 1), nn.PixelShuffle(8))
        nn.init.trunc_normal_(self.decoder[0].weight, std=0.02)
        nn.init.zeros_(self.decoder[0].bias)

    def forward_mask(self, features, mask):
        mask = F.interpolate(mask, size=features.shape[-2:], mode='nearest')
        return (features * (1 - mask) + features * mask * self.mask_gamma
                + self.mask_token * mask)

    def forward(self, images, mask):
        # The official tensor pipeline fills masked pixels with each image's mean.
        masked = images * (1 - mask) + images.mean((2, 3), keepdim=True) * mask
        encoder = self.encoder
        features = F.relu(encoder.bn1(encoder.conv1(masked)))
        features = encoder.layer3(encoder.layer2(encoder.layer1(features)))
        features = self.forward_mask(features, mask)
        features = encoder.layer4(features)
        return self.decoder(features)


def optimizer_for(model, batch_size):
    no_decay = {id(p) for m in model.modules() if isinstance(m, nn.BatchNorm2d)
                for p in m.parameters(recurse=False)}
    groups = []
    for name, parameter in model.named_parameters():
        token = name == 'mask_token'
        decay = 0 if token or id(parameter) in no_decay or name.endswith('.bias') else 0.05
        multiplier = 0.1 if token else 1.0
        groups.append({'params': [parameter], 'weight_decay': decay,
                       'lr_mult': multiplier})
    return torch.optim.AdamW(groups, lr=1.2e-3 * batch_size / 2048,
                             betas=(0.9, 0.999), eps=1e-8)


def learning_rate(step, steps_per_epoch, epochs, peak):
    warmup = 10 * steps_per_epoch
    # Official StepFixCosineAnnealing removes warmup from the cosine duration.
    progress = (step - warmup) / (epochs * steps_per_epoch - warmup)
    regular = 1e-5 + (peak - 1e-5) * (1 + math.cos(math.pi * progress)) / 2
    return regular * step / warmup if step < warmup else regular
