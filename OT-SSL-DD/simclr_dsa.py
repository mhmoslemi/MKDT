"""Differentiable SimCLR views with optional single-family/composed Siamese sampling."""
import math

import torch
from torchvision import transforms as T
from torchvision.transforms import functional as TF


class TensorGaussianBlur:
    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, pixels):
        sigma = float(torch.empty(1).uniform_(*self.sigma))
        radius = min(math.ceil(4 * sigma), min(pixels.shape[-2:]) - 1)
        # PIL blur/uint8 rounding cannot supply image gradients; use a floating Gaussian kernel.
        return TF.gaussian_blur(pixels, [2 * radius + 1] * 2, [sigma] * 2)


class SimCLRDSA:
    def __init__(self, view_transform, mean, std, compose_probability=0.5):
        self.mean = torch.tensor(mean).view(1, 3, 1, 1)
        self.std = torch.tensor(std).view(1, 3, 1, 1)
        self.compose_probability = compose_probability
        assert 0 <= compose_probability <= 1
        operations = list(view_transform.transforms)
        assert isinstance(operations[-2], T.ToTensor) and isinstance(operations[-1], T.Normalize)
        operations = operations[:-2]
        assert len(operations) == 6 and isinstance(operations[0], T.RandomResizedCrop)
        assert isinstance(operations[3], T.RandomApply) and len(operations[3].transforms) == 1
        blur = operations[3].transforms[0]
        operations[3] = T.RandomApply([TensorGaussianBlur(blur.sigma)], p=operations[3].p)
        assert isinstance(operations[4], T.RandomSolarize) and operations[4].p == 0
        self.operations = operations[:4] + operations[5:]

    def __call__(self, images, seed):
        mean, std = self.mean.to(images), self.std.to(images)
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            torch.manual_seed(int(seed))
            pixels = (images * std + mean).clamp(0, 1)
            composed = torch.rand(()) < self.compose_probability
            operations = self.operations if composed else [self.operations[int(torch.randint(len(self.operations), ()))]]
            for operation in operations:
                pixels = operation(pixels).clamp(0, 1)
            return (pixels.clamp(0, 1) - mean) / std
