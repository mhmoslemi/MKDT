"""Official digit splits presented as 32px RGB to the existing BN benchmark."""
import hashlib
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms as T
from PIL import Image

from common import digest_json

COUNTS = {'SVHN': (73257, 26032), 'MNIST': (60000, 10000)}


class DigitDataset(Dataset):
    def __init__(self, root, name, training):
        self.name, self.classes = name, 10
        if name == 'SVHN':
            base = datasets.SVHN(str(Path(root) / 'SVHN'),
                                 split='train' if training else 'test', download=False)
            self.images = torch.from_numpy(np.ascontiguousarray(base.data))
            self.targets = np.asarray(base.labels, dtype=np.int64)
            preprocessing = 'native_RGB_32; torchvision maps label 10 to 0; no extra split'
        elif name == 'MNIST':
            base = datasets.MNIST(str(root), train=training, download=False)
            # Resize once, not on every image of every training epoch.
            pixels = [np.array(Image.fromarray(image.numpy()).resize(
                (32, 32), Image.Resampling.BICUBIC), copy=True) for image in base.data]
            self.images = torch.from_numpy(np.stack(pixels))[:, None]
            self.targets = base.targets.numpy().astype(np.int64)
            preprocessing = 'PIL bicubic 28-to-32; grayscale repeated into RGB'
        else:
            raise ValueError(name)
        assert len(self.targets) == COUNTS[name][0 if training else 1]
        assert self.images.dtype == torch.uint8 and self.images.shape[-2:] == (32, 32)
        assert np.array_equal(np.unique(self.targets), np.arange(10))
        self.fingerprint = digest_json(dict(name=name, training=training,
            labels=self.targets.tolist(), preprocessing=preprocessing,
            pixels_sha256=hashlib.sha256(self.images.numpy().tobytes()).hexdigest()))

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return T.functional.to_pil_image(self.images[index]).convert('RGB'), int(self.targets[index])

    def normalization(self):
        """Population channel statistics, called on the training split only."""
        sums = torch.zeros(self.images.shape[1], dtype=torch.float64)
        squares = torch.zeros_like(sums)
        for block in self.images.split(256):
            pixels = block.double().div_(255)
            sums += pixels.sum((0, 2, 3))
            squares += pixels.square().sum((0, 2, 3))
        count = len(self) * 32 * 32
        mean = sums / count
        std = (squares / count - mean.square()).sqrt()
        assert torch.isfinite(std).all() and (std > 0).all()
        if len(mean) == 1:
            mean, std = mean.repeat(3), std.repeat(3)
        return [mean.tolist(), std.tolist()]
