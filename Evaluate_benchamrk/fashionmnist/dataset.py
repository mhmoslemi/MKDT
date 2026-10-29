"""FashionMNIST with the same 32px RGB preprocessing as the MNIST benchmark."""
import hashlib

import numpy as np
import torch
from PIL import Image
from torchvision.datasets import FashionMNIST

from common import digest_json
from datasets import DigitDataset as BaseDataset

COUNTS = {'FashionMNIST': (60000, 10000)}


class DigitDataset(BaseDataset):
    def __init__(self, root, name, training):
        assert name == 'FashionMNIST'
        self.name, self.classes = name, 10
        base = FashionMNIST(str(root), train=training, download=False)
        pixels = [np.array(Image.fromarray(image.numpy()).resize(
            (32, 32), Image.Resampling.BICUBIC), copy=True) for image in base.data]
        self.images = torch.from_numpy(np.stack(pixels))[:, None]
        self.targets = base.targets.numpy().astype(np.int64)
        assert len(self.targets) == COUNTS[name][0 if training else 1]
        assert self.images.dtype == torch.uint8 and self.images.shape[-2:] == (32, 32)
        assert np.array_equal(np.unique(self.targets), np.arange(10))
        self.fingerprint = digest_json(dict(name=name, training=training,
            labels=self.targets.tolist(),
            preprocessing='PIL bicubic 28-to-32; grayscale repeated into RGB',
            pixels_sha256=hashlib.sha256(self.images.numpy().tobytes()).hexdigest()))
