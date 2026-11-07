from pathlib import Path
import json
import numpy as np
import torch
from PIL import Image

torch.set_num_threads(1)
source = Path('/scratch/mmoslem3/MKDT/PLOTS/intro_figs/res_IIC_CIFAR10_ResNet18BN_0.2percent_final.pt')
checkpoint = torch.load(source, map_location='cpu', weights_only=True)
images = checkpoint['data'].detach().float()
assert checkpoint['dataset'] == 'CIFAR10'
assert images.ndim == 4 and images.shape[1:] == (3, 32, 32)
assert images.shape[0] > 0 and torch.isfinite(images).all()
mean, std = (torch.tensor(value, dtype=images.dtype).reshape(1, 3, 1, 1)
             for value in checkpoint['normalization'])
assert torch.isfinite(mean).all() and torch.isfinite(std).all() and (std > 0).all()
pixels = (images * std + mean).clamp(0, 1).mul(255).add(.5).to(torch.uint8)
pixels = pixels.permute(0, 2, 3, 1).contiguous().numpy()
destination = source.parent / 'fig'
destination.mkdir(exist_ok=True)
paths = [destination / f'image_{i:03d}.png' for i in range(len(pixels))]
assert not any(path.exists() for path in paths), 'Refusing to overwrite existing images'
for path, array in zip(paths, pixels):
    Image.fromarray(array).save(path, format='PNG')
    with Image.open(path) as saved:
        assert saved.mode == 'RGB' and saved.size == (32, 32)
        assert np.array_equal(np.asarray(saved), array)
print(json.dumps(dict(source=str(source), tensor='data', count=len(paths), shape=list(images.shape),
    normalization=checkpoint['normalization'], destination=str(destination),
    first=paths[0].name, last=paths[-1].name, verified=True)), flush=True)
