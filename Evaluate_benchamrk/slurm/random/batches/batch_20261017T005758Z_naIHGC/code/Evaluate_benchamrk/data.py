"""Raw images, composed SSL views, exact label budgets, and subset selection.

Reuse the repository's official split readers and label selector. Do not reuse
the old normalized 32px feature caches or DiffAugment training pipeline.
"""

import json
from pathlib import Path
import random
import sys

import numpy as np
from PIL import Image, ImageFilter
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms as T

from common import ROOT, atomic_json, digest_json, file_hash, seed_all, worker_seed

sys.path.insert(0, str(ROOT / 'benchmarks'))
from downstream_data import SPECS, locate, records, select_labels
from networks import ConvNetGAP

TARGETS = ('CIFAR10', 'CIFAR100', 'TinyImageNet', 'Aircraft', 'CUB2011', 'Dogs', 'Flowers')
SOURCE_COUNTS = {'CIFAR10': 50000, 'CIFAR100': 50000, 'TinyImageNet': 100000}


class RawDataset(Dataset):
    def __init__(self, root, name, training):
        self.name = name
        self.classes = SPECS[name][0]
        self.images = self.samples = self.base = None
        folder = locate(root, name)
        if name in ('CIFAR10', 'CIFAR100'):
            cls = getattr(datasets, name)
            self.base = cls(str(folder), train=training, download=False)
            self.targets = np.asarray(self.base.targets, dtype=np.int64)
            import hashlib
            pixels = hashlib.sha256(self.base.data.tobytes()).hexdigest()
            identity = {'pixels_sha256': pixels}
        elif name == 'TinyImageNet':
            path = folder / 'tinyimagenet.pt'
            saved = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
            split = 'train' if training else 'val'
            self.images = saved['images_' + split]
            self.targets = saved['labels_' + split].long().numpy()
            if self.images.dtype != torch.uint8 or tuple(self.images.shape[1:]) != (3, 64, 64):
                raise ValueError('Expected native 64x64 uint8 Tiny ImageNet archive')
            identity = {'archive_sha256': file_hash(path), 'split': split}
        else:
            self.samples = records(root, name, training)
            self.targets = np.asarray([label for _, label in self.samples], dtype=np.int64)
            # File metadata fingerprint: explicitly NOT claimed as a pixel hash.
            identity = {'files_size_mtime': [
                (str(path.relative_to(folder)), label, path.stat().st_size, path.stat().st_mtime_ns)
                for path, label in self.samples]}
        if len(self.targets) == 0 or self.targets.min() < 0 or self.targets.max() >= self.classes:
            raise ValueError(f'Invalid labels in {name}')
        if self.images is not None and len(self.images) != len(self.targets):
            raise ValueError('Image/label count mismatch')
        self.fingerprint = digest_json({'name': name, 'training': training,
                                       'labels': self.targets.tolist(), **identity})

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        if self.base is not None:
            image = self.base[index][0]
        elif self.images is not None:
            image = T.functional.to_pil_image(self.images[index])
        else:
            with Image.open(self.samples[index][0]) as handle:
                image = handle.convert('RGB')
        return image, int(self.targets[index])


class ImageView(Dataset):
    def __init__(self, base, transform):
        self.base, self.transform = base, transform

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        image, label = self.base[index]
        return self.transform(image), label


class SyntheticDataset(Dataset):
    def __init__(self, entry, source, percentage, size):
        path = Path(entry['path'])
        if not path.is_absolute():
            path = ROOT / path
        saved = torch.load(path, map_location='cpu', weights_only=True)
        images = saved[entry['key']] if entry.get('key') else saved
        expected_count = int(SOURCE_COUNTS[source] * percentage / 100)
        if not isinstance(images, torch.Tensor) or images.ndim != 4 or images.shape[:2] != (expected_count, 3):
            raise ValueError(f'{path}: expected {expected_count} RGB synthetic images')
        if images.shape[2:] != (size, size) or not images.is_floating_point() or not torch.isfinite(images).all():
            raise ValueError(f'{path}: expected finite floating NCHW images at native size {size}')
        if entry['space'] == 'repository_normalized':
            _, mean, std = SPECS[source]
            images = images * torch.tensor(std)[None, :, None, None] + torch.tensor(mean)[None, :, None, None]
        elif entry['space'] != 'unit_rgb':
            raise ValueError('Synthetic image space must be explicit')
        fraction = float(((images < 0) | (images > 1)).float().mean())
        # Explicit, recorded conversion to the RGB domain required by PIL color
        # augmentation. Never reinterpret normalized tensors as uint8 directly.
        if entry.get('pixel_policy') != 'clamp_to_unit_rgb':
            raise ValueError('Synthetic augmentation requires an explicit pixel_policy')
        self.images = images.clamp(0, 1).contiguous()
        self.metadata = {'path': str(path.resolve()), 'file_sha256': file_hash(path),
                         'space': entry['space'], 'pixel_policy': entry['pixel_policy'],
                         'clipped_fraction': fraction, 'count': len(images),
                         'generator': entry.get('generator', 'unspecified'),
                         'quantization': 'float_to_PIL_uint8_for_reference_augmentations'}
        self.fingerprint = digest_json(self.metadata)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return T.functional.to_pil_image(self.images[index]), -1


class GaussianBlur:
    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, image):
        return image.filter(ImageFilter.GaussianBlur(random.uniform(*self.sigma)))


class TwoViews:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image):
        return tuple(transform(image) for transform in self.transforms)


class ProbeCrop:
    """Keep native CIFAR/Tiny test images intact; crop larger transfer images."""
    def __init__(self, size, ratio):
        self.size = size
        self.transform = T.Compose([T.Resize(round(size * ratio), interpolation=T.InterpolationMode.BICUBIC),
                                    T.CenterCrop(size)])

    def __call__(self, image):
        return image if image.size == (self.size, self.size) else self.transform(image)


def ssl_transform(protocol, source, method):
    common, recipe = protocol['ssl'], protocol['methods'][method]
    views = []
    for i in range(2):
        transforms = [
            T.RandomResizedCrop(protocol['input_size'][source], scale=tuple(common['crop_scale']),
                                ratio=tuple(common['crop_ratio']), interpolation=T.InterpolationMode.BICUBIC),
            T.RandomApply([T.ColorJitter(*recipe['color_jitter'])], p=common['color_probability']),
            T.RandomGrayscale(p=common['grayscale_probability']),
            T.RandomApply([GaussianBlur(common['blur_sigma'])], p=recipe['blur_probabilities'][i]),
            T.RandomSolarize(128, p=recipe['solarize_probabilities'][i]),
            T.RandomHorizontalFlip(p=common['flip_probability']),
            T.ToTensor(), T.Normalize(*protocol['normalization'][source])]
        views.append(T.Compose(transforms))
    return TwoViews(views)


def probe_transform(protocol, source, training):
    size = protocol['input_size'][source]
    if training:
        operations = [T.RandomResizedCrop(size, interpolation=T.InterpolationMode.BICUBIC),
                      T.RandomHorizontalFlip()]
    else:
        operations = [ProbeCrop(size, protocol['evaluation']['test_resize_ratio'])]
    return T.Compose(operations + [T.ToTensor(), T.Normalize(*protocol['normalization'][source])])


def loader(dataset, batch, workers, seed, shuffle=False, drop_last=False):
    return DataLoader(dataset, batch_size=batch, shuffle=shuffle, drop_last=drop_last,
                      num_workers=workers, pin_memory=torch.cuda.is_available(),
                      worker_init_fn=worker_seed, generator=torch.Generator().manual_seed(seed))


def label_indices(dataset, percentage, seed):
    # Reuse the exact, nested, class-balanced repository selector. It never
    # expands the requested global budget to manufacture one label per class.
    return select_labels(dataset.targets, percentage, seed, policy='exact').tolist()


@torch.no_grad()
def kmeans_indices(features, count, seed, iterations, chunk):
    """Lloyd iterations with bounded distances and unique nearest exemplars."""
    generator = torch.Generator(device=features.device).manual_seed(seed + 1000)
    centers = features[torch.randperm(len(features), generator=generator, device=features.device)[:count]].clone()
    norm = features.square().sum(1)
    for _ in range(iterations):
        sums = torch.zeros_like(centers)
        counts = torch.zeros(count, device=features.device)
        for start in range(0, len(features), chunk):
            block = features[start:start + chunk]
            dist = norm[start:start + chunk, None] + centers.square().sum(1)[None] - 2 * block @ centers.T
            assignment = dist.argmin(1)
            sums.index_add_(0, assignment, block)
            counts.add_(torch.bincount(assignment, minlength=count))
        updated = sums / counts.clamp_min(1)[:, None]
        updated[counts == 0] = centers[counts == 0]
        converged = torch.allclose(updated, centers, atol=1e-4, rtol=0)
        centers = updated
        if converged:
            break
    used = torch.zeros(len(features), dtype=torch.bool, device=features.device)
    chosen = []
    for block in centers.split(64):
        distances = block.square().sum(1)[:, None] + norm[None] - 2 * block @ features.T
        for row in distances:
            row[used] = torch.inf
            index = int(row.argmin())
            used[index] = True
            chosen.append(index)
    return chosen


def select_subset(base, method, percentage, seed, protocol, cache, workers, device):
    count = int(len(base) * percentage / 100)
    if count < 2 or count > len(base):
        raise ValueError('SSL needs at least two distinct images')
    identity = {'dataset': base.name, 'fingerprint': base.fingerprint, 'seed': seed,
                'percentage': percentage, 'count': count, 'method': method,
                'selection': protocol['selection'], 'normalization': protocol['normalization'][base.name],
                'implementation': file_hash(Path(__file__)), 'networks': file_hash(ROOT / 'networks.py')}
    path = Path(cache) / f'{digest_json(identity)}.json'
    if path.exists():
        saved = json.loads(path.read_text())
        if saved['identity'] != identity:
            raise ValueError(f'Selection identity mismatch: {path}')
        chosen = saved['indices']
    elif method == 'random':
        chosen = np.random.default_rng(seed + 1000).permutation(len(base))[:count].tolist()
    elif method == 'kmeans':
        cfg = protocol['selection']
        seed_all(seed + cfg['feature_seed_offset'])
        model = ConvNetGAP(3, 1, 128, 3, 'relu', 'batchnorm', 'avgpooling')
        model.classifier = nn.Identity()
        model.to(device).eval()
        features = []
        stream = loader(ImageView(base, probe_transform(protocol, base.name, False)),
                        cfg['batch_size'], workers, seed)
        with torch.no_grad():
            for images, _ in stream:
                features.append(F.normalize(model(images.to(device)), dim=1))
        features = torch.cat(features)
        chosen = kmeans_indices(features, count, seed, cfg['iterations'], cfg['distance_chunk_size'])
        del model, features
    else:
        raise ValueError(method)
    if len(chosen) != count or len(set(chosen)) != count or min(chosen) < 0 or max(chosen) >= len(base):
        raise ValueError('Invalid subset indices')
    atomic_json(path, {'identity': identity, 'indices': chosen})
    return Subset(base, chosen), {'selection_sha256': digest_json(identity), 'indices': chosen}
