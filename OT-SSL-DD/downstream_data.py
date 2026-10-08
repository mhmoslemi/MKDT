"""Offline readers and compact 32x32 caches for the benchmark's target datasets."""

from pathlib import Path

import numpy as np
from PIL import Image
from scipy.io import loadmat
import torch
from torchvision.datasets import CIFAR100


# Match the existing target preprocessing in the repository's root utils.py.
SPECS = {
    'CIFAR100': (100, [0.5071, 0.4865, 0.4409], [0.2009, 0.1984, 0.2023]),
    'Aircraft': (100, [0.4804, 0.5116, 0.5349], [0.2021, 0.1953, 0.2297]),
    'CUB2011': (200, [0.4857, 0.4995, 0.4324], [0.2145, 0.2098, 0.2496]),
    'Dogs': (120, [0.4765, 0.4516, 0.3911], [0.2490, 0.2435, 0.2479]),
    'Flowers': (102, [0.4329, 0.3820, 0.2965], [0.2828, 0.2333, 0.2615]),
}
CACHE_PROTOCOL = 'downstream_32x32_v1'


def locate(root, dataset):
    root = Path(root)
    candidates = {
        'CIFAR100': [(root, 'cifar-100-python/train')],
        'Aircraft': [(root / 'fgvc-aircraft-2013b/data', 'variants.txt'),
                     (root / 'aircraft/fgvc-aircraft-2013b/data', 'variants.txt')],
        'CUB2011': [(root / 'CUB_200_2011', 'images.txt'),
                    (root / 'cub2011/CUB_200_2011', 'images.txt')],
        'Dogs': [(root / 'dogs', 'train_list.mat'), (root / 'stanford_dogs', 'train_list.mat'),
                 (root, 'train_list.mat')],
        'Flowers': [(root / 'flowers-102', 'setid.mat'),
                    (root / 'flowers/flowers-102', 'setid.mat')],
    }[dataset]
    matches = [folder for folder, marker in candidates if (folder / marker).is_file()]
    if len(matches) > 1:
        raise ValueError(f'Ambiguous {dataset} roots: {matches}; keep one canonical copy.')
    if not matches:
        raise FileNotFoundError(f'{dataset} is not staged. Expected {candidates[0][0] / candidates[0][1]}')
    return matches[0]


def read_pairs(path):
    return {int(key): value for key, value in (line.strip().split(maxsplit=1) for line in Path(path).read_text().splitlines() if line.strip())}


def records(root, dataset, training):
    """Read official split membership; never use test data for selection/training."""
    folder = locate(root, dataset)
    if dataset == 'Aircraft':
        classes = (folder / 'variants.txt').read_text().splitlines()
        mapping = {name.strip(): index for index, name in enumerate(classes)}
        split = 'trainval' if training else 'test'
        samples = []
        for line in (folder / f'images_variant_{split}.txt').read_text().splitlines():
            name, label = line.split(maxsplit=1)
            samples.append((folder / 'images' / f'{name}.jpg', mapping[label.strip()]))
        return samples
    if dataset == 'CUB2011':
        names = read_pairs(folder / 'images.txt')
        labels = read_pairs(folder / 'image_class_labels.txt')
        splits = read_pairs(folder / 'train_test_split.txt')
        if names.keys() != labels.keys() or names.keys() != splits.keys():
            raise ValueError('CUB metadata image IDs disagree')
        return [(folder / 'images' / name, int(labels[index]) - 1)
                for index, name in names.items() if int(splits[index]) == int(training)]
    if dataset == 'Dogs':
        metadata = loadmat(folder / ('train_list.mat' if training else 'test_list.mat'))
        names = metadata.get('file_list', metadata.get('annotation_list'))
        if names is None:
            raise ValueError('Dogs split has neither file_list nor annotation_list')
        labels = metadata['labels'].reshape(-1)
        if len(names) != len(labels):
            raise ValueError('Dogs split image/label counts disagree')
        samples = []
        for entry, label in zip(names.reshape(-1), labels):
            while isinstance(entry, np.ndarray):
                entry = entry.item() if entry.size == 1 else ''.join(entry.tolist())
            name = str(entry)
            if not name.lower().endswith('.jpg'):
                name += '.jpg'
            samples.append((folder / 'Images' / name, int(label) - 1))
        return samples
    if dataset == 'Flowers':
        # Follow existing repository protocol: official train only, not train+val.
        splits = loadmat(folder / 'setid.mat', squeeze_me=True)
        ids = np.atleast_1d(splits['trnid' if training else 'tstid']).astype(int)
        labels = np.atleast_1d(loadmat(folder / 'imagelabels.mat', squeeze_me=True)['labels'])
        return [(folder / 'jpg' / f'image_{index:05d}.jpg', int(labels[index - 1]) - 1) for index in ids]
    raise ValueError(f'{dataset} does not use image-file records')


def read_split(root, dataset, training):
    if dataset == 'CIFAR100':
        data = CIFAR100(str(locate(root, dataset)), train=training, download=False)
        return torch.from_numpy(data.data).permute(0, 3, 1, 2).contiguous(), torch.tensor(data.targets), []
    samples = records(root, dataset, training)
    images = torch.empty((len(samples), 3, 32, 32), dtype=torch.uint8)
    for index, (path, _) in enumerate(samples):
        with Image.open(path) as image:
            image = image.convert('RGB').resize((32, 32), Image.Resampling.LANCZOS)
            images[index] = torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1)
    return images, torch.tensor([label for _, label in samples]), [str(path) for path, _ in samples]


def prepare_target(root, dataset, destination):
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    train_images, train_labels, train_names = read_split(root, dataset, True)
    test_images, test_labels, test_names = read_split(root, dataset, False)
    classes, mean, std = SPECS[dataset]
    if train_names and set(train_names) & set(test_names):
        raise ValueError(f'{dataset}: train/test images overlap')
    for labels in (train_labels, test_labels):
        if not len(labels) or labels.min() < 0 or labels.max() >= classes:
            raise ValueError(f'{dataset}: invalid class labels')
    if len(train_labels.unique()) != classes:
        raise ValueError(f'{dataset}: expected {classes} training classes, found {len(train_labels.unique())}')
    value = dict(protocol=CACHE_PROTOCOL, dataset=dataset, num_classes=classes, mean=mean, std=std,
                 train_images=train_images, train_labels=train_labels,
                 test_images=test_images, test_labels=test_labels,
                 train_paths=train_names, test_paths=test_names)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix('.tmp')
    torch.save(value, temporary)
    temporary.replace(destination)
    return dict(dataset=dataset, train_images=len(train_labels), test_images=len(test_labels), num_classes=classes)


def load_target(path, device, expected_dataset):
    cache = torch.load(path, map_location='cpu', weights_only=True)
    if cache['protocol'] != CACHE_PROTOCOL or cache['dataset'] != expected_dataset:
        raise ValueError('Wrong target cache protocol or dataset')
    mean = torch.tensor(cache['mean'], device=device).view(1, 3, 1, 1)
    std = torch.tensor(cache['std'], device=device).view(1, 3, 1, 1)
    data = []
    for split in ('train', 'test'):
        images = cache[f'{split}_images'].to(device=device, dtype=torch.float32).div_(255)
        images.sub_(mean).div_(std)
        data.extend((images, cache[f'{split}_labels'].to(device)))
    return (*data, cache['num_classes'])


def select_labels(labels, percentage, seed, policy='exact'):
    """Nested class-balanced prefixes with exact overall budgets, even below C."""
    labels = np.asarray(labels)
    classes = np.unique(labels)
    count = max(1, int(len(labels) * percentage / 100))
    if policy == 'at_least_one_per_class':
        count = max(count, len(classes))
    elif policy != 'exact':
        raise ValueError(policy)
    if count > len(labels):
        raise ValueError('Label budget exceeds the target training split')
    generator = np.random.default_rng(seed + 5000)
    class_order = generator.permutation(classes)
    pools = [generator.permutation(np.flatnonzero(labels == label)) for label in class_order]
    chosen = []
    level = 0
    while len(chosen) < count:
        for pool in pools:
            if level < len(pool):
                chosen.append(int(pool[level]))
                if len(chosen) == count:
                    break
        level += 1
    return np.asarray(chosen, dtype=np.int64)
