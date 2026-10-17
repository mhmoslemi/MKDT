"""Shared provenance, atomic output, and existing repository backbones."""

import contextlib
import hashlib
import json
import os
from pathlib import Path
import random
import sys

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
import networks


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def code_hashes():
    files = sorted(HERE.glob('*.py')) + [ROOT / 'networks.py', ROOT / 'benchmarks/downstream_data.py']
    return {str(path.relative_to(ROOT)): file_hash(path) for path in files}


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(path)


def atomic_save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


@contextlib.contextmanager
def exclusive_run(folder):
    """OS lock releases on process exit, including crashes; no stale PID guessing."""
    import fcntl
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / '.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f'Another process is working in {folder}') from error
        yield


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def worker_seed(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


def rng_state():
    return {'python': random.getstate(), 'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state['python'])
    torch.set_rng_state(state['torch'])
    if state['cuda'] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda'])


def state_hash(model):
    digest = hashlib.sha256()
    for key, tensor in sorted(model.state_dict().items()):
        digest.update(key.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def build_encoder(model, size):
    """Use the root BN networks; classifier is removed, projector is separate."""
    if model == 'ResNet18':
        encoder = networks.ResNet18(channel=3, num_classes=1)
        dimension = encoder.classifier.in_features
        # Standard residual initialization; no AP substitution.
        for module in encoder.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        for module in encoder.modules():
            if isinstance(module, networks.BasicBlock):
                nn.init.zeros_(module.bn2.weight)
    elif model == 'VGG11':
        encoder = networks.VGG11BN(channel=3, num_classes=1)
        dimension = encoder.classifier.in_features
    elif model == 'ConvNet':
        encoder = networks.ConvNet(3, 1, 128, 4 if size == 64 else 3,
                                   'relu', 'batchnorm', 'avgpooling', (size, size))
        dimension = encoder.classifier.in_features
    else:
        raise ValueError(model)
    encoder.classifier = nn.Identity()
    if any(isinstance(module, (nn.GroupNorm, nn.InstanceNorm2d)) for module in encoder.modules()):
        raise RuntimeError('Benchmark accidentally selected a non-BN backbone')
    return encoder, dimension


def load_encoder(path, expected=None, device='cuda'):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if saved.get('kind') != 'ssl_reference_encoder' or not saved.get('complete'):
        raise ValueError(f'Not a complete benchmark encoder: {path}')
    if expected is not None and saved['identity'] != expected:
        raise ValueError(f'Checkpoint identity mismatch: {path}')
    model, dimension = build_encoder(saved['model'], saved['input_size'])
    model.load_state_dict(saved['encoder'], strict=True)
    if state_hash(model) != saved['encoder_sha256']:
        raise ValueError(f'Encoder state hash mismatch: {path}')
    return model.to(device).eval().requires_grad_(False), dimension, saved
