"""Small gradient/replay checks and measured GPU timing for the new distillation views."""
import ast
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'OT-SSL-DD'))
import method1
from simclr_dsa import SimCLRDSA


def main():
    for file in ('OT-SSL-DD/method1.py', 'OT-SSL-DD/simclr_dsa.py',
                 'Evaluate_benchamrk/mipa_fashion_singlechannel/worker.py'):
        ast.parse((ROOT / file).read_text(), filename=file)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    reference = method1.reference_modules()
    common = reference['common']
    prepared = json.loads((ROOT / 'Evaluate_benchamrk/fashionmnist/full/prepared.json').read_text())
    protocol = prepared['protocol']
    mean, std = protocol['normalization']['FashionMNIST']
    view = reference['data'].ssl_transform(protocol, 'FashionMNIST', 'simclr').transforms[0]
    assert tuple(view.transforms[0].scale) == tuple(protocol['ssl']['crop_scale'])
    assert view.transforms[1].p == protocol['ssl']['color_probability']
    assert view.transforms[2].p == protocol['ssl']['grayscale_probability']
    assert view.transforms[3].p == protocol['methods']['simclr']['blur_probabilities'][0]
    assert view.transforms[5].p == protocol['ssl']['flip_probability']
    for probability in (0., 0.5, 1.):
        augmenter = SimCLRDSA(view, mean, std, probability)
        for seed in range(12):
            gray = torch.rand(3, 1, 32, 32, device='cuda').requires_grad_(True)
            normalized = (gray.expand(-1, 3, -1, -1) - mean[0]) / std[0]
            rng = torch.get_rng_state().clone()
            output = augmenter(normalized, seed)
            assert torch.equal(torch.get_rng_state(), rng)
            assert torch.equal(output, augmenter(normalized, seed))
            assert torch.equal(output[:2], augmenter(normalized[:2], seed)), 'Siamese parameters changed with batch length'
            assert output.shape == (3, 3, 32, 32) and torch.isfinite(output).all()
            assert torch.allclose(output, output[:, :1].expand_as(output), atol=1e-5)
            gradient, = torch.autograd.grad(output.square().mean(), gray)
            assert gradient.shape == gray.shape and torch.isfinite(gradient).all() and gradient.abs().max() > 0
    print('CHECK PASSED: single/composed/mixed gradients, grayscale, seeded replay and SimCLR recipe settings', flush=True)

    args = SimpleNamespace(dataset='FashionMNIST', model='ResNet18BN', teacher_source='pretrained', teacher_seed=0,
        teacher_ssl='simclr', teacher_path=str(ROOT / 'Evaluate_benchamrk/fashionmnist/full/checkpoints'), device='cuda')
    teacher = method1.load_teacher(args, mean, std, (32, 32))
    encoder = method1.frozen_embedder(teacher)
    dataset = reference['_fashion_dataset'].DigitDataset(ROOT.parent / 'data', 'FashionMNIST', True)
    real = torch.stack([method1.transforms.functional.to_tensor(dataset[i][0]) for i in range(128)]).cuda()
    real = (real - torch.tensor(mean, device='cuda').view(1, 3, 1, 1)) / torch.tensor(std, device='cuda').view(1, 3, 1, 1)
    indices = np.random.default_rng(1000).permutation(len(dataset))[:120]
    initial = torch.stack([method1.transforms.functional.to_tensor(dataset[int(i)][0]) for i in indices]).cuda()
    gray = ((initial[:, :1] - mean[0]) / std[0]).detach().requires_grad_(True)
    optimizer = torch.optim.Adam([gray], lr=0.002)
    augmenter = SimCLRDSA(view, mean, std, 0.5)
    measured = []
    for step in range(55):
        torch.cuda.synchronize()
        started = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            first = encoder(augmenter(real, 100 + step * 2))
            second = encoder(augmenter(real, 101 + step * 2))
        rgb = gray.expand(-1, 3, -1, -1)
        syn_first = checkpoint(encoder, augmenter(rgb, 100 + step * 2), use_reentrant=False)
        syn_second = checkpoint(encoder, augmenter(rgb, 101 + step * 2), use_reentrant=False)
        loss = method1.bipartite_view_transition_loss(first, second, syn_first, syn_second, tau=0.2)[0]
        assert torch.isfinite(loss)
        loss.backward()
        assert torch.isfinite(gray.grad).all()
        optimizer.step()
        with torch.no_grad():
            gray.clamp_(-mean[0] / std[0], (1 - mean[0]) / std[0])
        torch.cuda.synchronize()
        if step >= 5:
            measured.append(time.monotonic() - started)
    assert all(parameter.grad is None for parameter in teacher.parameters())
    measured.sort()
    result = dict(complete=True, iteration_seconds_p95=measured[math.ceil(len(measured) * 0.95) - 1],
        samples=len(measured), augmentation_sha256=common.file_hash(ROOT / 'OT-SSL-DD/simclr_dsa.py'),
        method_sha256=common.file_hash(ROOT / 'OT-SSL-DD/method1.py'))
    common.atomic_json(Path(__file__).with_name('simclr_timing.json'), result)
    print('GPU TIMING:', json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
