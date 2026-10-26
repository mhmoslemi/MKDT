"""Run on the allocated GPU: compare loss/gradients to pinned official source."""
import ast
import json
import math
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import torch
from torch import nn
from torch.nn import functional as F

from common import atomic_json, build_encoder, file_hash, seed_all
from a2mim_teacher.model import A2MIM, block_mask, learning_rate, optimizer_for, reconstruction_loss


def definition(filename, name, namespace):
    tree = ast.parse((HERE / 'reference' / filename).read_text())
    node = next(node for node in tree.body if getattr(node, 'name', None) == name)
    node.decorator_list = []
    exec(compile(ast.Module(body=[node], type_ignores=[]), filename, 'exec'), namespace)
    return namespace[name]


def main():
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise RuntimeError('Run this check inside the requested GPU training job')
    torch.set_num_threads(1)
    seed_all(123)
    for filename, checksum in json.loads((HERE / 'sources.json').read_text())['sha256'].items():
        assert file_hash(HERE / 'reference' / filename) == checksum, filename
    namespace = {'torch': torch, 'nn': nn, 'F': F, 'fft': torch.fft}
    focal_l1 = definition('regression_loss.py.txt', 'focal_l1_loss', namespace)
    official_fft = definition('focal_loss.py.txt', 'FocalFrequencyLoss', namespace)

    class BaseModule(nn.Module):
        def __init__(self, init_cfg=None):
            super().__init__()

    class Regression(nn.Module):
        def forward(self, pred, target, reduction_override):
            return focal_l1(pred, target, reduction=reduction_override,
                            alpha=0.2, gamma=1.0, activate='sigmoid', residual=False)

    def build_loss(config):
        if config['type'] == 'FocalFrequencyLoss':
            return official_fft(**{k: v for k, v in config.items() if k != 'type'})
        return Regression()

    namespace.update(BaseModule=BaseModule, build_loss=build_loss,
                     print_log=lambda *args: None)
    official_head = definition('mim_head.py.txt', 'A2MIMHead', namespace)(
        loss={'type': 'RegressionLoss', 'mode': 'focal_l1_loss'},
        fft_weight=0.5, fft_focal=True, fft_unmask_weight=0.001,
        fft_unmask_replace='mixed').cuda()
    for size in (32, 64):
        target = torch.randn(2, 3, size, size, device='cuda')
        mask = block_mask(2, size, 8, target.device)
        expected_count = math.ceil((size // 8)**2 * 0.6) * 8**2
        assert torch.all(mask.sum((1, 2, 3)) == expected_count)
        prediction = torch.randn_like(target, requires_grad=True)
        ours = reconstruction_loss(prediction, target, mask)[0]
        official = official_head(target, prediction, mask[:, 0, ::4, ::4])['loss']
        torch.testing.assert_close(ours, official, atol=2e-6, rtol=2e-5)
        ours_grad = torch.autograd.grad(ours, prediction)[0]
        official_grad = torch.autograd.grad(official, prediction)[0]
        torch.testing.assert_close(ours_grad, official_grad, atol=2e-7, rtol=2e-5)
        # Exercise the actual native-size BN backbone, decoder and one AdamW update.
        model = A2MIM(size).cuda().train()
        optimizer = optimizer_for(model, 256)
        output = model(target, mask)
        assert output.shape == target.shape
        loss = reconstruction_loss(output, target, mask)[0]
        loss.backward()
        assert torch.isfinite(loss) and model.encoder.conv1.weight.grad.abs().sum() > 0
        assert model.mask_token.grad.abs().sum() > 0
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        optimizer.step()
        encoder, _ = build_encoder('ResNet18', size)
        encoder.load_state_dict(model.encoder.cpu().state_dict(), strict=True)
        encoder.eval()
        assert encoder(target.cpu()).shape == (2, 512)
        del optimizer, model, encoder
    assert learning_rate(0, 100, 300, 1.5e-4) == 0
    assert abs(learning_rate(1000, 100, 300, 1.5e-4) - 1.5e-4) < 1e-12
    assert abs(learning_rate(30000, 100, 300, 1.5e-4) - 1e-5) < 1e-12
    atomic_json(HERE / ('check_%s.json' % os.environ['SLURM_JOB_ID']), {
        'complete': True, 'official_loss_and_gradient_match': True,
        'native_sizes_checked': [32, 64], 'strict_encoder_load': True,
        'gpu': torch.cuda.get_device_name(), 'torch': str(torch.__version__)})
    print('CHECK PASSED: official A²MIM loss/gradients, native BN backbones, AdamW, encoder export.', flush=True)


if __name__ == '__main__':
    main()
