"""Pixel drift penalty: physical units, fixed target, and training integration."""

import sys
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import torch
from torch.utils.data import TensorDataset

import mina_IF


def test_pixel_mse_units_and_gradient_pull_toward_fixed_target():
    reference = torch.zeros(2, 3, 2, 2, dtype=torch.float64, requires_grad=True)
    std = torch.tensor([0.2, 0.3, 0.4], dtype=torch.float64).view(1, 3, 1, 1)
    # Equal physical displacement in every channel despite unequal normalization.
    images = (reference.detach() + 0.1 / std).requires_grad_(True)
    loss = mina_IF.pixel_mse_loss(images, reference, std)
    torch.testing.assert_close(loss, torch.tensor(0.01, dtype=torch.float64))
    loss.backward()
    assert reference.grad is None
    expected_gradient = 2 * (images.detach() - reference.detach()) * std.square() / images.numel()
    torch.testing.assert_close(images.grad, expected_gradient)
    assert mina_IF.pixel_mse_loss(images.detach() - images.grad, reference, std) < loss
    torch.testing.assert_close(
        mina_IF.pixel_mse_loss(reference, reference, std), torch.zeros_like(loss)
    )


def check_main_accumulates_negative_mi_and_one_positive_penalty(tmp_path, networks, weight):
    torch.set_num_threads(1)
    torch.manual_seed(8)
    real = torch.randn(4, 3, 2, 2) * 0.1
    initial = torch.randn(2, 3, 2, 2) * 0.1
    std = torch.tensor([0.2, 0.3, 0.4]).view(1, 3, 1, 1)
    captured = {}

    class Encoder(torch.nn.Module):
        def embed(self, images):
            return images.flatten(1)

    class Optimizer:
        def __init__(self, parameters, lr):
            self.image = list(parameters)[0]

        def zero_grad(self, set_to_none):
            self.image.grad = None
            # Introduce known drift so the MSE gradient is nonzero in this step.
            with torch.no_grad():
                self.image.add_(0.1)

        def step(self):
            captured['gradient'] = self.image.grad.detach().clone()

    class Scheduler:
        def __init__(self, *args, **kwargs):
            pass

        def step(self):
            pass

    argv = [
        'mina_IF.py', '--device', 'cpu', '--Iteration', '1', '--num_eval', '0',
        '--percentage', '50', '--batch_real', '4', '--random_models', 'ConvNet',
        '--num_random_networks', str(networks), '--pixel_mse_weight', str(weight),
        '--save_path', str(tmp_path / 'result'), '--data_path', str(tmp_path / 'data'),
    ]
    with ExitStack() as patches:
        replacements = [
            (mina_IF, 'get_dataset', lambda *args: (
                3, (2, 2), 2, None, [0.5] * 3, std.flatten().tolist(), TensorDataset(real), None, None
            )),
            (mina_IF, 'deepcluster_initialize', lambda *args, **kwargs: initial.clone()),
            (mina_IF, 'get_network', lambda *args, **kwargs: Encoder()),
            (mina_IF, 'make_diff_augmenter', lambda *args: lambda images, seed: images),
            (torch.optim, 'Adam', Optimizer),
            (torch.optim.lr_scheduler, 'CosineAnnealingLR', Scheduler),
            (sys, 'argv', argv),
        ]
        for obj, name, value in replacements:
            patches.enter_context(patch.object(obj, name, value))
        mina_IF.main()

    probe = (initial + 0.1).requires_grad_(True)
    negative_mi = mina_IF.information_theoretic_loss(real.flatten(1), real.flatten(1), probe.flatten(1), 0.2)
    mse = ((probe - initial) * std).square().mean()
    expected_total = negative_mi + weight * mse
    expected_gradient, = torch.autograd.grad(expected_total, probe)
    torch.testing.assert_close(captured['gradient'], expected_gradient, rtol=1e-4, atol=1e-7)
    saved = torch.load(next((tmp_path / 'result').glob('*.pt')), weights_only=True)
    torch.testing.assert_close(saved['pixel_mse_history'][0], mse.detach().double())
    torch.testing.assert_close(saved['total_loss_history'][0], expected_total.detach().double(), rtol=1e-4, atol=1e-7)
    assert saved['iic_params']['pixel_mse_weight'] == weight


class PixelMSETests(unittest.TestCase):
    def test_units_and_gradient(self):
        test_pixel_mse_units_and_gradient_pull_toward_fixed_target()

    def test_training_gradients_and_saved_losses(self):
        for networks in (1, 5):
            for weight in (0.0, 2.5):
                with self.subTest(networks=networks, weight=weight), tempfile.TemporaryDirectory() as directory:
                    check_main_accumulates_negative_mi_and_one_positive_penalty(Path(directory), networks, weight)


if __name__ == '__main__':
    unittest.main()
