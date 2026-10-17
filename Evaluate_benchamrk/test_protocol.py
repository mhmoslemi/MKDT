"""Numerical and lifecycle tests. Run check.sh in a permitted compute environment."""

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common
import data
import evaluate
import plan
import run
import train_ssl


class LossTests(unittest.TestCase):
    def test_simclr_matches_explicit_positive_denominator_and_gradients(self):
        torch.manual_seed(42)
        first = torch.randn(5, 9, requires_grad=True)
        second = torch.randn(5, 9, requires_grad=True)
        actual = train_ssl.simclr_loss(first, second, 0.2)
        z = F.normalize(torch.cat([first, second]), dim=1)
        reference = []
        for index in range(10):
            logits = z[index] @ z.T / 0.2
            reference.append(-logits[(index + 5) % 10] + torch.logsumexp(
                logits[torch.arange(10) != index], 0))
        expected = torch.stack(reference).mean()
        torch.testing.assert_close(actual, expected)
        expected_grads = torch.autograd.grad(expected, (first, second), retain_graph=True)
        actual_grads = torch.autograd.grad(actual, (first, second))
        for a, b in zip(actual_grads, expected_grads):
            torch.testing.assert_close(a, b)
        torch.testing.assert_close(actual, train_ssl.simclr_loss(second, first, 0.2))

    def test_barlow_matches_official_batchnorm_correlation_and_gradients(self):
        torch.manual_seed(17)
        a = torch.randn(7, 11, requires_grad=True)
        b = torch.randn(7, 11, requires_grad=True)
        actual = train_ssl.barlow_loss(a, b)
        normalized_a = F.batch_norm(a, None, None, training=True, eps=1e-5)
        normalized_b = F.batch_norm(b, None, None, training=True, eps=1e-5)
        correlation = normalized_a.T @ normalized_b / 7
        mask = torch.eye(11, dtype=torch.bool)
        expected = 0.1 * ((correlation[mask] - 1).square().sum() + 0.0051 * correlation[~mask].square().sum())
        torch.testing.assert_close(actual, expected)
        ref = torch.autograd.grad(expected, (a, b), retain_graph=True)
        got = torch.autograd.grad(actual, (a, b))
        for x, y in zip(got, ref):
            torch.testing.assert_close(x, y, atol=1e-6, rtol=1e-5)

    def test_single_image_batch_is_rejected(self):
        with self.assertRaises(ValueError):
            train_ssl.simclr_loss(torch.ones(1, 4), torch.ones(1, 4), 0.2)
        with self.assertRaises(ValueError):
            train_ssl.barlow_loss(torch.ones(1, 4), torch.ones(1, 4))

    def test_lars_bias_exclusion_and_momentum(self):
        cfg = json.loads((HERE / 'protocol.json').read_text())['ssl']
        weight = nn.Parameter(torch.tensor([[1., 2.], [3., 4.]]))
        bias = nn.Parameter(torch.tensor([1., 2.]))
        optimizer = train_ssl.LARS([weight, bias], 0.3, cfg)
        original = weight.detach().clone()
        gradient = torch.ones_like(weight) * 0.2
        expected_trust = cfg['lars_eta'] * original.norm() / (
            gradient.norm() + cfg['weight_decay'] * original.norm() + cfg['lars_eps'])
        expected_trust = min(float(expected_trust) / 0.3, 1.)
        expected = original - 0.3 * (gradient + cfg['weight_decay'] * original) * expected_trust
        weight.grad, bias.grad = gradient, torch.ones_like(bias)
        optimizer.step()
        torch.testing.assert_close(weight, expected)
        torch.testing.assert_close(bias, torch.tensor([0.7, 1.7]))
        bias.grad = torch.ones_like(bias)
        optimizer.step()
        torch.testing.assert_close(bias, torch.tensor([0.13, 1.13]))

    def test_schedule_warmup_peak_and_final(self):
        values = [train_ssl.cosine_lr(step, 100, 0.3, 10, 3e-5) for step in range(100)]
        self.assertAlmostEqual(values[0], 3e-5)
        self.assertAlmostEqual(values[9], 0.3)
        self.assertAlmostEqual(values[-1], 0.)
        self.assertTrue(all(a >= b for a, b in zip(values[10:], values[11:])))


class DataTests(unittest.TestCase):
    def test_exact_nested_labels_and_missing_class_reporting_basis(self):
        class Fake:
            targets = np.repeat(np.arange(102), 10)
        one = data.label_indices(Fake(), 1, 8)
        five = data.label_indices(Fake(), 5, 8)
        self.assertEqual(len(one), 10)
        self.assertEqual(len(five), 51)
        self.assertTrue(set(one) <= set(five))
        self.assertEqual(len(set(Fake.targets[one])), 10)

    def test_unique_kmeans_exemplars_even_when_features_identical(self):
        selected = data.kmeans_indices(torch.ones(20, 4), 7, 2, 2, 5)
        self.assertEqual(len(selected), 7)
        self.assertEqual(len(set(selected)), 7)

    def test_native_test_pixels_unchanged_and_composed_ssl_views(self):
        protocol = json.loads((HERE / 'protocol.json').read_text())
        image = Image.fromarray(np.random.default_rng(1).integers(0, 256, (32, 32, 3), dtype=np.uint8))
        self.assertIs(data.ProbeCrop(32, 8 / 7)(image), image)
        for method in protocol['methods']:
            transform = data.ssl_transform(protocol, 'CIFAR10', method)
            first, second = transform(image)
            self.assertEqual(tuple(first.shape), (3, 32, 32))
            self.assertFalse(torch.equal(first, second))
            for view in transform.transforms:
                self.assertTrue(any(isinstance(item, data.T.RandomResizedCrop) for item in view.transforms))
                self.assertTrue(any(isinstance(item, data.T.RandomGrayscale) for item in view.transforms))
                self.assertIsInstance(view.transforms[-1], data.T.Normalize)


class ArchitectureTests(unittest.TestCase):
    def test_bn_backbones_at_both_native_resolutions(self):
        torch.set_num_threads(2)
        for model in ('ConvNet', 'VGG11', 'ResNet18'):
            for size in (32, 64):
                encoder, dimension = common.build_encoder(model, size)
                encoder.eval()
                with torch.no_grad():
                    features = encoder(torch.zeros(2, 3, size, size))
                self.assertEqual(tuple(features.shape), (2, dimension))
                self.assertTrue(any(isinstance(m, nn.BatchNorm2d) for m in encoder.modules()))


def tiny_encoder(_name, _size):
    return nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.BatchNorm2d(4), nn.ReLU(),
                         nn.AdaptiveAvgPool2d(1), nn.Flatten()), 4


class FixtureDataset:
    """Known fake pixels: never mixed with real benchmark results."""
    def __init__(self, root, name, training):
        self.name, self.classes = name, 2
        self.targets = np.arange(100 if training else 10) % 2
        self.fingerprint = f'fixture-{name}-{training}'
        self.pixels = np.random.default_rng(93 if training else 95).integers(
            0, 256, (len(self.targets), 32, 32, 3), dtype=np.uint8)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return Image.fromarray(self.pixels[index]), int(self.targets[index])


class LifecycleTests(unittest.TestCase):
    def test_one_encoder_all_targets_cleanup_retention_and_restart(self):
        torch.set_num_threads(2)
        protocol = json.loads((HERE / 'protocol.json').read_text())
        protocol['full_seeds'] = protocol['subset_seeds'] = [0]
        protocol['ssl'].update(full_epochs=1, subset_epochs=1, warmup_epochs=0,
                               batch_size=4, checkpoint_every_epochs=1)
        protocol['evaluation']['epochs'] = 1
        protocol['methods']['barlowtwins']['projector'] = [8, 8, 8]
        protocol['methods']['simclr']['projector'] = [8, 4]
        with tempfile.TemporaryDirectory(prefix='test-', dir=HERE) as temporary:
            folder = Path(temporary)
            config = folder / 'protocol.json'
            config.write_text(json.dumps(protocol))
            with patch.object(run, 'RawDataset', FixtureDataset), \
                 patch.dict(run.SOURCE_COUNTS, {'CIFAR10': 100}), \
                 patch.object(common, 'build_encoder', tiny_encoder), \
                 patch.object(train_ssl, 'build_encoder', tiny_encoder), \
                 contextlib.redirect_stdout(io.StringIO()):
                for method, percentage in (('random', '5'), ('full', '100')):
                    args = ['run.py', '--source', 'CIFAR10', '--model', 'ResNet18',
                            '--method', method, '--ssl', 'barlowtwins', '--subset', percentage,
                            '--seeds', '0', '--protocol', str(config), '--output-root', str(folder / 'results'),
                            '--device', 'cpu', '--workers', '0', '--threads', '2']
                    with patch.object(sys, 'argv', args):
                        run.main()
                    group = folder / 'results/CIFAR10/ResNet18' / method / f'pt{percentage}/barlowtwins'
                    seed_folder = group / 'seed_000'
                    rows = [json.loads(path.read_text()) for path in (seed_folder / 'evaluations').glob('*.json')]
                    self.assertEqual(len(rows), 15 if method == 'full' else 14)
                    self.assertEqual(len({row['encoder_sha256'] for row in rows}), 1)
                    self.assertTrue(all(row['encoder_unchanged'] for row in rows))
                    self.assertEqual((seed_folder / 'encoder.pt').exists(), method == 'full')
                    self.assertFalse((seed_folder / 'training.pt').exists())
                    self.assertTrue(json.loads((group / 'summary.json').read_text())['complete'])
                    with patch.object(sys, 'argv', args), patch.object(run, 'train', side_effect=AssertionError('retrained')):
                        run.main()
                    # Corruption must not be silently accepted on a completed run.
                    victim = seed_folder / 'evaluations/CIFAR10_lbl1.json'
                    original = victim.read_text()
                    victim.write_text(original.replace('"complete": true', '"complete": false'))
                    with patch.object(sys, 'argv', args), self.assertRaises(ValueError):
                        run.main()
                    victim.write_text(original)

    def test_missing_synthetic_input_does_not_fallback(self):
        with tempfile.TemporaryDirectory(prefix='test-', dir=HERE) as temporary:
            path = Path(temporary) / 'inputs.json'
            path.write_text(json.dumps({'mkdt': {'CIFAR10': {'0.2': {'path': None}}}}))
            with self.assertRaises(FileNotFoundError):
                run.synthetic_entry(path, 'mkdt', 'CIFAR10', 0.2)


class CoverageTests(unittest.TestCase):
    def test_all_table_cells_and_seed_splits(self):
        protocol = json.loads((HERE / 'protocol.json').read_text())
        runners = plan.build_runners(protocol)
        cells = plan.coverage(runners)
        self.assertEqual(len(runners), 387)
        self.assertEqual(len(cells), 1890)
        for row in runners:
            self.assertEqual(len(row['seeds'].split(',')), 1 if row['method'] == 'full' else 10)
            self.assertEqual((HERE / row['script']).read_text(), plan.script_text(row))


if __name__ == '__main__':
    unittest.main()
