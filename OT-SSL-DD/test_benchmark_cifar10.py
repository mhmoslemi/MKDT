"""Run with unittest inside a Slurm allocation; no external test framework needed."""

import argparse
import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

import benchmark_cifar10 as bench


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 1)
        self.classifier = nn.Linear(4, 10)

    def embed(self, images):
        return self.conv(images).mean((2, 3))

    def forward(self, images):
        return self.classifier(self.embed(images))


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        bench.seed_all(7)

    def test_labels_exact_balanced_nested_and_independent(self):
        labels = np.repeat(np.arange(10), 5000)
        one = bench.labeled_indices(labels, 1, 0)
        five = bench.labeled_indices(labels, 5, 0)
        self.assertEqual(len(one), 500)
        self.assertEqual(len(five), 2500)
        np.testing.assert_array_equal(np.bincount(labels[one]), np.full(10, 50))
        self.assertTrue(set(one).issubset(five))
        np.testing.assert_array_equal(one, bench.labeled_indices(labels, 1, 0))
        self.assertFalse(np.array_equal(one, bench.labeled_indices(labels, 1, 1)))

    def test_kmeans_distinct_representatives_even_for_duplicate_features(self):
        features = torch.tensor([[0., 0.], [0., 0.], [10., 10.], [10., 10.]])
        indices = bench.kmeans_indices(features, 4, 0, iterations=3, chunk_size=2)
        self.assertEqual(sorted(indices), [0, 1, 2, 3])
        unique_features = torch.tensor([[0., 0.], [1., 0.], [20., 0.], [21., 0.]])
        selected = bench.kmeans_indices(unique_features, 2, 0, iterations=20, chunk_size=2)
        self.assertEqual(len(set(selected)), 2)
        self.assertTrue(any(i < 2 for i in selected) and any(i >= 2 for i in selected))

    def test_ssl_losses_are_finite_and_differentiable(self):
        for method in ('simclr', 'barlowtwins'):
            with self.subTest(method=method):
                first = torch.randn(4, 8, requires_grad=True)
                second = torch.randn(4, 8, requires_grad=True)
                loss = bench.ssl_loss(first, second, method)
                loss.backward()
                self.assertTrue(torch.isfinite(loss))
                self.assertTrue(torch.isfinite(first.grad).all())
                self.assertGreater(first.grad.abs().sum(), 0)

    def test_linear_probe_keeps_encoder_unchanged(self):
        network = TinyEncoder()
        before = {key: value.clone() for key, value in network.state_dict().items()}
        images = torch.randn(20, 3, 4, 4)
        labels = torch.arange(20) % 10
        score = bench.linear_probe(network, images, labels, images, labels, 2, 0)
        self.assertTrue(0 <= score <= 100)
        self.assertTrue(all(not p.requires_grad for p in network.parameters()))
        for key, value in network.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)

    def test_no_pretraining_updates_backbone(self):
        network = TinyEncoder()
        before = network.conv.weight.detach().clone()
        bench.train_supervised(network, torch.randn(20, 3, 4, 4), torch.arange(20) % 10, 2, 0)
        self.assertFalse(torch.equal(before, network.conv.weight))

    def test_two_independent_runs_save_correct_statistics_and_refuse_overwrite(self):
        images = torch.randn(1000, 3, 4, 4)
        labels = torch.arange(1000) % 10
        def factory(*args, seed, **kwargs):
            bench.seed_all(seed)
            return TinyEncoder()
        with tempfile.TemporaryDirectory() as directory:
            args = argparse.Namespace(output=str(Path(directory) / 'runs'), data_path='unused',
                device='cpu', seed_start=0, runs=2, model='ConvNet', method='random',
                ssl_method='simclr', subset_percentage=1, label_percentage=5,
                ssl_epochs=1, probe_epochs=1, selection_dir=None)
            with patch.object(bench, 'load_data', return_value=(images, labels, images[:100], labels[:100])), \
                 patch.object(bench, 'get_network', side_effect=factory):
                bench.benchmark(args)
                with self.assertRaises(FileExistsError):
                    bench.benchmark(args)
            output = Path(args.output)
            first = json.loads((output / 'seed_000.json').read_text())
            second = json.loads((output / 'seed_001.json').read_text())
            summary = json.loads((output / 'summary.json').read_text())
            self.assertNotEqual(first['subset_indices'], second['subset_indices'])
            self.assertNotEqual(first['labeled_indices'], second['labeled_indices'])
            scores = [first['accuracy_percent'], second['accuracy_percent']]
            self.assertAlmostEqual(summary['mean_percent'], float(np.mean(scores)))
            self.assertAlmostEqual(summary['std_percent'], float(np.std(scores, ddof=1)))

    def test_report_excludes_partial_results(self):
        report_path = Path(__file__).resolve().parents[1] / 'benchmarks/cifar10/report.py'
        spec = importlib.util.spec_from_file_location('benchmark_report', report_path)
        report = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(report)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'configs.tsv').write_text('config_id\tmodel\tssl_method\tmethod\tsubset_percentage\tlabel_percentage\n'
                'test\tConvNet\tsimclr\trandom\t1\t1\n')
            run = root / 'test/job_1'
            run.mkdir(parents=True)
            summary = dict(complete=False, completed_runs=2, seeds=[0, 1], mean_percent=99.9, std_percent=0.1)
            bench.write_json(run / 'summary.json', summary)
            report.render(root, root / 'tables.md')
            content = (root / 'tables.md').read_text()
            self.assertIn('Incomplete (2/15)', content)
            self.assertNotIn('99.90', content)
            summary.update(complete=True, completed_runs=15, seeds=list(range(15)))
            bench.write_json(run / 'summary.json', summary)
            report.render(root, root / 'tables.md')
            self.assertIn('99.90 ± 0.10', (root / 'tables.md').read_text())


if __name__ == '__main__':
    unittest.main()
