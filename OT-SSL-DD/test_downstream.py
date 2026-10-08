"""Offline fixtures and transfer correctness checks; execute within Slurm."""

import argparse
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
from scipy.io import savemat
import torch
from torch import nn

import benchmark_downstream as downstream
import downstream_data as data


class TinyEncoder(nn.Module):
    def __init__(self, classes=10):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 1)
        self.classifier = nn.Linear(4, classes)

    def embed(self, images):
        return self.conv(images).mean((2, 3))

    def forward(self, images):
        return self.classifier(self.embed(images))


def fixture(root, name):
    """Two classes, official metadata format, distinct train/test images."""
    root = Path(root)
    if name == 'Aircraft':
        folder = root / 'fgvc-aircraft-2013b/data'
        folder.mkdir(parents=True)
        (folder / 'variants.txt').write_text('Variant A\nVariant B\n')
        (folder / 'images_variant_trainval.txt').write_text('0001 Variant A\n0002 Variant B\n')
        (folder / 'images_variant_test.txt').write_text('0003 Variant A\n0004 Variant B\n')
        paths = [folder / f'images/{i:04d}.jpg' for i in range(1, 5)]
    elif name == 'CUB2011':
        folder = root / 'CUB_200_2011'
        folder.mkdir()
        (folder / 'images.txt').write_text('1 a/1.jpg\n2 b/2.jpg\n3 a/3.jpg\n4 b/4.jpg\n')
        (folder / 'image_class_labels.txt').write_text('1 1\n2 2\n3 1\n4 2\n')
        (folder / 'train_test_split.txt').write_text('1 1\n2 1\n3 0\n4 0\n')
        paths = [folder / 'images' / path for path in ('a/1.jpg', 'b/2.jpg', 'a/3.jpg', 'b/4.jpg')]
    elif name == 'Dogs':
        folder = root / 'dogs'
        folder.mkdir()
        savemat(folder / 'train_list.mat', dict(annotation_list=np.array([['a/1'], ['b/2']], dtype=object), labels=np.array([[1], [2]])))
        savemat(folder / 'test_list.mat', dict(file_list=np.array([['a/3.jpg'], ['b/4.jpg']], dtype=object), labels=np.array([[1], [2]])))
        paths = [folder / 'Images' / path for path in ('a/1.jpg', 'b/2.jpg', 'a/3.jpg', 'b/4.jpg')]
    elif name == 'Flowers':
        folder = root / 'flowers-102'
        folder.mkdir()
        savemat(folder / 'setid.mat', dict(trnid=np.array([1, 2]), valid=np.array([5, 6]), tstid=np.array([3, 4])))
        savemat(folder / 'imagelabels.mat', dict(labels=np.array([1, 2, 1, 2, 1, 2])))
        paths = [folder / f'jpg/image_{i:05d}.jpg' for i in range(1, 5)]
    else:
        raise ValueError(name)
    for i, path in enumerate(paths):
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new('RGB', (45, 27), (30 + i, 80, 120)).save(path)


class DownstreamTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)

    def test_exact_budgets_small_classes_resampling_and_nesting(self):
        for classes, per_class in ((102, 10), (200, 30), (100, 500), (120, 100)):
            with self.subTest(classes=classes):
                labels = np.repeat(np.arange(classes), per_class)
                small = data.select_labels(labels, 1, 0)
                large = data.select_labels(labels, 5, 0)
                self.assertEqual(len(small), len(labels) // 100)
                self.assertEqual(len(large), len(labels) * 5 // 100)
                self.assertEqual(len(set(small)), len(small))
                self.assertTrue(set(small).issubset(large))
                np.testing.assert_array_equal(small, data.select_labels(labels, 1, 0))
                self.assertFalse(np.array_equal(small, data.select_labels(labels, 1, 1)))
        labels = np.repeat(np.arange(102), 10)
        represented = {tuple(np.unique(labels[data.select_labels(labels, 1, seed)])) for seed in range(15)}
        self.assertEqual(len(represented), 15)

    def test_all_image_dataset_formats_and_cache(self):
        for name in ('Aircraft', 'CUB2011', 'Dogs', 'Flowers'):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                fixture(root, name)
                classes, mean, std = data.SPECS[name]
                with patch.dict(data.SPECS, {name: (2, mean, std)}):
                    stats = data.prepare_target(root, name, root / 'packed.pt')
                self.assertEqual(stats['train_images'], 2)
                images, labels, test, test_labels, classes = data.load_target(root / 'packed.pt', 'cpu', name)
                self.assertEqual(tuple(images.shape), (2, 3, 32, 32))
                self.assertEqual(classes, 2)
                self.assertEqual(labels.tolist(), [0, 1])
                self.assertEqual(test_labels.tolist(), [0, 1])
                self.assertTrue(torch.isfinite(images).all())
                with self.assertRaises(FileExistsError):
                    data.prepare_target(root, name, root / 'packed.pt')

    def test_cifar100_never_downloads_and_preserves_100_classes(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / 'cifar-100-python'
            folder.mkdir()
            (folder / 'train').touch()
            fake = argparse.Namespace(data=np.zeros((100, 32, 32, 3), dtype=np.uint8), targets=list(range(100)))
            with patch.object(data, 'CIFAR100', return_value=fake) as constructor:
                data.prepare_target(directory, 'CIFAR100', Path(directory) / 'packed.pt')
            self.assertEqual(constructor.call_count, 2)
            self.assertTrue(all(call.kwargs['download'] is False for call in constructor.call_args_list))
            *_, classes = data.load_target(Path(directory) / 'packed.pt', 'cpu', 'CIFAR100')
            self.assertEqual(classes, 100)

    def test_probe_keeps_backbone_frozen_with_200_way_head(self):
        network = TinyEncoder()
        original = {key: value.clone() for key, value in network.state_dict().items()}
        images = torch.randn(8, 3, 4, 4)
        labels = torch.tensor([0, 50, 100, 199, 10, 80, 160, 198])
        score = downstream.probe(network, images, labels, images, labels, 200, 2, 0)
        self.assertTrue(0 <= score <= 100)
        self.assertFalse(network.training)
        self.assertTrue(all(not p.requires_grad for p in network.parameters()))
        for key, value in network.state_dict().items():
            torch.testing.assert_close(value, original[key], rtol=0, atol=0)

    def test_supervised_target_baseline_trains_the_encoder(self):
        network = TinyEncoder(102)
        original = network.conv.weight.detach().clone()
        images = torch.randn(8, 3, 4, 4)
        downstream.source.train_supervised(network, images, torch.arange(8) + 90, 2, 0)
        self.assertFalse(torch.equal(original, network.conv.weight))

    def test_checkpoint_round_trip_and_wrong_configuration_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            images = torch.randn(1000, 3, 4, 4)
            args = argparse.Namespace(output=str(Path(directory) / 'encoders'), data_path='unused',
                device='cpu', runs=1, model='ConvNet', method='random', ssl_method='simclr',
                subset_percentage=1, ssl_epochs=1, selection_dir=None)
            with patch.object(downstream.source, 'load_data', return_value=(images, None, None, None)), \
                 patch.object(downstream.source, 'get_network', side_effect=lambda *args, **kwargs: TinyEncoder()):
                downstream.pretrain(args)
                args.encoder_dir = args.output
                loaded = downstream.load_encoder(args, 0)
                checkpoint = torch.load(Path(args.output) / 'seed_000.pt', weights_only=True)
                for key, value in loaded.state_dict().items():
                    torch.testing.assert_close(value, checkpoint['state_dict'][key])
                args.ssl_method = 'barlowtwins'
                with self.assertRaises(ValueError):
                    downstream.load_encoder(args, 0)

    def test_reporting_preserves_cifar_and_unmeasured_methods(self):
        path = Path(__file__).resolve().parents[1] / 'benchmarks/downstream/report.py'
        spec = importlib.util.spec_from_file_location('downstream_report', path)
        report = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(report)
        template = '\n'.join([
            r'\label{tab:cifar10-convnet-simclr}',
            r'\multicolumn{8}{c}{\textbf{1\% Pretraining Subset Size}} \\',
            r'\multirow{7}{*}{1\%}',
            r'& Random Subset & $39.02 \pm 1.07$ & & & & & \\',
            r'& KRR-ST & & & & & & \\',
            r'& \textbf{MKDT} & & & & & & \\',
            r'& ours & & & & & & \\',
        ])
        results = {('CIFAR100', 'ConvNet', 'simclr', 'random', 1, 1): r'$12.34 \pm 0.56$'}
        text = report.fill_latex(template, results)
        self.assertIn(r'$39.02 \pm 1.07$', text)
        self.assertIn(r'$12.34 \pm 0.56$', text)
        for name in ('KRR-ST', r'\textbf{MKDT}', 'ours'):
            self.assertIn(f'& {name} & & & & & &', text)
        self.assertEqual(text.count(r'\\'), template.count(r'\\'))


if __name__ == '__main__':
    unittest.main()
