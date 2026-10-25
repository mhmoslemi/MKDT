"""Small CPU checks; run only inside an allocation, not on the login node."""
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest

import torch

spec = importlib.util.spec_from_file_location('cifar_finish_worker', Path(__file__).with_name('worker.py'))
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


class ContinueTraining(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.parent, self.folder = root / 'old/seed_001', root / 'new/ssl_seed_001'
        self.parent.mkdir(parents=True)
        self.folder.mkdir(parents=True)
        signature = {'source': 'CIFAR10', 'model': 'ResNet18', 'subset': 2, 'ssl': 'simclr',
                     'source_fingerprint': 'same-data', 'synthetic': {'images': 'same-images'},
                     'code_hashes': {'trainer': 'same-code'},
                     'protocol': {'ssl': {'subset_epochs': 2000, 'batch_size': 4},
                                  'subset_seeds': [0, 1]}}
        identity = {'signature': worker.digest_json(signature), 'seed': 1, 'selection_method': 'ours'}
        self.old = {'signature': signature, 'identity': identity}
        self.new = copy.deepcopy(self.old)
        self.new['signature']['protocol']['ssl']['subset_epochs'] = 3000
        self.new['identity']['signature'] = worker.digest_json(self.new['signature'])
        self.state = {'identity': identity, 'epoch': 2000, 'steps': 2000, 'steps_per_epoch': 1,
                      'model': {'encoder': torch.tensor([1., 2.]), 'projector': torch.tensor([3., 4.])},
                      'optimizer': {'momentum': torch.tensor([5., 6.])},
                      'rng': {'torch': torch.tensor([7, 8], dtype=torch.uint8)}}
        worker.atomic_json(self.parent / 'config.json', self.old)
        worker.atomic_save(self.parent / 'training.pt', self.state)

    def tearDown(self):
        self.temporary.cleanup()

    def test_preserves_weights_projector_optimizer_rng_and_original(self):
        original_hash = worker.file_hash(self.parent / 'training.pt')
        worker.copy_continuation(self.parent, self.folder, self.new, 4)
        saved = torch.load(self.folder / 'training.pt', map_location='cpu', weights_only=True)
        self.assertEqual(saved['identity'], self.new['identity'])
        for key in ('model', 'optimizer', 'rng'):
            torch.testing.assert_close(saved[key], self.state[key])
        for key in ('epoch', 'steps', 'steps_per_epoch'):
            self.assertEqual(saved[key], self.state[key])
        self.assertEqual(worker.file_hash(self.parent / 'training.pt'), original_hash)
        self.assertEqual(worker.read(self.folder / 'continuation.json')['resumed_epoch'], 2000)
        # A second invocation must retain newer destination progress.
        saved['epoch'] = saved['steps'] = 2500
        worker.atomic_save(self.folder / 'training.pt', saved)
        worker.copy_continuation(self.parent, self.folder, self.new, 4)
        self.assertEqual(torch.load(self.folder / 'training.pt', weights_only=True)['epoch'], 2500)

    def test_rejects_different_images_instead_of_restarting(self):
        self.new['signature']['synthetic']['images'] = 'different-images'
        with self.assertRaisesRegex(ValueError, 'different images'):
            worker.copy_continuation(self.parent, self.folder, self.new, 4)
        self.assertFalse((self.folder / 'training.pt').exists())

    def test_encoder_only_cannot_masquerade_as_full_resume(self):
        (self.parent / 'training.pt').unlink()
        worker.atomic_json(self.parent / 'ssl_complete.json', {'complete': True, 'epochs': 2000})
        with self.assertRaisesRegex(FileNotFoundError, 'projector, optimizer and RNG'):
            worker.copy_continuation(self.parent, self.folder, self.new, 4)
        self.assertFalse((self.folder / 'training.pt').exists())


class Cifar100Tasks(unittest.TestCase):
    def test_main_only_scope_and_schedule(self):
        path = Path(__file__).with_name('worker_cifar100.py')
        spec = importlib.util.spec_from_file_location('cifar100_finish_worker', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        manifest = path.with_name('cifar100_tasks.tsv')
        with manifest.open() as stream:
            tasks = list(worker.csv.DictReader(stream, delimiter='\t'))
        self.assertEqual(len(tasks), 60)
        self.assertEqual(len({row['task'] for row in tasks}), 60)
        for row in tasks:
            task = module.load_task(manifest, row['task'])
            self.assertEqual(task['source'], 'CIFAR100')
            self.assertEqual(task['scope'], 'main')
            self.assertEqual(task['output_dir'], task['checkpoint_dir'])
        with self.assertRaisesRegex(ValueError, 'Unsupported CIFAR100'):
            module.load_task(path.with_name('tasks.tsv'), 'ours_CIFAR10_ConvNet_p0p2_simclr_s0')


if __name__ == '__main__':
    unittest.main()
