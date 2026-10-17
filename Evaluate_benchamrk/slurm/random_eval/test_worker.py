"""Small synthetic fixtures, run only inside Slurm; never benchmark scores."""
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import worker
import common
from report import report
from test_protocol import FixtureDataset, tiny_encoder
import torch


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        if 'SLURM_JOB_ID' not in os.environ:
            raise RuntimeError('Run these tests in Slurm')
        self.tmp = tempfile.TemporaryDirectory(dir=os.environ['SLURM_TMPDIR'])
        self.batch = Path(self.tmp.name)
        self.protocol = json.loads((worker.BENCH / 'protocol.json').read_text())
        self.protocol['ssl']['subset_epochs'] = 1
        self.protocol['evaluation']['epochs'] = 1
        self.hashes = common.code_hashes()
        self.datasets = lambda target: (FixtureDataset('', target, True), FixtureDataset('', target, False))
        self.group = 'CIFAR10/ConvNet/random/pt0.2/simclr'
        self.cells = [{'table': 'cifar10_main.tex', 'line': '1', 'source': 'CIFAR10',
                       'model': 'ConvNet', 'subset': '0.2', 'ssl': 'simclr', 'target': 'CIFAR10',
                       'labels': str(label)} for label in (1, 5)]
        self.plan = {'checkpoint_root': str(self.batch / 'checkpoints'),
                     'data_root': str(self.batch / 'fixture_data'),
                     'groups': [{'group': self.group, 'cells': self.cells}], 'jobs': []}
        self.encoder_patch = patch.object(common, 'build_encoder', tiny_encoder)
        self.encoder_patch.start()
        torch.set_num_threads(1)

    def tearDown(self):
        self.encoder_patch.stop()
        self.tmp.cleanup()

    def make_seed(self, seed=0):
        unit = {'source': 'CIFAR10', 'model': 'ConvNet', 'subset': 0.2, 'ssl': 'simclr',
                'seed': seed, 'group': self.group, 'cells': self.cells,
                'relative': self.group + f'/seed_{seed:03d}'}
        folder = Path(self.plan['checkpoint_root']) / unit['relative']
        folder.mkdir(parents=True)
        signature = {'protocol': self.protocol, 'code_hashes': self.hashes,
                     'selection_method': 'random', 'source_fingerprint': 'fixture-CIFAR10-True',
                     **{key: unit[key] for key in ('source', 'model', 'subset', 'ssl')}}
        identity = {'signature': common.digest_json(signature), 'seed': seed,
                    'selection': {}, 'selection_method': 'random'}
        common.atomic_json(folder / 'config.json', {'signature': signature, 'identity': identity})
        model, _ = tiny_encoder('', 32)
        saved = {'complete': True, 'kind': 'ssl_reference_encoder', 'identity': identity,
                 'encoder': model.state_dict(), 'encoder_sha256': common.state_hash(model),
                 'epochs': 1, 'model': 'ConvNet', 'source': 'CIFAR10', 'method': 'simclr',
                 'seed': seed, 'input_size': 32, 'normalization': self.protocol['normalization']['CIFAR10']}
        torch.save(saved, folder / 'encoder.pt')
        common.atomic_json(folder / 'ssl_complete.json', {
            key: value for key, value in saved.items() if key != 'encoder'} | {
                'checkpoint_file_sha256': common.file_hash(folder / 'encoder.pt')})
        self.plan['jobs'].append({'task': str(seed), 'units': [unit]})
        return unit, folder

    def evaluate(self, unit):
        worker.evaluate_seed(unit, self.plan, self.batch / 'results', self.datasets,
                             self.protocol, self.hashes, device='cuda')

    def test_gpu_probe_retention_resume_and_corruption(self):
        unit, folder = self.make_seed()
        before = common.file_hash(folder / 'encoder.pt')
        self.evaluate(unit)
        self.assertEqual(before, common.file_hash(folder / 'encoder.pt'))
        # Completed heads must be verified/reused without fitting again.
        with patch.object(torch.optim, 'SGD', side_effect=AssertionError('head was retrained')):
            self.evaluate(unit)
        output = self.batch / 'results' / unit['relative']
        result = output / 'evaluations/CIFAR10_lbl1.json'
        row = json.loads(result.read_text())
        row['accuracy_percent'] += 1
        common.atomic_json(result, row)
        with self.assertRaisesRegex(ValueError, 'modified'):
            self.evaluate(unit)
        status = report(self.batch, self.plan)
        self.assertEqual(len(status['invalid_encoders']), 1)
        self.assertEqual(status['completed_probes'], 0)

    def test_unfinished_checkpoint_is_rejected(self):
        unit, folder = self.make_seed()
        marker = json.loads((folder / 'ssl_complete.json').read_text())
        marker['epochs'] = 0
        common.atomic_json(folder / 'ssl_complete.json', marker)
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            self.evaluate(unit)
        self.assertFalse((self.batch / 'results' / unit['relative'] / 'complete.json').exists())

    def test_wrong_identity_and_changed_source_are_rejected(self):
        unit, _ = self.make_seed()
        wrong = copy.deepcopy(unit)
        wrong['seed'] = 9
        with self.assertRaisesRegex(ValueError, 'planned recipe'):
            self.evaluate(wrong)
        original = self.datasets
        def changed(target):
            train, test = original(target)
            train.fingerprint = 'changed-source'
            return train, test
        self.datasets = changed
        with self.assertRaisesRegex(ValueError, 'Source train split'):
            self.evaluate(unit)

    def test_ten_seed_summary_only_after_complete_coverage(self):
        for seed in range(10):
            unit, _ = self.make_seed(seed)
            self.evaluate(unit)
            if seed == 0:
                partial = report(self.batch, self.plan)
                self.assertEqual(partial['completed_table_cells'], 0)
                self.assertFalse((self.batch / 'results' / self.group / 'summary.json').exists())
        status = report(self.batch, self.plan)
        self.assertEqual(status['completed_encoders'], 10)
        self.assertEqual(status['completed_probes'], 20)
        self.assertEqual(status['completed_table_cells'], 2)
        summary = json.loads((self.batch / 'results' / self.group / 'summary.json').read_text())
        self.assertTrue(summary['complete'])
        self.assertEqual(summary['std_ddof'], 1)
        self.assertTrue(all(len(r['seeds']) == 10 for r in summary['results']))

    def test_worker_entry_writes_job_and_coverage_reports(self):
        unit, _ = self.make_seed()
        self.plan['jobs'] = [{'task': '01', 'units': [unit], 'probes': 2,
                              'account': os.environ['SLURM_JOB_ACCOUNT']}]
        common.atomic_json(self.batch / 'plan.json', self.plan)
        common.atomic_json(self.batch / 'protocol.json', self.protocol)
        with patch.object(worker, 'BENCH', self.batch), \
             patch.object(worker, 'RawDataset', FixtureDataset), \
             patch.object(torch, 'set_num_interop_threads'), \
             patch.object(sys, 'argv', ['worker.py', str(self.batch), '01']):
            worker.main()
        job = json.loads((self.batch / 'results/jobs/01.json').read_text())
        self.assertTrue(job['complete'])
        status = json.loads((self.batch / 'results/progress.json').read_text())
        self.assertEqual(status['completed_probes'], 2)
        self.assertEqual(status['invalid_encoders'], [])


class BackboneSmokeTests(unittest.TestCase):
    def test_six_real_backbone_shapes_on_gpu(self):
        self.assertTrue(torch.cuda.is_available())
        for model in ('ConvNet', 'VGG11', 'ResNet18'):
            for size in (32, 64):
                encoder, dimension = common.build_encoder(model, size)
                encoder = encoder.cuda().eval()
                with torch.no_grad():
                    features = encoder(torch.zeros(2, 3, size, size, device='cuda'))
                self.assertEqual(tuple(features.shape), (2, dimension))
                self.assertTrue(torch.isfinite(features).all())
                del encoder, features
                torch.cuda.empty_cache()


if __name__ == '__main__':
    unittest.main(verbosity=2)
