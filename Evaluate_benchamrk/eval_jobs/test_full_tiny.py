"""Small CPU fixtures; run on the allocated node before TinyImageNet probing."""
import copy
import os
from pathlib import Path
import statistics
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn

import interactive_full_tiny as tiny
import common


class TinyFullTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(
            prefix='tiny-full-check-', dir=os.environ.get('SLURM_TMPDIR', str(tiny.BENCH)))
        self.root = Path(self.tmp.name)
        self.hashes, self.drivers, self.extension = {'fixture': 'code'}, {'fixture': 'driver'}, {'fixture': 'extension'}

    def tearDown(self):
        self.tmp.cleanup()

    def test_schedule_and_explicit_acceptances(self):
        plan = tiny.build_plan(self.root / 'checkpoints', self.root / 'data')
        units = plan['jobs'][0]['units']
        self.assertEqual(len(units), 17)
        self.assertEqual(plan['jobs'][0]['probes'], 110)
        self.assertEqual(sum(len(g['cells']) for g in plan['groups']), 36)
        self.assertEqual([u['seed'] for u in units if u['group'] == tiny.VGG_GROUP], [0, 2])
        self.assertEqual(len(tiny.ACCEPTED_TINY), 8)
        self.assertNotIn(('TinyImageNet', 'VGG11', 'simclr', 1), tiny.ACCEPTED_TINY)
        self.assertEqual({r['accepted_epoch'] for r in tiny.ACCEPTED_TINY.values()}, {950})
        self.assertEqual(len({0, 2, tiny.repeat_seed(0), tiny.repeat_seed(2)}), 4)

    def fixture_plan(self):
        actual = tiny.build_plan(self.root / 'checkpoints', self.root / 'data')
        units = [u for u in actual['jobs'][0]['units'] if u['group'] == tiny.VGG_GROUP]
        group = next(g for g in actual['groups'] if g['group'] == tiny.VGG_GROUP)
        plan = {**actual, 'groups': [group], 'jobs': [{'task': 'TinyImageNet', 'units': units, 'probes': 8}]}
        for unit in units:
            folder = self.root / unit['relative']
            signature = {'protocol': {'evaluation': {'epochs': 100}}}
            identity = {'seed': unit['seed'], 'selection_method': 'full',
                        'signature': common.digest_json(signature)}
            common.atomic_json(folder / 'config.json', {'signature': signature, 'identity': identity})
            base = {'complete': True, 'identity': identity, 'encoder_sha256': 'fixture-encoder',
                    'checkpoint': str(folder / 'fixture.pt'), 'checkpoint_file_sha256': 'fixture-checksum',
                    'checkpoint_retained': True, 'ssl_epochs': 950, 'accepted_rolling_checkpoint': None,
                    'evaluations': [[c['target'], int(c['labels'])] for c in unit['cells']],
                    'evaluation_code_hashes': self.hashes, 'driver_hashes': self.drivers}
            primary_hashes, repeated_hashes = {}, {}
            for cell in unit['cells']:
                name = f"{cell['target']}_lbl{cell['labels']}.json"
                expected = {'identity': identity, 'source': 'TinyImageNet', 'target': 'TinyImageNet',
                            'seed': unit['seed'], 'label_percentage': int(cell['labels']),
                            'labeled_indices': [unit['seed']], 'train_fingerprint': 'fixture-train',
                            'test_fingerprint': 'fixture-test', 'evaluation': {'epochs': 100},
                            'evaluation_code_hashes': self.hashes}
                row = {'complete': True, 'expected': expected, 'encoder_sha256': 'fixture-encoder',
                       'encoder_unchanged': True, 'accuracy_percent': 20 + 10 * unit['seed'], 'seconds': 1}
                path = folder / 'evaluations' / name
                common.atomic_json(path, row)
                primary_hashes[name] = common.file_hash(path)
                repeated = copy.deepcopy(row)
                repeated['expected'].update(seed=tiny.repeat_seed(unit['seed']),
                                            labeled_indices=[tiny.repeat_seed(unit['seed'])])
                repeated['accuracy_percent'] += 10
                path = folder / 'probe_repeat_001' / 'evaluations' / name
                common.atomic_json(path, repeated)
                repeated_hashes[name] = common.file_hash(path)
            common.atomic_json(folder / 'complete.json', {**base, 'result_hashes': primary_hashes})
            common.atomic_json(folder / 'probe_repeat_001' / 'complete.json', {
                **tiny.repeat_marker(base, tiny.repeat_seed(unit['seed']), self.extension),
                'result_hashes': repeated_hashes})
        return plan

    def test_four_probes_count_as_two_encoders_and_reuse_reports(self):
        plan = self.fixture_plan()
        for _ in range(2):
            progress = tiny.report(self.root, plan, self.hashes, self.drivers, self.extension)
            self.assertTrue(progress['complete'])
            self.assertEqual(progress['completed_encoders'], 2)
            self.assertEqual(progress['completed_probes'], 8)
            rows = tiny.full.read(self.root / tiny.VGG_GROUP / 'summary.json')['results']
            for row in rows:
                self.assertEqual(row['completed_seeds'], 2)
                self.assertEqual(row['expected_seeds'], 2)
                self.assertEqual(row['completed_probe_runs'], 4)
                self.assertEqual(row['mean_percent'], 35)
                self.assertAlmostEqual(row['std_percent'], statistics.stdev([20, 30, 40, 50]))

    def test_missing_repeat_keeps_cells_incomplete(self):
        plan = self.fixture_plan()
        path = self.root / tiny.VGG_GROUP / 'seed_002/probe_repeat_001/complete.json'
        path.rename(path.with_name('interrupted.json'))
        progress = tiny.report(self.root, plan, self.hashes, self.drivers, self.extension)
        self.assertFalse(progress['complete'])
        self.assertEqual(progress['completed_table_cells'], 0)
        self.assertEqual(progress['completed_probes'], 6)

    def test_changed_repeat_result_is_rejected(self):
        plan = self.fixture_plan()
        path = self.root / tiny.VGG_GROUP / 'seed_002/probe_repeat_001/evaluations/TinyImageNet_lbl1.json'
        row = tiny.full.read(path)
        row['accuracy_percent'] += 1
        common.atomic_json(path, row)
        progress = tiny.report(self.root, plan, self.hashes, self.drivers, self.extension)
        self.assertEqual(progress['completed_table_cells'], 0)
        self.assertEqual(len(progress['invalid_encoders']), 1)

    def test_accepted_snapshot_strict_epoch_and_encoder_extraction(self):
        unit = {'source': 'TinyImageNet', 'model': 'VGG11', 'ssl': 'simclr', 'seed': 2}
        identity = {'seed': 2, 'selection_method': 'full', 'signature': 'fixture'}
        protocol = tiny.full.expected_protocol()

        class SmallSSL(nn.Module):
            def __init__(self, *args):
                super().__init__()
                self.encoder = nn.Linear(4, 2)

        model = SmallSSL()
        rolling = {'identity': identity, 'epoch': 950, 'steps': 950 * 390,
                   'steps_per_epoch': 390, 'model': model.state_dict()}
        checkpoint = self.root / 'training.pt'
        torch.save(rolling, checkpoint)
        with patch.object(tiny.full, 'validate_config', return_value={'identity': identity}), \
                patch.object(tiny.full, 'SSLModel', SmallSSL), \
                patch.object(common, 'build_encoder', side_effect=lambda *args: (nn.Linear(4, 2), 2)):
            _, saved, _, derived, _ = tiny.full.derive_accepted_encoder(
                self.root, self.root / 'accepted', unit, protocol, {}, '')
            self.assertEqual(saved['epochs'], 950)
            self.assertTrue(saved['accepted_partial'])
            self.assertEqual(saved['encoder_sha256'], common.state_hash(model.encoder))
            self.assertTrue(derived.is_file())
            rolling['epoch'] = 900
            torch.save(rolling, checkpoint)
            with self.assertRaisesRegex(ValueError, 'epoch-950'):
                tiny.full.derive_accepted_encoder(
                    self.root, self.root / 'rejected', unit, protocol, {}, '')


if __name__ == '__main__':
    if not os.environ.get('SLURM_JOB_ID'):
        raise SystemExit('Run these checks on your allocated node')
    torch.set_num_threads(1)
    unittest.main(verbosity=1)
