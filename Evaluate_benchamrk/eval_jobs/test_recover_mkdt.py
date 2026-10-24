"""Small recovery fixtures; run only on an allocated node, without training."""
import itertools
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import recover_mkdt as recovery


class RecoveryTests(unittest.TestCase):
    def unit(self, source='TinyImageNet', model='ConvNet', subset=2, ssl='simclr', seed=0):
        return {'source': source, 'model': model, 'subset': subset, 'ssl': ssl, 'seed': seed,
                'relative': f'{source}/{model}/mkdt/pt{subset}/{ssl}/seed_{seed:03d}'}

    def test_only_the_other_twelve_timeouts_are_accepted(self):
        units = [self.unit(source, model, subset, ssl, seed)
                 for source, model, subset, ssl, seed in itertools.product(
                     recovery.SOURCES, ('ConvNet', 'VGG11', 'ResNet18'),
                     (1, 2), ('simclr', 'barlowtwins'), range(6))]
        accepted = [unit for unit in units if recovery.accepted_timeout(unit)]
        self.assertEqual(len(accepted), 12)
        self.assertEqual(sum(unit['source'] == 'TinyImageNet' for unit in accepted), 3)
        self.assertFalse(any(recovery.accepted_timeout(unit) for unit in units
                             if unit['source'] == 'TinyImageNet' and unit['subset'] == 2))

    def test_missing_resume_file_never_restarts_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            unit = self.unit()
            folder = root / unit['relative']
            folder.mkdir(parents=True)
            (folder / 'config.json').write_text('{}')
            args = type('Args', (), {'checkpoints': root})()
            with patch.object(recovery.subprocess, 'run') as execute:
                with self.assertRaisesRegex(FileNotFoundError, 'refusing to restart'):
                    recovery.train_missing(unit, args)
                execute.assert_not_called()

    def test_resume_calls_the_unchanged_original_worker(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            unit = self.unit()
            folder = root / unit['relative']
            folder.mkdir(parents=True)
            (folder / 'config.json').write_text('{}')
            (folder / 'training.pt').touch()
            args = type('Args', (), {
                'checkpoints': root, 'data_root': root / 'data',
                'synthetic_root': root / 'synthetic'})()
            with patch.object(recovery.subprocess, 'run') as execute:
                recovery.train_missing(unit, args)
            command = execute.call_args.args[0]
            self.assertEqual(command[:3], [recovery.sys.executable, '-u',
                             str(recovery.BENCH / 'mkdt_ssl_jobs/worker.py')])
            self.assertEqual(command[-3:], [str(args.data_root), str(args.synthetic_root), str(root)])
            with (recovery.BENCH / 'mkdt_ssl_jobs/tasks.tsv').open() as stream:
                task = next(row for row in recovery.csv.DictReader(stream, delimiter='\t')
                            if row['task'] == command[3])
            self.assertEqual(task['epochs'], '4000')
            self.assertEqual(task['source'], 'TinyImageNet')
            self.assertEqual(task['seed_start'], '0')
            self.assertTrue(execute.call_args.kwargs['check'])


if __name__ == '__main__':
    unittest.main()
