"""Small protocol/report fixtures; run on an allocated node without training."""
import itertools
import os
from pathlib import Path
import statistics
import tempfile
import unittest
from unittest.mock import patch

import interactive_mkdt_2k as mkdt
import interactive_mkdt_2k_main as main_only


class MKDT2kTests(unittest.TestCase):
    def test_main_only_targets_and_report(self):
        for source, model in itertools.product(mkdt.SOURCES, mkdt.MODELS):
            self.assertEqual(main_only.main_targets(source, model), (source,))
        main = {'source': 'CIFAR10', 'target': 'CIFAR10'}
        downstream = {'source': 'CIFAR10', 'target': 'Aircraft'}
        with patch.object(main_only, 'original_write_tsv') as write:
            main_only.write_main_tsv(Path('table_summary.tsv'), ['source', 'target'], [main, downstream])
        self.assertEqual(write.call_args.args[2], [main])

    def test_all_settings_keep_the_requested_budgets(self):
        for source, size in itertools.product(mkdt.SOURCES, mkdt.SIZES):
            protocol = mkdt.protocol_for(source, size)
            self.assertEqual(protocol['ssl']['subset_epochs'], 2000)
            self.assertEqual(protocol['ssl']['checkpoint_every_epochs'], 500)
            self.assertEqual(protocol['evaluation']['epochs'], 100)
            self.assertEqual(protocol['evaluation']['label_percentages'], [1, 5])
            self.assertEqual(protocol['subset_seeds'], [0, 1])
        self.assertEqual(mkdt.PROBE_SEEDS, (0, 1, 2))

    def test_input_paths_match_the_original_mkdt_manifest(self):
        import csv
        with (mkdt.BENCH / 'mkdt_ssl_jobs/tasks.tsv').open() as stream:
            rows = list(csv.DictReader(stream, delimiter='\t'))
        for source, model, size in itertools.product(mkdt.SOURCES, mkdt.MODELS, mkdt.SIZES):
            entry = mkdt.synthetic_entry(Path('/fixture'), source, model, size)
            original = next(row for row in rows if row['source'] == source and row['model'] == model
                            and int(row['subset']) == size)
            self.assertEqual(Path(entry['path']).relative_to('/fixture').as_posix(), original['images_relative'])
            self.assertEqual(entry['generator'], original['generator'])
            self.assertEqual(entry['space'], 'repository_normalized')

    def test_six_results_are_required_for_a_cell_mean(self):
        with tempfile.TemporaryDirectory(dir=os.environ.get('SLURM_TMPDIR', str(mkdt.BENCH))) as temporary:
            group = Path(temporary)
            signature = {'source': 'CIFAR10', 'model': 'ConvNet', 'subset': 1,
                         'protocol': {'evaluation': {'epochs': 100}}, 'code_hashes': {'fixture': 'hash'}}
            checksum = mkdt.digest_json(signature)
            paths = []
            for index, (ssl_seed, probe_seed) in enumerate(itertools.product(mkdt.SSL_SEEDS, mkdt.PROBE_SEEDS)):
                folder = group / f'ssl_seed_{ssl_seed:03d}'
                identity = {'signature': checksum, 'seed': ssl_seed, 'selection_method': 'mkdt'}
                mkdt.atomic_json(folder / 'config.json', {'signature': signature, 'identity': identity})
                mkdt.atomic_json(folder / 'ssl_complete.json', {
                    'complete': True, 'identity': identity, 'epochs': 2000, 'encoder_sha256': f'encoder{ssl_seed}'})
                expected = {'identity': identity, 'source': 'CIFAR10', 'target': 'CIFAR10',
                            'seed': probe_seed, 'label_percentage': 1, 'evaluation': {'epochs': 100},
                            'evaluation_code_hashes': {'fixture': 'hash'},
                            'train_fingerprint': 'train', 'test_fingerprint': 'test'}
                path = folder / 'evaluations' / f'CIFAR10_lbl1_probe{probe_seed}.json'
                mkdt.atomic_json(path, {'complete': True, 'encoder_unchanged': True,
                                       'encoder_sha256': f'encoder{ssl_seed}',
                                       'expected': expected, 'accuracy_percent': index})
                paths.append(path)
            row = mkdt.summarize(group, signature, ['CIFAR10'])[0]
            self.assertTrue(row['complete'])
            self.assertEqual(row['completed_probes'], 6)
            self.assertEqual(row['mean_percent'], statistics.mean(range(6)))
            self.assertEqual(row['std_percent'], statistics.stdev(range(6)))
            paths[-1].unlink()
            row = mkdt.summarize(group, signature, ['CIFAR10'])[0]
            self.assertFalse(row['complete'])
            self.assertEqual(row['completed_probes'], 5)
            self.assertIsNone(row['mean_percent'])


if __name__ == '__main__':
    unittest.main()
