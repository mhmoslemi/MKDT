# Account-only snapshot: preserve worker.py and existing run identities for queued probes.
"""Existing seed-0 subsets, one fresh 2k SimCLR encoder and five frozen probes."""
import ast
import fcntl
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent
ROOT = BENCH.parent
sys.path.insert(0, str(BENCH))
import torch
from common import atomic_json, code_hashes, digest_json, exclusive_run, file_hash, load_encoder
from data import RawDataset, SOURCE_COUNTS
from evaluate import linear_probe
from train_ssl import train


def read(path):
    return json.loads(Path(path).read_text())


def summarize(folder, task, config, marker, train_set, test_set):
    samples = []
    for seed in range(5):
        path = folder / 'evaluations' / ('%s_lbl%d_probe%d.json' % (task['source'], task['labels'], seed))
        if not path.is_file():
            continue
        result = read(path)
        expected = result['expected']
        if (not result.get('complete') or not result.get('encoder_unchanged')
                or result['encoder_sha256'] != marker['encoder_sha256']
                or expected['identity'] != config['identity']
                or [expected['source'], expected['target'], expected['label_percentage'], expected['seed']]
                != [task['source'], task['source'], task['labels'], seed]
                or expected['evaluation'] != config['signature']['protocol']['evaluation']
                or expected['evaluation_code_hashes'] != code_hashes()
                or expected['train_fingerprint'] != train_set.fingerprint
                or expected['test_fingerprint'] != test_set.fingerprint
                or not math.isfinite(result['accuracy_percent']) or not 0 <= result['accuracy_percent'] <= 100):
            raise ValueError('Invalid frozen probe: %s' % path)
        samples.append({'ssl_seed': 0, 'probe_seed': seed, 'accuracy_percent': result['accuracy_percent'],
                        'result_file': str(path), 'encoder_sha256': result['encoder_sha256']})
    complete = len(samples) == 5
    scores = [s['accuracy_percent'] for s in samples]
    summary = {'complete': complete, 'source': task['source'], 'model': 'ResNet18', 'kind': task['kind'],
        'subset': task['subset'], 'ssl': 'simclr', 'labels': task['labels'], 'ssl_seed': 0,
        'ssl_epochs': 2000, 'linear_epochs': 100, 'completed_probes': len(samples), 'expected_probes': 5,
        'std_ddof': 1, 'samples': samples, 'table_scope': 'CVPR_tables/table.tex',
        'cell_key': '|'.join(map(str, (task['source'], 'ResNet18', task['kind'], task['subset'], 'simclr', task['source'], task['labels'])))}
    if complete:
        summary.update(mean_percent=statistics.mean(scores), std_percent=statistics.stdev(scores))
        summary['tex_value'] = '\\dsacc{%.2f}{%.2f}' % (summary['mean_percent'], summary['std_percent'])
    path = folder / ('summary_lbl%d.json' % task['labels'])
    atomic_json(path, summary)
    if complete:
        with (ROOT / 'CVPR_tables/.probe_completion.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            subprocess.run(['node', str(HERE / 'tables.js'), str(path)], check=True)


def main():
    task, data_root = read(sys.argv[1]), Path(sys.argv[2]).resolve()
    if (not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available()
            or os.environ.get('SLURM_JOB_ACCOUNT') != 'aip-yiweilu'):
        raise RuntimeError('Use the Lu Vulcan GPU submission')
    assert task['stage'] in ('ssl', 'probe') and task['kind'] in ('random', 'kmeans')
    assert (task['source'], task['subset']) in (('CIFAR10', 2), ('CIFAR100', 2),
                                               ('TinyImageNet', 0.2), ('TinyImageNet', 1), ('TinyImageNet', 2))
    for relative, expected in task['code_hashes'].items():
        if file_hash(ROOT / relative) != expected:
            raise ValueError('Code changed since submission: ' + relative)
    ast.parse(Path(__file__).read_text(), filename=__file__)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    origin = BENCH / task['selection_config']
    if file_hash(origin) != task['selection_config_sha256']:
        raise ValueError('Original subset metadata changed')
    previous = read(origin)
    indices = previous['identity']['selection']['indices']
    train_set, test_set = RawDataset(data_root, task['source'], True), RawDataset(data_root, task['source'], False)
    count = int(SOURCE_COUNTS[task['source']] * task['subset'] / 100)
    if (previous['identity']['seed'] != 0 or previous['identity']['selection_method'] != task['kind']
            or previous['signature']['source_fingerprint'] != train_set.fingerprint
            or previous['signature']['code_hashes'] != code_hashes()
            or len(indices) != count or len(set(indices)) != count or not all(0 <= i < len(train_set) for i in indices)):
        raise ValueError('Saved seed-0 subset or source data mismatch')
    protocol = read(BENCH / 'protocol.json')
    protocol['ssl'].update(subset_epochs=2000, checkpoint_every_epochs=500)
    protocol['subset_seeds'] = [0]
    assert protocol['evaluation']['epochs'] == 100
    signature = {'experiment': 'main_table_subset_simclr2k_1ssl5probes', 'source': task['source'],
        'source_fingerprint': train_set.fingerprint, 'model': 'ResNet18', 'selection_method': task['kind'],
        'subset': task['subset'], 'ssl': 'simclr', 'protocol': protocol, 'code_hashes': code_hashes(),
        'driver_code_hashes': task['code_hashes'], 'probe_seeds': [0, 1, 2, 3, 4],
        'selection_origin': {'config': task['selection_config'], 'sha256': task['selection_config_sha256']},
        'table_scope': 'CVPR_tables/table.tex'}
    identity = {'signature': digest_json(signature), 'seed': 0,
                'selection_method': task['kind'], 'selection': previous['identity']['selection']}
    config = {'signature': signature, 'identity': identity}
    folder = HERE / 'results' / task['source'] / 'ResNet18' / task['kind'] / ('pt%g' % task['subset']) / 'simclr/ssl_seed_000'
    if task['stage'] == 'ssl':
        with exclusive_run(folder):
            path = folder / 'config.json'
            if path.exists() and read(path) != config:
                raise ValueError('New run configuration changed; refusing to mix epochs/results')
            atomic_json(path, config)
            selected = torch.utils.data.Subset(train_set, indices)
            checkpoint, saved = train(selected, protocol, task['source'], 'ResNet18', 'simclr', 0,
                                      folder, identity, workers=0, device='cuda')
            assert saved['epochs'] == 2000 and saved['steps'] == 2000 * (count // min(count, protocol['ssl']['batch_size']))
            atomic_json(folder / 'ssl_complete.json', {'complete': True, 'identity': identity, 'epochs': 2000,
                'encoder_sha256': saved['encoder_sha256'], 'checkpoint_file_sha256': file_hash(checkpoint)})
        return
    assert task['labels'] in (1, 5) and task['probe_seed'] in range(5)
    with (folder / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        if read(folder / 'config.json') != config:
            raise ValueError('Encoder configuration differs from the new 2k protocol')
        marker = read(folder / 'ssl_complete.json')
        checkpoint = folder / 'encoder.pt'
        if (not marker['complete'] or marker['identity'] != identity or marker['epochs'] != 2000
                or marker['checkpoint_file_sha256'] != file_hash(checkpoint)):
            raise ValueError('Incomplete or changed 2k encoder')
        encoder, _, saved = load_encoder(checkpoint, identity, device='cpu')
        assert saved['epochs'] == 2000 and saved['encoder_sha256'] == marker['encoder_sha256']
        del encoder, saved
        name = '%s_lbl%d_probe%d' % (task['source'], task['labels'], task['probe_seed'])
        with exclusive_run(folder / 'probe_locks' / name):
            linear_probe(checkpoint, identity, protocol, task['source'], task['source'], train_set, test_set,
                         task['labels'], task['probe_seed'], folder / 'evaluations' / (name + '.json'), workers=0, device='cuda')
        with (folder / ('.summary_lbl%d.lock' % task['labels'])).open('a') as summary_lock:
            fcntl.flock(summary_lock, fcntl.LOCK_EX)
            summarize(folder, task, config, marker, train_set, test_set)


if __name__ == '__main__':
    main()
