"""CIFAR10 fresh-random-network ablation; reuse method1 and benchmark evaluation."""
import fcntl
import functools
import itertools
import json
import os
from pathlib import Path
import re
import subprocess
import sys

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent
ROOT = BENCH.parent
sys.path[:0] = [str(BENCH / 'eval_jobs'), str(BENCH)]
import torch
import interactive_ours as ours
from common import load_encoder

TARGETS = ('CIFAR10', 'CIFAR100', 'TinyImageNet')


def read(path):
    return json.loads(Path(path).read_text())


def paths(task):
    size = '%g' % task['size']
    synthetic = ROOT / 'OT-SSL-DD/ours_syntehtic_random_network/CIFAR10/ResNet18BN' / ('pt' + size)
    checkpoint = synthetic / ('res_IIC_CIFAR10_ResNet18BN_%spercent_final.pt' % size)
    group = HERE / 'results/CIFAR10/ResNet18BN' / ('pt' + size) / 'simclr'
    folder = group / ('ssl_seed_%03d' % task['seed'])
    return synthetic, checkpoint, group, folder


def finish(task, files):
    ours.atomic_json(HERE / 'status' / (task['name'] + '.json'), {
        'complete': True, 'code_hashes': task['code_hashes'],
        'artifacts': {str(path.relative_to(ROOT)): ours.file_hash(path) for path in files}})


def update_table():
    table = ROOT / 'CVPR_tables/ablations/cifar10_random_network.tex'
    text = table.read_text()
    for size, labels in itertools.product((0.2, 1), (1, 5)):
        group = HERE / 'results/CIFAR10/ResNet18BN' / ('pt%g' % size) / 'simclr'
        summary = group / 'summary.json'
        if not summary.is_file():
            continue
        cells = {row['target']: row for row in read(summary)['cells'] if row['labels'] == labels}
        values = [('\\dsacc{%.2f}{%.2f}' % (cells[t]['mean_percent'], cells[t]['std_percent']))
                  if cells[t]['complete'] else '' for t in TARGETS]
        tag = 'ablation pt=%g lbl=%d' % (size, labels)
        pattern = r'^& & Ours \(random network\).*% ' + re.escape(tag) + r'$'
        replacement = '& & Ours (random network) & ' + ' & '.join(values) + r' \\ % ' + tag
        text, count = re.subn(pattern, lambda _match: replacement, text, flags=re.MULTILINE)
        assert count == 1, 'Missing/duplicate ablation table row: ' + tag
    temporary = table.with_name('.%s.%d.tmp' % (table.name, os.getpid()))
    temporary.write_text(text)
    temporary.replace(table)


def main():
    task, data_root = read(sys.argv[1]), Path(sys.argv[2]).resolve()
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise RuntimeError('Run in the submitted GPU job')
    assert task['stage'] in ('generation', 'ssl', 'probe') and task['size'] in (0.2, 1)
    assert task['seed'] in (0, 1)
    for relative, expected in task['code_hashes'].items():
        if ours.file_hash(ROOT / relative) != expected:
            raise ValueError('Code changed since submission: ' + relative)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    synthetic_dir, synthetic_file, group, folder = paths(task)
    if task['stage'] == 'generation':
        command = [sys.executable, '-u', str(ROOT / 'OT-SSL-DD/method1.py')]
        for key, value in task['generation_args'].items():
            command.append('--' + key)
            if key != 'dsa':
                command.extend(str(item) for item in (value if isinstance(value, list) else [value]))
        command += ['--data_path', str(data_root), '--save_path', str(synthetic_dir),
                    '--device', 'cuda', '--num_eval', '0', '--resume', '--save_every', '100']
        with ours.exclusive_run(synthetic_dir):
            subprocess.run(command, cwd=ROOT / 'OT-SSL-DD', check=True)
            loaded = ours.load_synthetic(synthetic_file, 'CIFAR10', 'ResNet18BN', task['size'])
            if loaded is None:
                raise ValueError('Generation did not finish')
            finish(task, [synthetic_file] + [synthetic_dir / (
                'vis_IIC_CIFAR10_ResNet18BN_%gpercent_iter%d.png' % (task['size'], i))
                for i in (1, task['generation_args']['Iteration'])])
        return

    saved = torch.load(synthetic_file, map_location='cpu', weights_only=True, mmap=True)
    if saved['run_args'] != task['generation_args'] or saved['teacher_source'] != 'fresh':
        raise ValueError('Not the requested fresh-network ablation synthetic data')
    del saved
    loaded = ours.load_synthetic(synthetic_file, 'CIFAR10', 'ResNet18BN', task['size'])
    if loaded is None:
        raise ValueError('Synthetic generation must complete before evaluation')
    synthetic, metadata = loaded
    protocol = ours.expected_protocol('random', 'CIFAR10', task['size'])
    protocol['ssl'].update(subset_epochs=2000, checkpoint_every_epochs=500)
    protocol['subset_seeds'] = [0, 1]

    @functools.lru_cache(maxsize=3)
    def datasets(target):
        return ours.RawDataset(data_root, target, True), ours.RawDataset(data_root, target, False)

    signature = {'protocol': protocol, 'source': 'CIFAR10', 'model': 'ResNet18',
                 'subset': task['size'], 'ssl': 'simclr', 'selection_method': 'ours',
                 'ablation': 'fresh_random_network_each_distillation_iteration',
                 'source_fingerprint': datasets('CIFAR10')[0].fingerprint, 'synthetic': metadata,
                 'ssl_seeds': [0, 1], 'probe_seeds': [0, 1, 2],
                 'code_hashes': ours.code_hashes(), 'ablation_code_hashes': task['code_hashes']}
    identity = {'signature': ours.digest_json(signature), 'seed': task['seed'], 'selection_method': 'ours'}
    config = {'signature': signature, 'identity': identity}
    with ours.exclusive_run(folder):
        config_path = folder / 'config.json'
        if config_path.exists() and read(config_path) != config:
            raise ValueError('Ablation inputs/code changed; refusing to overwrite results')
        if task['stage'] == 'ssl':
            ours.atomic_json(config_path, config)
            checkpoint, saved = ours.train(synthetic, protocol, 'CIFAR10', 'ResNet18', 'simclr',
                                          task['seed'], folder, identity, workers=0, device='cuda')
            assert saved['epochs'] == 2000
            marker = folder / 'ssl_complete.json'
            ours.atomic_json(marker, {'complete': True, 'identity': identity, 'epochs': saved['epochs'],
                             'encoder_sha256': saved['encoder_sha256'],
                             'checkpoint_file_sha256': ours.file_hash(checkpoint)})
            finish(task, [config_path, checkpoint, marker])
            return
        if not config_path.is_file():
            raise ValueError('Missing SSL config')
        checkpoint = folder / 'encoder.pt'
        marker = read(folder / 'ssl_complete.json')
        if (not marker['complete'] or marker['identity'] != identity or marker['epochs'] != 2000
                or marker['checkpoint_file_sha256'] != ours.file_hash(checkpoint)):
            raise ValueError('SSL completion/checksum mismatch')
        encoder, _, saved = load_encoder(checkpoint, identity, device='cpu')
        assert saved['encoder_sha256'] == marker['encoder_sha256'] and saved['epochs'] == 2000
        del encoder, saved, synthetic
        results = []
        for target, labels, probe_seed in itertools.product(TARGETS, (1, 5), (0, 1, 2)):
            path = folder / 'evaluations' / ('%s_lbl%d_probe%d.json' % (target, labels, probe_seed))
            train_set, test_set = datasets(target)
            result = ours.linear_probe(checkpoint, identity, protocol, 'CIFAR10', target, train_set,
                                       test_set, labels, probe_seed, path, workers=0, device='cuda')
            assert result['encoder_unchanged'] and result['encoder_sha256'] == marker['encoder_sha256']
            results.append(path)
            with (HERE / '.report.lock').open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                ours.summarize(group, identity['signature'], 'CIFAR10', 'ResNet18BN',
                               task['size'], 'simclr', TARGETS)
                update_table()
        finish(task, results)


if __name__ == '__main__':
    main()
