"""Frozen probes of the user-approved CIFAR100/ResNet50 epoch-950 snapshot."""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import statistics
import sys

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent
ROOT = BENCH.parent
sys.path[:0] = [str(HERE), str(BENCH)]

import torch
import common
from data import RawDataset
from evaluate import linear_probe
from models import install
from train_ssl import SSLModel

SOURCE = HERE / 'results/CIFAR100/ResNet50/full/pt100/simclr/ssl_seed_000'
GROUP = HERE / 'full950/results/CIFAR100/ResNet50/full/pt100/simclr'
OUTPUT = GROUP / 'ssl_seed_000'
TARGETS = ('CIFAR100', 'CIFAR10', 'TinyImageNet')


def read(path):
    return json.loads(Path(path).read_text())


def prepare():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / '.encoder.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        original = read(SOURCE / 'config.json')
        assert common.digest_json(original['signature']) == original['identity']['signature']
        assert original['signature']['code_hashes'] == common.code_hashes()
        state = torch.load(SOURCE / 'training.pt', map_location='cpu', weights_only=True, mmap=True)
        assert state['identity'] == original['identity'] and state['epoch'] == 950
        assert state['steps_per_epoch'] == 195 and state['steps'] == 950 * 195
        acceptance = dict(source_checkpoint=str(SOURCE / 'training.pt'),
            source_checkpoint_sha256=common.file_hash(SOURCE / 'training.pt'),
            accepted_epoch=950, requested_epoch=1000,
            reason='user_requested_frozen_evaluation_at_epoch_950', ssl_updates=0)
        signature = dict(original['signature'], accepted_snapshot=acceptance,
                         snapshot_probe_sha256=common.file_hash(__file__))
        identity = dict(original['identity'], signature=common.digest_json(signature))
        config = dict(identity=identity, signature=signature)
        destination = OUTPUT / 'encoder.pt'
        if (OUTPUT / 'config.json').exists():
            assert read(OUTPUT / 'config.json') == config, 'Snapshot/probe inputs changed'
        common.atomic_json(OUTPUT / 'config.json', config)
        if not destination.exists():
            protocol = signature['protocol']
            model = SSLModel('ResNet50', protocol['input_size']['CIFAR100'], protocol['methods']['simclr'])
            model.load_state_dict(state['model'], strict=True)
            encoder = model.encoder.cpu().eval().requires_grad_(False)
            saved = dict(kind='ssl_reference_encoder', complete=True, identity=identity,
                source='CIFAR100', model='ResNet50', method='simclr', seed=0,
                epochs=950, requested_epochs=1000, accepted_partial=True,
                steps=state['steps'], effective_batch=protocol['ssl']['batch_size'],
                input_size=protocol['input_size']['CIFAR100'],
                normalization=protocol['normalization']['CIFAR100'],
                encoder=encoder.state_dict(), encoder_sha256=common.state_hash(encoder))
            common.atomic_save(destination, saved)
            del model, encoder, saved
        encoder, _, saved = common.load_encoder(destination, identity, 'cpu')
        assert saved['epochs'] == 950 and saved['requested_epochs'] == 1000 and saved['accepted_partial']
        del encoder, state
        common.atomic_json(OUTPUT / 'ssl_complete.json', dict(complete=True, identity=identity,
            epochs=950, requested_epochs=1000, accepted_partial=True,
            encoder_sha256=saved['encoder_sha256'], checkpoint_file_sha256=common.file_hash(destination)))
        common.atomic_json(OUTPUT / 'accepted_snapshot.json', acceptance)
    return destination, config, saved['encoder_sha256']


def report(target, labels, config, encoder_hash, samples):
    scores = [r['accuracy_percent'] for r in samples]
    cell = dict(target=target, labels=labels, complete=True, completed_probes=5, expected_probes=5,
        mean_percent=statistics.mean(scores), std_percent=statistics.stdev(scores),
        samples=[dict(ssl_seed=0, probe_seed=i, accuracy_percent=score) for i, score in enumerate(scores)])
    with (GROUP / '.report.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        common.atomic_json(GROUP / ('summary_%s_lbl%d.json' % (target, labels)), cell)
        cells = []
        for name in TARGETS:
            for budget in (1, 5):
                path = GROUP / ('summary_%s_lbl%d.json' % (name, budget))
                if path.exists():
                    cells.append(read(path))
        common.atomic_json(GROUP / 'summary.json', dict(source='CIFAR100', model='ResNet50',
            kind='full', subset=100, ssl='simclr', epochs=950, requested_epochs=1000,
            accepted_partial=True, std_ddof=1, cells=cells, complete=len(cells) == 6,
            identity=config['identity'], encoder_sha256=encoder_hash))
        value = r'\dsacc{%.2f}{%.2f}' % (cell['mean_percent'], cell['std_percent'])
        # Preserve every other cell, including the selected older MIPA results.
        for filename, tag, column in (
            ('cross_arch_combined.tex', 'crossarch backbone=ResNet50 kind=full pt=100 lbl=%d' % labels,
             7 + TARGETS.index(target)),
            ('cross_arch_cifar100.tex', 'crossarch kind=full pt=100 lbl=%d' % labels,
             6 + TARGETS.index(target))):
            table = ROOT / 'CVPR_tables/ablations' / filename
            text = table.read_text()
            rows = [line for line in text.splitlines() if line.endswith('% ' + tag)]
            assert len(rows) == 1
            old = rows[0]
            fields = old.split('&')
            assert len(fields) == (10 if filename == 'cross_arch_combined.tex' else 9)
            previous = fields[column]
            color = r'\cellcolor{gray!12}' if r'\cellcolor{gray!12}' in previous else ''
            ending = r' \\ % ' + tag if column == len(fields) - 1 else ' '
            fields[column] = ' ' + color + value + ending
            text = text.replace(old, '&'.join(fields))
            temporary = table.with_name('.%s.%d.tmp' % (table.name, os.getpid()))
            temporary.write_text(text)
            temporary.replace(table)
    print('CELL COMPLETE: %s, labels %d%%: %.4f +/- %.4f; SSL epoch 950, zero SSL updates' %
          (target, labels, cell['mean_percent'], cell['std_percent']), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('target', choices=TARGETS)
    parser.add_argument('labels', type=int, choices=(1, 5))
    args = parser.parse_args()
    assert os.environ.get('SLURM_JOB_ID') and torch.cuda.is_available()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    install()
    checkpoint, config, encoder_hash = prepare()
    protocol = config['signature']['protocol']
    assert protocol['evaluation']['epochs'] == 100
    train = RawDataset(ROOT.parent / 'data', args.target, True)
    test = RawDataset(ROOT.parent / 'data', args.target, False)
    samples = []
    with common.exclusive_run(OUTPUT / ('probe_%s_lbl%d' % (args.target, args.labels))):
        for seed in range(5):
            result = linear_probe(checkpoint, config['identity'], protocol, 'CIFAR100', args.target,
                train, test, args.labels, seed,
                OUTPUT / 'evaluations' / ('%s_lbl%d_probe%d.json' % (args.target, args.labels, seed)),
                workers=0, device='cuda')
            assert result['complete'] and result['encoder_unchanged'] and result['encoder_sha256'] == encoder_hash
            assert math.isfinite(result['accuracy_percent'])
            samples.append(result)
        report(args.target, args.labels, config, encoder_hash, samples)


if __name__ == '__main__':
    main()
