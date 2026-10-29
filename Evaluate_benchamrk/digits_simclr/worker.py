"""First stage: full-data digit SimCLR teachers and main-table probes only."""
import argparse
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent
ROOT = BENCH.parent
sys.path.insert(0, str(BENCH))

import torch
from torchvision.datasets import MNIST

from common import atomic_json, code_hashes, digest_json, exclusive_run, file_hash, load_encoder
from evaluate import linear_probe
from run import validate_protocol
import train_ssl
from datasets import COUNTS, DigitDataset


def read(path):
    return json.loads(Path(path).read_text())


def implementation():
    return {**code_hashes(), **{str(path.relative_to(ROOT)): file_hash(path)
        for path in (Path(__file__), HERE / 'datasets.py')}}


def quiet_ssl(*values, **kwargs):
    if kwargs.get('file') is None and len(values) == 1 and str(values[0]).startswith('SSL '):
        prefix, line = values[0].split(': ', 1)
        row = json.loads(line)
        if row['epoch'] != 1 and row['epoch'] % 200:
            return
        row.pop('projector_std', None)
        row.pop('seconds', None)
        values = (prefix + ': ' + json.dumps(row),)
    print(*values, **kwargs)


def folder(source, seed):
    return HERE / 'checkpoints' / source / 'ResNet18' / 'full' / 'pt100' / 'simclr' / ('seed_%03d' % seed)


def configuration(source, seed, prepared):
    signature = dict(protocol=prepared['protocol'], code_hashes=prepared['code_hashes'],
        source=source, model='ResNet18', ssl='simclr', selection_method='full', subset=100,
        source_fingerprint=prepared['datasets'][source]['train_fingerprint'],
        test_fingerprint=prepared['datasets'][source]['test_fingerprint'],
        input_scope='entire_official_training_split', image_count=COUNTS[source][0])
    identity = dict(signature=digest_json(signature), seed=seed,
                    selection_method='full', selection={'scope': 'full'})
    return dict(signature=signature, identity=identity)


def summarize(source):
    with exclusive_run(HERE / 'summaries' / source):
        cells = []
        for labels in (1, 5):
            rows = []
            for seed in (0, 1, 2):
                output = folder(source, seed)
                path = output / 'evaluations' / ('%s_lbl%d.json' % (source, labels))
                if not path.exists():
                    continue
                row, cfg = read(path), read(output / 'config.json')
                if (not row['complete'] or not row['encoder_unchanged']
                        or row['expected']['identity'] != cfg['identity']):
                    raise ValueError('Invalid saved full-data probe: ' + str(path))
                rows.append(row['accuracy_percent'])
            cells.append(dict(label_percentage=labels, completed_probes=len(rows), expected_probes=3,
                complete=len(rows) == 3, accuracies_percent=rows,
                mean_percent=statistics.mean(rows) if rows else None,
                std_percent=statistics.stdev(rows) if len(rows) > 1 else None))
        atomic_json(HERE / 'summaries' / source / 'summary.json',
                    dict(source=source, method='full', complete=all(x['complete'] for x in cells), cells=cells))


def full(source, seed, data_root):
    prepared = read(HERE / 'prepared.json')
    if prepared['code_hashes'] != implementation():
        raise ValueError('Code changed since preparation; do not mix checkpoints')
    protocol = prepared['protocol']
    validate_protocol(protocol)
    train_set, test_set = DigitDataset(data_root, source, True), DigitDataset(data_root, source, False)
    expected = prepared['datasets'][source]
    if (train_set.fingerprint != expected['train_fingerprint']
            or test_set.fingerprint != expected['test_fingerprint']):
        raise ValueError('Digit dataset changed since preparation')
    output = folder(source, seed)
    cfg = configuration(source, seed, prepared)
    with exclusive_run(output):
        if (output / 'config.json').exists() and read(output / 'config.json') != cfg:
            raise ValueError('Existing full-data run has different settings')
        atomic_json(output / 'config.json', cfg)
        atomic_json(output / 'environment.json', dict(job=os.environ['SLURM_JOB_ID'],
            account=os.environ.get('SLURM_JOB_ACCOUNT'), gpu=torch.cuda.get_device_name(),
            torch=str(torch.__version__), python=sys.version))
        checkpoint, metadata = train_ssl.train(train_set, protocol, source, 'ResNet18', 'simclr',
            seed, output, cfg['identity'], workers=0, device='cuda', full=True)
        assert metadata['epochs'] == 1000 and metadata['steps'] == 1000 * (len(train_set) // 256)
        encoder, _, checked = load_encoder(checkpoint, cfg['identity'], device='cpu')
        assert checked['encoder_sha256'] == metadata['encoder_sha256']
        del encoder
        atomic_json(output / 'ssl_complete.json', dict(complete=True, epochs=1000,
            identity=cfg['identity'], encoder_sha256=metadata['encoder_sha256'],
            checkpoint_file_sha256=file_hash(checkpoint)))
        for labels in (1, 5):
            result = linear_probe(checkpoint, cfg['identity'], protocol, source, source,
                train_set, test_set, labels, seed,
                output / 'evaluations' / ('%s_lbl%d.json' % (source, labels)), workers=0, device='cuda')
            assert result['complete'] and result['encoder_unchanged']
        atomic_json(output / 'complete.json', dict(complete=True, identity=cfg['identity']))
    summarize(source)


def prepare(data_root, launcher, venv):
    with exclusive_run(HERE / 'preparation'):
        # Torchvision checks each official archive checksum; existing data is reused.
        MNIST(str(data_root), train=True, download=True)
        MNIST(str(data_root), train=False, download=True)
        hashes = implementation()
        protocol = read(BENCH / 'protocol.json')
        protocol['sources'] = list(COUNTS)
        protocol['models'], protocol['full_seeds'] = ['ResNet18'], [0, 1, 2]
        protocol['ssl'].update(full_epochs=1000, subset_epochs=2000, checkpoint_every_epochs=50)
        validate_protocol(protocol)
        entries = {}
        for source in COUNTS:
            train_set, test_set = DigitDataset(data_root, source, True), DigitDataset(data_root, source, False)
            protocol['normalization'][source] = train_set.normalization()
            protocol['input_size'][source] = 32
            entries[source] = dict(train_fingerprint=train_set.fingerprint,
                test_fingerprint=test_set.fingerprint, train_count=len(train_set), test_count=len(test_set))
        prepared = dict(protocol=protocol, code_hashes=hashes, datasets=entries)
        if (HERE / 'prepared.json').exists() and read(HERE / 'prepared.json') != prepared:
            raise ValueError('Previously prepared data or code changed')
        atomic_json(HERE / 'prepared.json', prepared)
        estimates = {}
        for source in COUNTS:
            train_set, test_set = DigitDataset(data_root, source, True), DigitDataset(data_root, source, False)
            calibration = HERE / 'calibration' / source
            short = json.loads(json.dumps(protocol))
            short['ssl']['full_epochs'] = 1
            identity = dict(signature=digest_json(dict(prepared=prepared, calibration=source)),
                            seed=0, selection_method='calibration')
            checkpoint, _ = train_ssl.train(train_set, short, source, 'ResNet18', 'simclr', 0,
                calibration, identity, workers=0, device='cuda', full=True)
            epoch = json.loads((calibration / 'ssl.jsonl').read_text().splitlines()[-1])
            probe_seconds = 0
            for labels in (1, 5):
                result = linear_probe(checkpoint, identity, short, source, source,
                    train_set, test_set, labels, 0, calibration / ('probe_lbl%d.json' % labels),
                    workers=0, device='cuda')
                assert result['labeled_count'] == int(len(train_set) * labels / 100)
                probe_seconds += result['seconds']
            # Project the measured real training epoch; add startup/checkpoint I/O and one minute.
            requested = math.ceil(1000 * epoch['seconds'] + probe_seconds + 120 + 60)
            estimates[source] = dict(epoch_seconds=epoch['seconds'], probe_seconds=probe_seconds,
                startup_checkpoint_seconds=120, buffer_seconds=60, requested_seconds=requested)
            print('MEASURED ESTIMATE:', source, estimates[source], flush=True)
        atomic_json(HERE / 'timing.json', estimates)
    subprocess.run(['bash', str(launcher), '--submit-prepared', str(venv), str(data_root)], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'full'))
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--source', choices=tuple(COUNTS))
    parser.add_argument('--seed', type=int, choices=(0, 1, 2), default=0)
    parser.add_argument('--launcher', type=Path)
    parser.add_argument('--venv', type=Path)
    args = parser.parse_args()
    if (not os.environ.get('SLURM_JOB_ID')
            or os.environ.get('SLURM_CLUSTER_NAME', os.environ.get('CC_CLUSTER')) != 'vulcan'
            or not torch.cuda.is_available() or torch.cuda.device_count() != 1):
        raise RuntimeError('Use the supplied single-GPU Vulcan launcher')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    train_ssl.print = quiet_ssl
    if args.stage == 'prepare':
        if args.launcher is None or args.venv is None:
            parser.error('prepare requires --launcher and --venv')
        prepare(args.data_root, args.launcher, args.venv)
    else:
        if args.source is None:
            parser.error('full requires --source')
        full(args.source, args.seed, args.data_root)


if __name__ == '__main__':
    main()
