"""Digit Random/K-means: one 2k SimCLR encoder, five probes per label budget."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import statistics
import sys

HERE = Path(__file__).resolve().parent
BENCH, ROOT = HERE.parent, HERE.parent.parent
sys.path.insert(0, str(BENCH))

import torch
from common import atomic_json, code_hashes, digest_json, exclusive_run, file_hash, load_encoder
from data import select_subset
from evaluate import linear_probe
from run import validate_protocol
import train_ssl

spec = importlib.util.spec_from_file_location('subset_digit_data', BENCH / 'digits_simclr/datasets.py')
digit_data = importlib.util.module_from_spec(spec)
spec.loader.exec_module(digit_data)


def read(path):
    return json.loads(Path(path).read_text())


def progress(*values, **kwargs):
    if kwargs.get('file') is None and len(values) == 1 and str(values[0]).startswith('SSL '):
        prefix, line = values[0].split(': ', 1)
        row = json.loads(line)
        if row['epoch'] not in (1, 2000) and row['epoch'] % 200:
            return
        row.pop('projector_std', None)
        row.pop('seconds', None)
        values = (prefix + ': ' + json.dumps(row),)
    print(*values, **kwargs)


def summarize(output, source, kind, subset, labels, identity, encoder_hash):
    samples = []
    for seed in range(5):
        path = output / 'evaluations' / ('%s_lbl%d_probe%d.json' % (source, labels, seed))
        if not path.exists():
            continue
        row = read(path)
        if (not row['complete'] or not row['encoder_unchanged'] or row['expected']['identity'] != identity
                or row['encoder_sha256'] != encoder_hash
                or row['expected']['evaluation_code_hashes'] != code_hashes()
                or [row['expected']['source'], row['expected']['target'], row['expected']['label_percentage'],
                    row['expected']['seed']] != [source, source, labels, seed]):
            raise ValueError('Invalid saved digit probe: ' + str(path))
        samples.append(dict(ssl_seed=0, probe_seed=seed, accuracy_percent=row['accuracy_percent'],
                            encoder_sha256=encoder_hash, result_file=str(path)))
    result = dict(source=source, target=source, model='ResNet18', kind=kind, subset=subset,
        ssl='simclr', ssl_seed=0, ssl_epochs=2000, linear_epochs=100, labels=labels,
        complete=len(samples) == 5, completed_probes=len(samples), expected_probes=5,
        std_ddof=1, samples=samples, table_scope='CVPR_tables/table.tex')
    if result['complete']:
        scores = [row['accuracy_percent'] for row in samples]
        result.update(mean_percent=statistics.mean(scores), std_percent=statistics.stdev(scores))
        result['tex_value'] = '\\dsacc{%.2f}{%.2f}' % (result['mean_percent'], result['std_percent'])
    atomic_json(output / ('summary_lbl%d.json' % labels), result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', choices=tuple(digit_data.COUNTS), required=True)
    parser.add_argument('--method', choices=('random', 'kmeans'), required=True)
    parser.add_argument('--subset', type=float, choices=(0.2, 1, 2), required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--worker-sha256', required=True)
    args = parser.parse_args()
    if (not os.environ.get('SLURM_JOB_ID')
            or os.environ.get('SLURM_CLUSTER_NAME', os.environ.get('CC_CLUSTER')) != 'vulcan'
            or not torch.cuda.is_available() or torch.cuda.device_count() != 1):
        raise RuntimeError('Use the supplied single-GPU Vulcan launcher')
    if file_hash(__file__) != args.worker_sha256:
        raise ValueError('Subset worker changed since submission')
    prepared = read(BENCH / 'digits_simclr/prepared.json')
    for relative, expected in prepared['code_hashes'].items():
        if file_hash(ROOT / relative) != expected:
            raise ValueError('Prepared digit/shared code changed: ' + relative)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    protocol = prepared['protocol']
    protocol['ssl'].update(subset_epochs=2000, checkpoint_every_epochs=500)
    protocol['subset_seeds'] = [0]
    validate_protocol(protocol)
    assert protocol['evaluation']['epochs'] == 100 and protocol['ssl']['batch_size'] == 256
    source, kind = args.source, args.method
    train_set = digit_data.DigitDataset(args.data_root, source, True)
    test_set = digit_data.DigitDataset(args.data_root, source, False)
    expected = prepared['datasets'][source]
    if (train_set.fingerprint != expected['train_fingerprint']
            or test_set.fingerprint != expected['test_fingerprint']):
        raise ValueError('Digit source split or ordering changed')
    output = HERE / 'results' / source / 'ResNet18' / kind / ('pt%g' % args.subset) / 'simclr/ssl_seed_000'
    with exclusive_run(output):
        selected, selection = select_subset(train_set, kind, args.subset, 0, protocol,
                                            HERE / 'selections', workers=0, device='cuda')
        assert len(selected) == int(len(train_set) * args.subset / 100)
        print('SELECTED %s / %s / %g%%: %d distinct images; selection seed=0, SSL seed=0' % (
            source, kind, args.subset, len(selected)), flush=True)
        signature = dict(experiment='digits_random_kmeans_simclr2k_1ssl5probes',
            source=source, source_fingerprint=train_set.fingerprint, test_fingerprint=test_set.fingerprint,
            model='ResNet18', selection_method=kind, subset=args.subset, ssl='simclr', protocol=protocol,
            code_hashes=code_hashes(), worker_sha256=args.worker_sha256,
            digit_adapter_sha256=file_hash(BENCH / 'digits_simclr/datasets.py'),
            probe_seeds=list(range(5)), label_percentages=[1, 5], table_scope='CVPR_tables/table.tex')
        identity = dict(signature=digest_json(signature), seed=0, selection_method=kind, selection=selection)
        config = dict(signature=signature, identity=identity)
        if (output / 'config.json').exists() and read(output / 'config.json') != config:
            raise ValueError('Saved digit subset run has different settings')
        atomic_json(output / 'config.json', config)
        atomic_json(output / 'environment.json', dict(job_id=os.environ['SLURM_JOB_ID'],
            account=os.environ.get('SLURM_JOB_ACCOUNT'), gpu=torch.cuda.get_device_name(),
            python=sys.version, torch=str(torch.__version__)))
        train_ssl.print = progress
        checkpoint, metadata = train_ssl.train(selected, protocol, source, 'ResNet18', 'simclr',
            0, output, identity, workers=0, device='cuda')
        batch = min(256, len(selected))
        assert metadata['epochs'] == 2000 and metadata['steps'] == 2000 * (len(selected) // batch)
        checked, _, saved = load_encoder(checkpoint, identity, device='cpu')
        assert saved['encoder_sha256'] == metadata['encoder_sha256']
        del checked, saved
        atomic_json(output / 'ssl_complete.json', dict(complete=True, epochs=2000, identity=identity,
            encoder_sha256=metadata['encoder_sha256'], checkpoint_file_sha256=file_hash(checkpoint)))
        for labels in (1, 5):
            for seed in range(5):
                path = output / 'evaluations' / ('%s_lbl%d_probe%d.json' % (source, labels, seed))
                result = linear_probe(checkpoint, identity, protocol, source, source, train_set, test_set,
                    labels, seed, path, workers=0, device='cuda')
                assert result['complete'] and result['encoder_unchanged']
                summary = summarize(output, source, kind, args.subset, labels, identity, metadata['encoder_sha256'])
            assert summary['complete']
        atomic_json(output / 'complete.json', dict(complete=True, identity=identity,
            ssl_epochs=2000, linear_epochs=100, probes_per_cell=5, completed_probes=10))
        print('COMPLETE %s / %s / %g%%: both five-probe main cells saved.' % (
            source, kind, args.subset), flush=True)


if __name__ == '__main__':
    main()
