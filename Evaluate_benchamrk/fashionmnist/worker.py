"""Isolated FashionMNIST adapter for the existing full, subset and MKDT workers."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
BENCH, ROOT = HERE.parent, HERE.parent.parent
sys.path[:0] = [str(BENCH / 'digits_simclr'), str(BENCH), str(ROOT)]

import numpy as np
import torch

from common import atomic_json, code_hashes, digest_json, exclusive_run, file_hash, load_encoder
from data import select_subset
from evaluate import linear_probe
from run import validate_protocol
import train_ssl


def module_at(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read(path):
    return json.loads(Path(path).read_text())


fashion = module_at('fashion_data', HERE / 'dataset.py')
full_worker = module_at('fashion_full', BENCH / 'digits_simclr/worker.py')
subset_worker = module_at('fashion_subset', BENCH / 'digits_subset_simclr2k/worker.py')
mkdt_worker = module_at('fashion_mkdt', BENCH / 'digits_mkdt/worker.py')
original_mkdt_hashes = mkdt_worker.hashes


def hashes():
    files = [Path(__file__), HERE / 'dataset.py', Path(full_worker.__file__),
             Path(subset_worker.__file__)]
    return {**original_mkdt_hashes(),
            **{str(path.relative_to(ROOT)): file_hash(path) for path in files}}


# These are process-local adapters; no shared source or running-job configs change.
full_worker.HERE = HERE / 'full'
full_worker.COUNTS, full_worker.DigitDataset = fashion.COUNTS, fashion.DigitDataset
full_worker.implementation = hashes
mkdt_worker.HERE = HERE / 'mkdt'
mkdt_worker.digit_data, mkdt_worker.hashes = fashion, hashes


def save_config(path, value):
    if path.exists() and read(path) != value:
        raise ValueError('Existing FashionMNIST settings/data/code differ: ' + str(path))
    atomic_json(path, value)


def prepare(data_root):
    with exclusive_run(HERE / 'preparation'):
        train_set = fashion.DigitDataset(data_root, 'FashionMNIST', True)
        test_set = fashion.DigitDataset(data_root, 'FashionMNIST', False)
        assert np.array_equal(np.bincount(train_set.targets), np.full(10, 6000))
        assert np.array_equal(np.bincount(test_set.targets), np.full(10, 1000))
        image, _ = train_set[0]
        assert image.mode == 'RGB' and image.size == (32, 32)
        protocol = read(BENCH / 'protocol.json')
        protocol['sources'], protocol['models'], protocol['full_seeds'] = ['FashionMNIST'], ['ResNet18'], [0, 1, 2]
        protocol['ssl'].update(full_epochs=1000, subset_epochs=2000, checkpoint_every_epochs=50)
        normalization = train_set.normalization()
        protocol['normalization']['FashionMNIST'] = normalization
        protocol['input_size']['FashionMNIST'] = 32
        validate_protocol(protocol)
        entries = dict(FashionMNIST=dict(train_count=len(train_set), test_count=len(test_set),
            train_fingerprint=train_set.fingerprint, test_fingerprint=test_set.fingerprint))
        prepared = dict(protocol=protocol, code_hashes=hashes(), datasets=entries)
        save_config(HERE / 'full/prepared.json', prepared)
        cfg = dict(data_root=str(data_root.resolve()), code_hashes=hashes(), datasets=entries,
            normalization={'FashionMNIST': normalization},
            teacher=dict(mkdt_worker.original_teacher.PAPER_DEFAULTS),
            experts=dict(model='ConvNet', normalization='original_MKDT_instancenorm', count=100,
                epochs=20, batch_size=256, optimizer='sgd', lr=0.1, momentum=0.9, weight_decay=1e-4,
                loss='mse', snapshots='initial_plus_each_epoch', seeds=list(range(100))))
        save_config(HERE / 'mkdt/recipe.json', cfg)
        # Sanity-check both existing encoder families on the new source adapter.
        backbone = mkdt_worker.original_teacher.build_backbone('FashionMNIST')
        assert backbone.conv1.in_channels == 3 and isinstance(backbone.maxpool, torch.nn.Identity)
        atomic_json(HERE / 'ready.json', dict(complete=True, worker_sha256=file_hash(__file__),
            train_count=len(train_set), test_count=len(test_set), normalization=normalization))
        print('READY FashionMNIST: 60000 train / 10000 test, 32px RGB; normalization:', normalization, flush=True)


def subset(args):
    source = 'FashionMNIST'
    prepared = read(HERE / 'full/prepared.json')
    if prepared['code_hashes'] != hashes():
        raise ValueError('Prepared FashionMNIST/shared implementation changed')
    protocol = prepared['protocol']
    protocol['ssl'].update(subset_epochs=2000, checkpoint_every_epochs=500)
    protocol['subset_seeds'] = [0]
    validate_protocol(protocol)
    train_set, test_set = fashion.DigitDataset(args.data_root, source, True), fashion.DigitDataset(args.data_root, source, False)
    expected = prepared['datasets'][source]
    if (train_set.fingerprint != expected['train_fingerprint']
            or test_set.fingerprint != expected['test_fingerprint']):
        raise ValueError('FashionMNIST split/order changed')
    output = HERE / 'subset/results' / source / 'ResNet18' / args.method / ('pt%g' % args.subset) / 'simclr/ssl_seed_000'
    with exclusive_run(output):
        selected, selection = select_subset(train_set, args.method, args.subset, 0, protocol,
            HERE / 'subset/selections', workers=0, device='cuda')
        assert len(selected) == int(len(train_set) * args.subset / 100)
        signature = dict(experiment='fashionmnist_simclr2k_1ssl5probes', source=source,
            source_fingerprint=train_set.fingerprint, test_fingerprint=test_set.fingerprint,
            model='ResNet18', selection_method=args.method, subset=args.subset, ssl='simclr',
            protocol=protocol, code_hashes=code_hashes(), implementation=hashes(),
            probe_seeds=list(range(5)), label_percentages=[1, 5], table_scope='CVPR_tables/table.tex')
        identity = dict(signature=digest_json(signature), seed=0, selection_method=args.method, selection=selection)
        save_config(output / 'config.json', dict(signature=signature, identity=identity))
        atomic_json(output / 'environment.json', dict(job_id=os.environ['SLURM_JOB_ID'],
            account=os.environ.get('SLURM_JOB_ACCOUNT'), gpu=torch.cuda.get_device_name(),
            python=sys.version, torch=str(torch.__version__)))
        checkpoint, metadata = train_ssl.train(selected, protocol, source, 'ResNet18', 'simclr',
            0, output, identity, workers=0, device='cuda')
        assert metadata['epochs'] == 2000 and metadata['steps'] == 2000 * (len(selected) // min(256, len(selected)))
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
                summary = subset_worker.summarize(output, source, args.method, args.subset, labels,
                    identity, metadata['encoder_sha256'])
            assert summary['complete']
        atomic_json(output / 'complete.json', dict(complete=True, identity=identity, ssl_epochs=2000,
            linear_epochs=100, probes_per_cell=5, completed_probes=10))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'full', 'subset', 'teacher', 'features', 'experts', 'init'))
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--worker-sha256', required=True)
    parser.add_argument('--seed', type=int, choices=(0, 1, 2), default=0)
    parser.add_argument('--method', choices=('random', 'kmeans'))
    parser.add_argument('--subset', type=float, choices=(0.2, 1, 2))
    parser.add_argument('--start', type=int, choices=range(0, 100, 5), default=0)
    args = parser.parse_args()
    if (not os.environ.get('SLURM_JOB_ID')
            or os.environ.get('SLURM_CLUSTER_NAME', os.environ.get('CC_CLUSTER')) != 'vulcan'):
        raise RuntimeError('Use the Vulcan submitter')
    if file_hash(__file__) != args.worker_sha256:
        raise ValueError('FashionMNIST worker changed since submission')
    if args.stage not in ('prepare', 'init') and (not torch.cuda.is_available() or torch.cuda.device_count() != 1):
        raise RuntimeError('Exactly one allocated GPU is required')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.stage in ('teacher', 'features', 'experts', 'init'):
        torch.backends.cudnn.benchmark = True
    train_ssl.print = subset_worker.progress
    if args.stage == 'prepare':
        prepare(args.data_root)
    elif args.stage == 'full':
        full_worker.full('FashionMNIST', args.seed, args.data_root)
    elif args.stage == 'subset':
        if args.method is None or args.subset is None:
            parser.error('subset requires --method and --subset')
        subset(args)
    elif args.stage == 'teacher':
        mkdt_worker.teacher('FashionMNIST', args.data_root)
    elif args.stage == 'features':
        mkdt_worker.features('FashionMNIST', args.data_root)
    elif args.stage == 'experts':
        mkdt_worker.experts('FashionMNIST', args.start, args.data_root)
    else:
        mkdt_worker.initialize('FashionMNIST')


if __name__ == '__main__':
    main()
