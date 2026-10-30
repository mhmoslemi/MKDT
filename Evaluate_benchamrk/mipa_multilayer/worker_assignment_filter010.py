"""One confidence-filtered, single-view cosine MIPA set, one standard SimCLR encoder, two probes per label budget."""
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import statistics
import sys

import torch

HERE = Path(__file__).resolve().parent
BENCH, ROOT = HERE.parent, HERE.parent.parent
sys.path.insert(0, str(ROOT / 'OT-SSL-DD'))
import eval as reference_eval


def read(path):
    return json.loads(Path(path).read_text())


def progress(*values, **kwargs):
    if len(values) == 1 and isinstance(values[0], str) and kwargs.get('file') is None:
        match = re.search(r'iter = (\d+)', values[0])
        if match and int(match[1]) % 200:
            return
        if values[0].startswith('SSL '):
            prefix, line = values[0].split(': ', 1)
            stats = json.loads(line)
            if stats['epoch'] != 1 and stats['epoch'] % 200:
                return
            stats.pop('projector_std', None)
            stats.pop('seconds', None)
            values = (prefix + ': ' + json.dumps(stats),)
    print(*values, **kwargs)


def main():
    manifest, data_root = (Path(value).resolve() for value in sys.argv[1:])
    assert manifest.is_relative_to(HERE / 'batches')
    job = read(manifest)
    assert os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_JOB_ACCOUNT') == job['account'] == 'aip-boyuwang'
    assert os.environ.get('SLURM_CLUSTER_NAME', os.environ.get('CC_CLUSTER')) == 'vulcan'
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    reference = reference_eval.reference_modules()
    common = reference['common']
    snapshot = manifest.parent / 'method1.py'
    for file, sha in job['code_hashes'].items():
        assert common.file_hash(snapshot if file == 'OT-SSL-DD/method1.py' else ROOT / file) == sha, 'Code changed: ' + file
    assert common.file_hash(ROOT / job['selection']['config']) == job['selection']['config_sha256']
    assert common.file_hash(ROOT / job['teacher']['path']) == job['teacher']['file_sha256']
    assert common.file_hash((ROOT / job['teacher']['path']).parent / 'ssl_complete.json') == job['teacher']['marker_sha256']
    args = job['generation_args']
    source = args['dataset']
    similarity = job.get('similarity', 'cosine')
    assert similarity == 'cosine'
    assert source == 'CIFAR10' and args['Iteration'] == 5000 and args['percentage'] == 0.2
    assert args.get('feature_layers', 'last') == 'last' and args['teacher_seed'] == [0]
    assert args.get('cosine_views', 1) == 1 and args['lr_img'] == 0.002
    assert args['assignment_filter'] and args['assignment_threshold'] == 0.1
    assert args['dsa'] and args['disp_weight'] == args['pair_weight'] == 0
    assert args.get('single_channel', False) == (source == 'FashionMNIST')
    assert job['evaluation'] == dict(ssl_epochs=2000, ssl_seed=0, probe_epochs=100, probe_seeds=[0, 1], labels=[1, 5])
    spec = importlib.util.spec_from_file_location('filtered_cosine_method', snapshot)
    method = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(method)
    method.print = progress

    synthetic_path = ROOT / job['synthetic_path']
    folder = manifest.parent / 'results' / source / 'ResNet18BN/pt0.2/simclr/ssl_seed_000'
    with common.exclusive_run(manifest.parent / 'session'):
        original_initialize = method.deepcluster_initialize

        def initialize(dataset, count, device, *positional, **keywords):
            images = original_initialize(dataset, count, device, *positional, **keywords)
            indices = job['selection']['indices']
            assert len(indices) == count
            assert dataset.base.fingerprint == job['selection']['source_fingerprint']
            assert torch.equal(images, torch.stack([method._images_from_batch(dataset[i]) for i in indices]).to(device))
            return images

        method.deepcluster_initialize = initialize
        argv = [str(snapshot)]
        for key, value in args.items():
            if isinstance(value, bool):
                argv.append('--' + (key if value else 'no-' + key))
            else:
                argv += ['--' + key] + [str(v) for v in (value if isinstance(value, list) else [value])]
        sys.argv = argv + ['--device', 'cuda', '--data_path', str(data_root), '--save_path', str(synthetic_path.parent),
                          '--init', 'random', '--similarity', similarity, '--cosine_views', '1', '--feature_layers', 'last', '--distill_aug_backend', 'dsa',
                          '--resume', '--save_every', '500', '--num_eval', '0']
        method.main()
        saved = torch.load(synthetic_path, map_location='cpu', weights_only=True, mmap=True)
        assert saved['complete'] and saved['iteration'] == 5000 and saved['run_args'] == args
        assert saved['iic_params']['feature_layers'] == 'last' and saved['iic_params']['similarity'] == similarity
        assert saved['iic_params']['cosine_views'] == 1
        assert saved['iic_params']['assignment_filter'] and saved['iic_params']['assignment_threshold'] == 0.1
        assert saved['iic_params']['strategy'] == args['distill_aug_strategy']
        images = saved['data']
        assert images.shape == (len(job['selection']['indices']), 3, 32, 32) and torch.isfinite(images).all()
        protocol = read(BENCH / 'protocol.json')
        dataset_class = reference['data'].RawDataset
        train_set, test_set = (dataset_class(data_root, source, training) for training in (True, False))
        assert train_set.fingerprint == job['selection']['source_fingerprint']
        protocol['ssl'].update(subset_epochs=2000, checkpoint_every_epochs=500)
        protocol['evaluation'].update(epochs=100, lr=0.1, batch_size=256)
        protocol['subset_seeds'] = [0]
        mean, std = (torch.tensor(v).view(1, 3, 1, 1) for v in saved['normalization'])
        synthetic = reference_eval.SyntheticImages((images * std + mean).clamp(0, 1).contiguous())
        signature = dict(experiment=HERE.name, source=source, model='ResNet18', subset=0.2, ssl='simclr',
            protocol=protocol, generation_args=args, selection=job['selection'], teacher=job['teacher'],
            synthetic=dict(path=str(synthetic_path), file_sha256=common.file_hash(synthetic_path), normalization=saved['normalization']),
            evaluation=job['evaluation'], code_hashes=common.code_hashes(), driver_code_hashes=job['code_hashes'])
        identity = dict(signature=common.digest_json(signature), seed=0, selection_method='ours')
        config = dict(signature=signature, identity=identity)
        config_file = folder / 'config.json'
        if config_file.exists() and read(config_file) != config:
            raise ValueError('Evaluation inputs changed; refusing to mix results')
        common.atomic_json(config_file, config)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        reference['train_ssl'].print = progress
        checkpoint, trained = reference['train_ssl'].train(synthetic, protocol, source, 'ResNet18', 'simclr',
            0, folder, identity, workers=0, device='cuda')
        assert trained['epochs'] == 2000
        common.atomic_json(folder / 'ssl_complete.json', dict(complete=True, identity=identity, epochs=2000,
            encoder_sha256=trained['encoder_sha256'], checkpoint_file_sha256=common.file_hash(checkpoint)))
        for labels in (1, 5):
            samples = []
            for seed in (0, 1):
                output = folder / 'evaluations' / ('%s_lbl%d_probe%d.json' % (source, labels, seed))
                result = reference['evaluate'].linear_probe(checkpoint, identity, protocol, source, source,
                    train_set, test_set, labels, seed, output, workers=0, device='cuda')
                assert result['complete'] and result['encoder_unchanged'] and result['encoder_sha256'] == trained['encoder_sha256']
                assert math.isfinite(result['accuracy_percent'])
                samples.append(dict(ssl_seed=0, probe_seed=seed, accuracy_percent=result['accuracy_percent'], result_file=str(output)))
            scores = [sample['accuracy_percent'] for sample in samples]
            common.atomic_json(folder / ('summary_%s_lbl%d.json' % (source, labels)), dict(complete=True, identity=identity,
                source=source, target=source, subset=0.2, feature_layers='last', cosine_views=1, assignment_filter=True, assignment_threshold=0.1, similarity=similarity, single_channel=source == 'FashionMNIST',
                ssl_epochs=2000, linear_epochs=100, labels=labels, ssl_seeds=[0], probe_seeds=[0, 1], std_ddof=1,
                samples=samples, mean_percent=statistics.mean(scores), std_percent=statistics.stdev(scores)))
            print('COMPLETE CELL: %s lbl=%d: %.2f +/- %.2f' % (source, labels, statistics.mean(scores), statistics.stdev(scores)), flush=True)
        common.atomic_json(folder / 'complete.json', dict(complete=True, identity=identity, **job['evaluation']))


if __name__ == '__main__':
    main()
