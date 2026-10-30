"""FashionMNIST grayscale-only MIPA, followed by one SimCLR encoder and two probes per budget."""
import ast
import hashlib
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


def main():
    manifest, data_root = (Path(value).resolve() for value in sys.argv[1:])
    assert manifest.is_relative_to(HERE / 'batches')
    job = read(manifest)
    assert os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_JOB_ACCOUNT') == job['account'] == 'aip-yiweilu'
    assert os.environ.get('SLURM_CLUSTER_NAME', os.environ.get('CC_CLUSTER')) == 'vulcan'
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    reference = reference_eval.reference_modules()
    common = reference['common']
    snapshot = manifest.parent / 'method1.py'
    ast.parse(snapshot.read_text(), filename=str(snapshot))
    for file, sha in job['code_hashes'].items():
        assert common.file_hash(snapshot if file == 'OT-SSL-DD/method1.py' else ROOT / file) == sha, 'Code changed: ' + file
    assert common.file_hash(ROOT / job['selection']['config']) == job['selection']['config_sha256']
    assert common.file_hash(ROOT / job['teacher']['path']) == job['teacher']['file_sha256']
    assert common.file_hash((ROOT / job['teacher']['path']).parent / 'ssl_complete.json') == job['teacher']['marker_sha256']
    args = job['generation_args']
    mixed = args.get('distill_aug_backend') == 'simclr'
    assert args['dataset'] == 'FashionMNIST' and args['percentage'] == 0.2 and args['Iteration'] == (10000 if mixed else 6000)
    if mixed:
        assert args['simclr_compose_probability'] == 0.5
    assert args['single_channel'] and args['teacher_seed'] == [0] and args['similarity'] == 'cosine'
    assert job['evaluation'] == dict(ssl_epochs=2000, ssl_seed=0, probe_epochs=100, probe_seeds=[0, 1], labels=[1, 5])
    spec = importlib.util.spec_from_file_location('fashion_singlechannel_method', snapshot)
    method = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(method)

    # A single grayscale parameter receives the summed gradient through all RGB copies.
    gray = torch.zeros(2, 1, 4, 4, requires_grad=True)
    optimizer = torch.optim.Adam([gray], lr=0.002)
    (gray.expand(-1, 3, -1, -1) * torch.tensor([1., 2., 3.]).view(1, 3, 1, 1)).sum().backward()
    assert torch.equal(gray.grad, torch.full_like(gray, 6))
    optimizer.step()
    assert optimizer.state[gray]['exp_avg'].shape == gray.shape
    print('CHECK PASSED: one-channel Adam state and differentiable RGB repetition', flush=True)

    synthetic_path = ROOT / job['synthetic_path']
    folder = HERE / ('results_simclr_mix10k' if mixed else 'results') / 'FashionMNIST/ResNet18BN/pt0.2/simclr/ssl_seed_000'
    with common.exclusive_run(HERE / ('session_simclr_mix10k' if mixed else 'session')):
        original_initialize = method.deepcluster_initialize

        def initialize(dataset, count, device, *positional, **keywords):
            images = original_initialize(dataset, count, device, *positional, **keywords)
            indices = job['selection']['indices']
            assert len(indices) == count == 120
            assert torch.equal(images, torch.stack([method._images_from_batch(dataset[i]) for i in indices]).to(device))
            return images

        def progress(*values, **kwargs):
            if len(values) == 1 and isinstance(values[0], str):
                match = re.search(r'iter = (\d+)', values[0])
                if match and int(match[1]) % 200:
                    return
                if kwargs.get('file') is None and values[0].startswith('SSL '):
                    prefix, line = values[0].split(': ', 1)
                    stats = json.loads(line)
                    if stats['epoch'] != 1 and stats['epoch'] % 200:
                        return
                    stats.pop('projector_std', None)
                    stats.pop('seconds', None)
                    values = (prefix + ': ' + json.dumps(stats),)
            print(*values, **kwargs)

        method.deepcluster_initialize, method.print = initialize, progress
        argv = [str(snapshot)]
        for key, value in args.items():
            if isinstance(value, bool):
                argv.append('--' + (key if value else 'no-' + key))
            else:
                argv += ['--' + key] + [str(v) for v in (value if isinstance(value, list) else [value])]
        sys.argv = argv + ['--device', 'cuda', '--data_path', str(data_root), '--save_path', str(synthetic_path.parent),
                           '--resume', '--save_every', '500', '--num_eval', '0']
        method.main()
        saved = torch.load(synthetic_path, map_location='cpu', weights_only=True, mmap=True)
        assert saved['complete'] and saved['iteration'] == args['Iteration'] and saved['run_args']['single_channel']
        images = saved['data']
        assert images.shape == (120, 3, 32, 32) and torch.isfinite(images).all()
        assert torch.equal(images, images[:, :1].expand_as(images))
        states = list(saved['optimizer']['state'].values())
        assert len(states) == 1 and states[0]['exp_avg'].shape == (120, 1, 32, 32)
        print('VERIFIED: optimized [120,1,32,32], saved [120,3,32,32] with identical channels', flush=True)

        prepared = read(BENCH / 'fashionmnist/full/prepared.json')
        protocol = prepared['protocol']
        protocol['ssl'].update(subset_epochs=2000, checkpoint_every_epochs=500)
        protocol['evaluation'].update(epochs=100, lr=0.1, batch_size=256)
        protocol['subset_seeds'] = [0]
        dataset_class = reference['_fashion_dataset'].DigitDataset
        train_set, test_set = (dataset_class(data_root, 'FashionMNIST', training) for training in (True, False))
        assert train_set.fingerprint == job['selection']['source_fingerprint']
        assert test_set.fingerprint == prepared['datasets']['FashionMNIST']['test_fingerprint']
        assert saved['normalization'] == protocol['normalization']['FashionMNIST']
        mean, std = (torch.tensor(v).view(1, 3, 1, 1) for v in saved['normalization'])
        synthetic = reference_eval.SyntheticImages((images * std + mean).clamp(0, 1).contiguous())
        signature = dict(experiment=HERE.name, source='FashionMNIST', model='ResNet18', subset=0.2, ssl='simclr',
            protocol=protocol, generation_args=args, selection=job['selection'], teacher=job['teacher'],
            synthetic=dict(path=str(synthetic_path), file_sha256=common.file_hash(synthetic_path),
                images_sha256=hashlib.sha256(images.contiguous().numpy().tobytes()).hexdigest(),
                normalization=saved['normalization'], pixel_policy='denormalize_clamp_then_PIL_uint8'),
            ssl_seeds=[0], probe_seeds=[0, 1], labels=[1, 5], code_hashes=common.code_hashes(), driver_code_hashes=job['code_hashes'])
        identity = dict(signature=common.digest_json(signature), seed=0, selection_method='ours')
        config = dict(signature=signature, identity=identity)
        config_file = folder / 'config.json'
        if config_file.exists() and read(config_file) != config:
            raise ValueError('Evaluation inputs changed; refusing to mix results')
        common.atomic_json(config_file, config)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        reference['train_ssl'].print = progress
        checkpoint, trained = reference['train_ssl'].train(synthetic, protocol, 'FashionMNIST', 'ResNet18', 'simclr',
            0, folder, identity, workers=0, device='cuda')
        assert trained['epochs'] == 2000
        common.atomic_json(folder / 'ssl_complete.json', dict(complete=True, identity=identity, epochs=2000,
            encoder_sha256=trained['encoder_sha256'], checkpoint_file_sha256=common.file_hash(checkpoint)))
        for labels in (1, 5):
            samples = []
            for seed in (0, 1):
                output = folder / 'evaluations' / ('FashionMNIST_lbl%d_probe%d.json' % (labels, seed))
                result = reference['evaluate'].linear_probe(checkpoint, identity, protocol, 'FashionMNIST', 'FashionMNIST',
                    train_set, test_set, labels, seed, output, workers=0, device='cuda')
                assert result['complete'] and result['encoder_unchanged'] and result['encoder_sha256'] == trained['encoder_sha256']
                assert math.isfinite(result['accuracy_percent'])
                samples.append(dict(ssl_seed=0, probe_seed=seed, accuracy_percent=result['accuracy_percent'], result_file=str(output)))
            scores = [sample['accuracy_percent'] for sample in samples]
            common.atomic_json(folder / ('summary_FashionMNIST_lbl%d.json' % labels), dict(complete=True, identity=identity,
                source='FashionMNIST', target='FashionMNIST', subset=0.2, single_channel=True, ssl_epochs=2000,
                linear_epochs=100, labels=labels, ssl_seeds=[0], probe_seeds=[0, 1], std_ddof=1, samples=samples,
                mean_percent=statistics.mean(scores), std_percent=statistics.stdev(scores)))
            print('COMPLETE CELL: FashionMNIST lbl=%d: %.2f +/- %.2f' % (labels, statistics.mean(scores), statistics.stdev(scores)), flush=True)
        common.atomic_json(folder / 'complete.json', dict(complete=True, identity=identity, ssl_epochs=2000, labels=[1, 5], probe_seeds=[0, 1]))


if __name__ == '__main__':
    main()
