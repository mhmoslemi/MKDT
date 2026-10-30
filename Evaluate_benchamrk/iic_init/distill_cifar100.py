"""IIC-initialized MIPA, then reference SimCLR and three probes per label budget."""
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import sys

import torch

HERE = Path(__file__).resolve().parent
BENCH, ROOT = HERE.parent, HERE.parent.parent
sys.path.insert(0, str(BENCH))
sys.path.insert(0, str(ROOT / 'OT-SSL-DD'))
import eval as reference_eval
from mipa_multilayer.worker import progress


def read(path):
    return json.loads(Path(path).read_text())


def main(manifest):
    job = read(manifest)
    assert manifest.is_relative_to(HERE / 'distillation')
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
    selection = job['selection']
    assert common.file_hash(ROOT / selection['config']) == selection['config_sha256']
    assert len(selection['indices']) == len(set(selection['indices'])) == 100
    assert selection['method'] == 'iic' and selection['epoch'] == 80
    teacher = ROOT / job['teacher']['path']
    assert common.file_hash(teacher) == job['teacher']['file_sha256']
    assert common.file_hash(teacher.parent / 'ssl_complete.json') == job['teacher']['marker_sha256']
    args = job['generation_args']
    assert args['dataset'] == 'CIFAR100' and args['percentage'] == 0.2 and args['Iteration'] == 5000
    assert args['batch_real'] == 128 and args['lr_img'] == 0.002 and args['teacher_seed'] == [0]
    assert args['dsa'] and args['disp_weight'] == args['pair_weight'] == 0 and args['temperature'] == 0.2
    assert job['evaluation'] == dict(ssl_epochs=2000, ssl_seed=0, probe_epochs=100, probe_seeds=[0, 1, 2], labels=[1, 5])
    spec = importlib.util.spec_from_file_location('iic_initialized_method', snapshot)
    method = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(method)
    method.print = progress
    synthetic_path = ROOT / job['synthetic_path']
    folder = manifest.parent / 'results/CIFAR100/ResNet18BN/pt0.2/simclr/ssl_seed_000'
    with common.exclusive_run(manifest.parent / 'session'):
        def initialize(dataset, count, device, *unused, **keywords):
            assert dataset.base.fingerprint == selection['source_fingerprint'] and count == len(selection['indices'])
            print('Initializing 100 images from saved epoch-80 IIC main clusters (no labels used).', flush=True)
            return torch.stack([method._images_from_batch(dataset[index]) for index in selection['indices']]).to(device)

        method.deepcluster_initialize = initialize
        argv = [str(snapshot)]
        for key, value in args.items():
            if isinstance(value, bool):
                argv.append('--' + (key if value else 'no-' + key))
            else:
                argv += ['--' + key] + [str(item) for item in (value if isinstance(value, list) else [value])]
        previous_argv = sys.argv
        sys.argv = argv + ['--device', 'cuda', '--data_path', job['data_root'], '--save_path', str(synthetic_path.parent),
                          '--init', 'iic', '--similarity', 'cosine', '--cosine_views', '1', '--feature_layers', 'last',
                          '--distill_aug_backend', 'dsa', '--resume', '--save_every', '500', '--num_eval', '0']
        try:
            method.main()
        finally:
            sys.argv = previous_argv
        saved = torch.load(synthetic_path, map_location='cpu', weights_only=True, mmap=True)
        assert saved['complete'] and saved['iteration'] == 5000 and saved['run_args'] == args and saved['init'] == 'iic'
        params = saved['iic_params']
        assert params['similarity'] == 'cosine' and params['feature_layers'] == 'last' and params['cosine_views'] == 1
        assert not params['assignment_filter'] and params['strategy'] == args['distill_aug_strategy']
        images = saved['data']
        assert images.shape == (100, 3, 32, 32) and torch.isfinite(images).all()
        protocol = read(BENCH / 'protocol.json')
        data = reference['data']
        train_set, test_set = (data.RawDataset(job['data_root'], 'CIFAR100', training) for training in (True, False))
        assert train_set.fingerprint == selection['source_fingerprint']
        from torchvision import transforms
        normalized = data.ImageView(train_set, transforms.Compose([transforms.ToTensor(), transforms.Normalize(*saved['normalization'])]))
        expected_initial = torch.stack([normalized[index][0] for index in selection['indices']])
        assert torch.equal(expected_initial, saved['initial_data']), 'Distillation did not start from the saved IIC images'
        protocol['ssl'].update(subset_epochs=2000, checkpoint_every_epochs=500)
        protocol['evaluation'].update(epochs=100, lr=0.1, batch_size=256)
        protocol['subset_seeds'] = [0]
        mean, std = (torch.tensor(value).view(1, 3, 1, 1) for value in saved['normalization'])
        synthetic = reference_eval.SyntheticImages((images * std + mean).clamp(0, 1).contiguous())
        signature = dict(experiment='iic_initialized_mipa', source='CIFAR100', model='ResNet18', subset=0.2, ssl='simclr',
            protocol=protocol, generation_args=args, selection=selection, teacher=job['teacher'],
            synthetic=dict(path=str(synthetic_path), file_sha256=common.file_hash(synthetic_path), normalization=saved['normalization']),
            evaluation=job['evaluation'], code_hashes=common.code_hashes(), driver_code_hashes=job['code_hashes'])
        identity = dict(signature=common.digest_json(signature), seed=0, selection_method='ours')
        config = dict(signature=signature, identity=identity)
        config_file = folder / 'config.json'
        if config_file.exists() and read(config_file) != config:
            raise ValueError('Evaluation inputs changed; refusing to mix results')
        common.atomic_json(config_file, config)
        reference['train_ssl'].print = progress
        checkpoint, trained = reference['train_ssl'].train(synthetic, protocol, 'CIFAR100', 'ResNet18', 'simclr',
            0, folder, identity, workers=0, device='cuda')
        assert trained['epochs'] == 2000
        common.atomic_json(folder / 'ssl_complete.json', dict(complete=True, identity=identity, epochs=2000,
            encoder_sha256=trained['encoder_sha256'], checkpoint_file_sha256=common.file_hash(checkpoint)))
        for labels in (1, 5):
            samples = []
            for seed in (0, 1, 2):
                output = folder / 'evaluations' / ('CIFAR100_lbl%d_probe%d.json' % (labels, seed))
                result = reference['evaluate'].linear_probe(checkpoint, identity, protocol, 'CIFAR100', 'CIFAR100',
                    train_set, test_set, labels, seed, output, workers=0, device='cuda')
                assert result['complete'] and result['encoder_unchanged'] and result['encoder_sha256'] == trained['encoder_sha256']
                assert math.isfinite(result['accuracy_percent'])
                samples.append(dict(ssl_seed=0, probe_seed=seed, accuracy_percent=result['accuracy_percent'], result_file=str(output)))
            scores = [sample['accuracy_percent'] for sample in samples]
            common.atomic_json(folder / ('summary_CIFAR100_lbl%d.json' % labels), dict(complete=True, identity=identity,
                source='CIFAR100', target='CIFAR100', subset=0.2, initialization='iic_epoch80', similarity='cosine', cosine_views=1,
                ssl_epochs=2000, linear_epochs=100, labels=labels, ssl_seeds=[0], probe_seeds=[0, 1, 2], std_ddof=1,
                samples=samples, mean_percent=statistics.mean(scores), std_percent=statistics.stdev(scores)))
            print('COMPLETE CELL: CIFAR100 lbl=%d: %.2f +/- %.2f' % (labels, statistics.mean(scores), statistics.stdev(scores)), flush=True)
        assert common.file_hash(teacher) == job['teacher']['file_sha256']
        common.atomic_json(folder / 'complete.json', dict(complete=True, identity=identity, teacher_unchanged=True, **job['evaluation']))


if __name__ == '__main__':
    assert len(sys.argv) == 2
    main(Path(sys.argv[1]).resolve())
