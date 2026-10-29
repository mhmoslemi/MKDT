"""Use the existing MIPA pipeline with the official digit dataset adapters."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
BENCH, ROOT = HERE.parent, HERE.parent.parent
SOURCES = ('MNIST', 'FashionMNIST', 'SVHN')
COUNTS = {'MNIST': 60000, 'FashionMNIST': 60000, 'SVHN': 73257}
ITERATIONS = {0.2: 10000, 1: 8000, 2: 6000}


def module_at(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = module_at('existing_mipa_pipeline', BENCH / 'mipa_random128/worker.py')
sys.modules['common'] = base.reference_eval.reference_modules()['common']
digits = module_at('datasets', BENCH / 'digits_simclr/datasets.py')
sys.modules['datasets'] = digits
fashion = module_at('mipa_fashion_data', BENCH / 'fashionmnist/dataset.py')
base.HERE, base.SOURCES = HERE, SOURCES
base.SYNTHETICS = ROOT / 'OT-SSL-DD/ours_syntehtic_digits_random'


def generation_args(source, subset):
    teacher_root = BENCH / ('fashionmnist/full/checkpoints' if source == 'FashionMNIST' else 'digits_simclr/checkpoints')
    return dict(dataset=source, percentage=subset, model='ResNet18BN', seed=0,
        Iteration=ITERATIONS[subset], lr_img=0.001, batch_real=1024 if subset == 0.2 else 128,
        temperature=0.2, disp_weight=0.0, pair_weight=0.0, num_random_networks=1,
        random_models='ResNet18BN', dsa=True, distill_aug_strategy='color_crop_cutout_flip_scale_rotate',
        distill_aug_mode='M', teacher_source='pretrained', teacher_seed=[0], teacher_ssl='simclr',
        teacher_path=str(teacher_root), feature_batch=128)


class DatasetParser(argparse.ArgumentParser):
    def add_argument(self, *names, **kwargs):
        if names == ('--dataset',):
            kwargs['choices'] = SOURCES
        return super().add_argument(*names, **kwargs)


def configure(job, data_root):
    reference = base.reference_eval.reference_modules()
    common = reference['common']
    assert job['source'] in SOURCES and job['subset'] in ITERATIONS
    assert job['generation_args'] == generation_args(job['source'], job['subset'])
    for file, sha in job['code_hashes'].items():
        assert common.file_hash(ROOT / file) == sha, 'Code changed: ' + file
    prepared_file = ROOT / job['prepared_file']
    assert common.file_hash(prepared_file) == job['prepared_sha256']
    prepared = base.read(prepared_file)
    for file, sha in prepared['code_hashes'].items():
        assert common.file_hash(ROOT / file) == sha, 'Teacher implementation changed: ' + file
    origin = ROOT / job['selection']['config']
    assert common.file_hash(origin) == job['selection']['config_sha256']
    previous = base.read(origin)
    assert previous['identity']['seed'] == 0 and previous['identity']['selection_method'] == 'random'
    assert previous['identity']['selection']['indices'] == job['selection']['indices']
    assert previous['signature']['source_fingerprint'] == job['selection']['source_fingerprint']
    teacher_config = ROOT / job['teacher']['config']
    assert common.file_hash(teacher_config) == job['teacher']['config_sha256']
    cfg = base.read(teacher_config)
    assert cfg['identity'] == job['teacher']['identity']
    assert cfg['signature']['source'] == job['source'] and cfg['signature']['ssl'] == 'simclr'
    assert cfg['signature']['selection_method'] == 'full' and cfg['identity']['seed'] == 0
    assert cfg['signature']['protocol'] == prepared['protocol']
    assert common.digest_json(cfg['signature']) == cfg['identity']['signature']

    def raw_dataset(root, source, training):
        assert source in SOURCES
        cls = fashion.DigitDataset if source == 'FashionMNIST' else digits.DigitDataset
        return cls(root, source, training)

    reference['data'].RawDataset = raw_dataset
    reference['data'].SOURCE_COUNTS.update(COUNTS)
    source = job['source']
    base.reference_eval.SYNTHETIC_NORMALIZATION[source] = prepared['protocol']['normalization'][source]
    original_read = base.read

    def read(path):
        if Path(path).resolve() == BENCH / 'protocol.json':
            return json.loads(json.dumps(prepared['protocol']))
        return original_read(path)

    base.read, base.generation_args = read, generation_args
    base.method1.argparse = SimpleNamespace(**dict(vars(argparse), ArgumentParser=DatasetParser))
    job['data_root'] = str(data_root)
    return reference, prepared


def check(job, data_root):
    reference, prepared = configure(job, data_root)
    common, data = reference['common'], reference['data']
    source = job['source']
    parser = DatasetParser(add_help=False)
    parser.add_argument('--dataset', choices=('CIFAR10',))
    assert parser.parse_args(['--dataset', source]).dataset == source
    raw = data.RawDataset(data_root, source, True)
    assert raw.fingerprint == job['selection']['source_fingerprint']
    assert raw.fingerprint == prepared['datasets'][source]['train_fingerprint']
    normalization = prepared['protocol']['normalization'][source]
    transform = base.method1.transforms.Compose([base.method1.transforms.ToTensor(),
        base.method1.transforms.Normalize(*normalization)])
    selected = base.initial_images(job, data.ImageView(raw, transform), len(job['selection']['indices']), 'cpu')
    assert selected.shape == (int(COUNTS[source] * job['subset'] / 100), 3, 32, 32)
    assert base.torch.isfinite(selected).all()
    encoder, _ = common.build_encoder('ResNet18', 32)
    encoder.eval().requires_grad_(False)
    images = selected[:2].clone().requires_grad_(True)
    encoder(images).square().mean().backward()
    assert images.grad is not None and base.torch.isfinite(images.grad).all()
    assert all(p.grad is None for p in encoder.parameters())
    assert prepared['protocol']['ssl']['batch_size'] == 256
    assert prepared['protocol']['evaluation']['epochs'] == 100
    print('CHECK PASSED:', source, job['subset'], 'exact Random indices, normalization, BN features and image gradients', flush=True)


def main():
    manifest, data_root = (Path(p).resolve() for p in sys.argv[1:3])
    assert manifest.is_relative_to(HERE / 'batches')
    job = json.loads(manifest.read_text())
    assert os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_JOB_ACCOUNT') == job['account']
    base.torch.set_num_threads(1)
    base.torch.set_num_interop_threads(1)
    if sys.argv[3:] == ['--check']:
        check(job, data_root)
        return
    assert not sys.argv[3:]
    assert os.environ.get('SLURM_CLUSTER_NAME', os.environ.get('CC_CLUSTER')) == 'vulcan'
    assert base.torch.cuda.is_available() and base.torch.cuda.device_count() == 1
    reference, prepared = configure(job, data_root)
    common = reference['common']
    path = ROOT / job['teacher']['path']
    marker = base.read(path.parent / 'ssl_complete.json')
    assert marker['complete'] and marker['epochs'] == 1000 and marker['identity'] == job['teacher']['identity']
    assert common.file_hash(path) == marker['checkpoint_file_sha256']
    encoder, _, saved = common.load_encoder(path, job['teacher']['identity'], 'cpu')
    assert saved['source'] == job['source'] and saved['model'] == 'ResNet18'
    assert saved['method'] == 'simclr' and saved['seed'] == 0 and saved['epochs'] == 1000
    assert saved['encoder_sha256'] == marker['encoder_sha256']
    assert saved['normalization'] == prepared['protocol']['normalization'][job['source']]
    del encoder, saved
    job['teacher'].update(file_sha256=common.file_hash(path), marker_sha256=common.file_hash(path.parent / 'ssl_complete.json'))
    if job['stage'] == 'generation':
        base.generation(job, data_root, common)
    else:
        assert job['stage'] == 'ssl_main'
        base.torch.backends.cuda.matmul.allow_tf32 = False
        base.torch.backends.cudnn.allow_tf32 = False
        base.ssl_and_main(job, reference)


if __name__ == '__main__':
    main()
