"""MIPA initialization ablation; reuse the existing distillation and evaluation loops."""
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import sys

import torch

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent
ROOT = BENCH.parent
OT = ROOT / 'OT-SSL-DD'
sys.path.insert(0, str(OT))
import eval as reference_eval
import method1

VARIANTS = {**{'random_seed_%03d' % seed: dict(method='random', seed=seed, kmeans_iterations=0)
              for seed in range(3)},
            'kmeans100_seed_000': dict(method='kmeans', seed=0, kmeans_iterations=100)}


def generation_args():
    return dict(dataset='CIFAR10', percentage=0.2, model='ResNet18BN', seed=0,
                Iteration=10000, lr_img=0.001, batch_real=1024, temperature=0.2,
                disp_weight=0.0, pair_weight=0.0, num_random_networks=1,
                random_models='ResNet18BN', dsa=True,
                distill_aug_strategy='color_crop_cutout_flip_scale_rotate', distill_aug_mode='M',
                teacher_source='pretrained', teacher_seed=[0], teacher_ssl='simclr',
                teacher_path=str(BENCH / 'full_ssl_checkpoints_1000ep_3seeds'), feature_batch=128)


def synthetic_path(variant):
    return OT / 'ours_syntehtic_init_ablation/CIFAR10/ResNet18BN/pt0.2' / variant / 'res_IIC_CIFAR10_ResNet18BN_0.2percent_final.pt'


def contract(job, variant):
    return dict(generation_args=generation_args(), initialization=VARIANTS[variant],
                teacher=job['teacher'], code_hashes=job['code_hashes'])


@torch.no_grad()
def initialize(spec, output, common, dst_train, num_syn, device, seed, channel, num_classes,
               im_size, model_name='ConvNet', batch_size=128):
    assert num_syn == 100 and seed == 0 and model_name == 'ResNet18BN'
    rng = torch.Generator().manual_seed(spec['seed'])
    iterations_run = 0
    if spec['method'] == 'random':
        indices = torch.randperm(len(dst_train), generator=rng)[:num_syn]
        assert len(indices.unique()) == num_syn
    else:
        network = method1.get_network(model_name, channel, num_classes, im_size, seed=seed).to(device)
        encoder = method1.frozen_embedder(network)
        loader = torch.utils.data.DataLoader(dst_train, batch_size=batch_size, shuffle=False, num_workers=0)
        features = torch.cat([encoder(method1._images_from_batch(batch).to(device)).cpu() for batch in loader])
        centers = features[torch.randperm(len(features), generator=rng)[:num_syn]].clone()
        # Same Lloyd updates, convergence check and nearest-image rule as method1, with a 100-iteration budget.
        for iteration in range(spec['kmeans_iterations']):
            assigns = torch.cat([torch.cdist(block, centers).argmin(dim=1) for block in features.split(1024)])
            counts = torch.bincount(assigns, minlength=num_syn)
            new_centers = torch.zeros_like(centers).index_add_(0, assigns, features)
            new_centers /= counts.clamp_min(1)[:, None]
            new_centers[counts == 0] = centers[counts == 0]
            iterations_run = iteration + 1
            if torch.allclose(centers, new_centers, atol=1e-4):
                break
            centers = new_centers
        indices = torch.cat([torch.cdist(block, features).argmin(dim=1) for block in centers.split(32)])
        del network, encoder, features, centers
    images = torch.stack([method1._images_from_batch(dst_train[int(index)]) for index in indices])
    common.atomic_json(output / 'initialization.json', dict(**spec, iterations_run=iterations_run,
        indices=indices.tolist(), unique_images=len(indices.unique()),
        initial_images_sha256=hashlib.sha256(images.contiguous().numpy().tobytes()).hexdigest()))
    print('INITIALIZATION:', spec, 'unique images:', len(indices.unique()), 'K-means updates:', iterations_run, flush=True)
    return images.to(device)


def generate(job, variant, data_root, common):
    path = synthetic_path(variant)
    expected = contract(job, variant)
    with common.exclusive_run(path.parent):
        config = path.parent / 'pipeline_config.json'
        if config.exists() and json.loads(config.read_text()) != expected:
            raise ValueError('Generation inputs changed: ' + variant)
        common.atomic_json(config, expected)
        if path.exists():
            saved = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
            if saved.get('initialization') != VARIANTS[variant] or saved['run_args'] != generation_args():
                raise ValueError('Not this initialization experiment')
            del saved
        original_initializer = method1.deepcluster_initialize
        original_save = common.atomic_save
        original_print = method1.__dict__.get('print', print)
        argv = [str(OT / 'method1.py')]
        for key, value in generation_args().items():
            if isinstance(value, bool):
                argv.append('--' + (key if value else 'no-' + key))
            else:
                argv += ['--' + key] + [str(v) for v in (value if isinstance(value, list) else [value])]
        argv += ['--device', 'cuda', '--data_path', str(data_root), '--save_path', str(path.parent),
                 '--resume', '--save_every', '100', '--num_eval', '0']

        def save(destination, value):
            if Path(destination).resolve() == path:
                assert value['run_args'] == generation_args()
                value = dict(value, initialization=VARIANTS[variant])
            original_save(destination, value)
            if Path(destination).resolve() == path:
                common.atomic_json(path.parent / 'generation_state.json', dict(
                    iteration=value['iteration'], complete=value['complete'], contract=expected))

        def progress(*values, **kwargs):
            if len(values) == 1 and isinstance(values[0], str):
                hit = re.search(r'iter = (\d+)', values[0])
                if hit and int(hit[1]) % 200:
                    return
            print(*values, **kwargs)

        method1.deepcluster_initialize = lambda *a, **kw: initialize(VARIANTS[variant], path.parent, common, *a, **kw)
        common.atomic_save = save
        method1.print = progress
        previous = sys.argv
        sys.argv = argv
        try:
            method1.main()
        finally:
            sys.argv = previous
            method1.deepcluster_initialize = original_initializer
            common.atomic_save = original_save
            method1.print = original_print
        saved = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        assert saved['complete'] and saved['iteration'] == 10000 and saved['initialization'] == VARIANTS[variant]


def evaluate(job, variant, data_root, reference):
    common, data = reference['common'], reference['data']
    path = synthetic_path(variant)
    assert json.loads((path.parent / 'pipeline_config.json').read_text()) == contract(job, variant)
    saved = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    assert saved['complete'] and saved['iteration'] == 10000
    assert saved['run_args'] == generation_args() and saved['initialization'] == VARIANTS[variant]
    images = saved['data'].detach().cpu().contiguous()
    assert images.shape == (100, 3, 32, 32) and torch.isfinite(images).all()
    normalization = saved['normalization']
    assert normalization == reference_eval.SYNTHETIC_NORMALIZATION['CIFAR10']
    mean, std = (torch.tensor(v).view(1, 3, 1, 1) for v in normalization)
    pixels = images * std + mean
    synthetic = reference_eval.SyntheticImages(pixels.clamp(0, 1).contiguous())
    train_set, test_set = data.RawDataset(data_root, 'CIFAR10', True), data.RawDataset(data_root, 'CIFAR10', False)
    protocol = json.loads((BENCH / 'protocol.json').read_text())
    protocol['ssl'].update(subset_epochs=2000, checkpoint_every_epochs=500)
    protocol['subset_seeds'] = [0]
    assert protocol['evaluation']['epochs'] == 100 and protocol['ssl']['batch_size'] == 256
    signature = dict(experiment=HERE.name, source='CIFAR10', model='ResNet18', subset=0.2,
        ssl='simclr', selection_method='ours', initialization=VARIANTS[variant], generation_args=generation_args(),
        teacher=job['teacher'], protocol=protocol, source_fingerprint=train_set.fingerprint,
        synthetic=dict(path=str(path), file_sha256=common.file_hash(path), iteration=10000,
            images_sha256=hashlib.sha256(images.numpy().tobytes()).hexdigest(), normalization=normalization,
            initialization=read_initialization(path), pixel_policy='denormalize_clamp_then_PIL_uint8',
            clipped_fraction=float(((pixels < 0) | (pixels > 1)).float().mean())),
        ssl_seeds=[0], probe_seeds=list(range(5)), labels=1, code_hashes=common.code_hashes(),
        driver_code_hashes=job['code_hashes'])
    identity = dict(signature=common.digest_json(signature), seed=0, selection_method='ours')
    config = dict(signature=signature, identity=identity)
    folder = HERE / 'results' / variant / 'simclr/ssl_seed_000'
    del saved, images, pixels
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    with common.exclusive_run(folder):
        file = folder / 'config.json'
        if file.exists() and json.loads(file.read_text()) != config:
            raise ValueError('Evaluation inputs changed: ' + variant)
        common.atomic_json(file, config)

        def progress(*values, **kwargs):
            if kwargs.get('file') is None and len(values) == 1 and isinstance(values[0], str) and values[0].startswith('SSL '):
                prefix, line = values[0].split(': ', 1)
                stats = json.loads(line)
                if stats['epoch'] not in (1, 2000) and stats['epoch'] % 200:
                    return
                stats.pop('projector_std', None)
                stats.pop('seconds', None)
                values = (prefix + ': ' + json.dumps(stats),)
            print(*values, **kwargs)

        reference['train_ssl'].print = progress
        checkpoint, trained = reference['train_ssl'].train(synthetic, protocol, 'CIFAR10', 'ResNet18',
            'simclr', 0, folder, identity, workers=0, device='cuda')
        assert trained['epochs'] == 2000
        common.atomic_json(folder / 'ssl_complete.json', dict(complete=True, identity=identity, epochs=2000,
            encoder_sha256=trained['encoder_sha256'], checkpoint_file_sha256=common.file_hash(checkpoint)))
        samples = []
        for seed in range(5):
            output = folder / 'evaluations' / ('CIFAR10_lbl1_probe%d.json' % seed)
            result = reference['evaluate'].linear_probe(checkpoint, identity, protocol, 'CIFAR10', 'CIFAR10',
                train_set, test_set, 1, seed, output, workers=0, device='cuda')
            score = result['accuracy_percent']
            assert result['complete'] and result['encoder_unchanged'] and result['encoder_sha256'] == trained['encoder_sha256']
            assert math.isfinite(score) and 0 <= score <= 100
            samples.append(dict(ssl_seed=0, probe_seed=seed, accuracy_percent=score, result_file=str(output)))
        scores = [sample['accuracy_percent'] for sample in samples]
        summary = dict(complete=True, experiment=HERE.name, variant=variant, initialization=VARIANTS[variant],
            source='CIFAR10', target='CIFAR10', model='ResNet18', subset=0.2, ssl='simclr', labels=1,
            generation_args=generation_args(), ssl_epochs=2000, linear_epochs=100, identity=identity,
            completed_probes=5, expected_probes=5, std_ddof=1, samples=samples,
            mean_percent=statistics.mean(scores), std_percent=statistics.stdev(scores))
        common.atomic_json(folder / 'summary_lbl1.json', summary)
        print('COMPLETE:', variant, '%.2f +/- %.2f' % (summary['mean_percent'], summary['std_percent']), flush=True)


def read_initialization(path):
    return json.loads((path.parent / 'initialization.json').read_text())


def main():
    manifest = Path(sys.argv[1]).resolve()
    if not manifest.is_relative_to(HERE / 'batches'):
        raise ValueError('Unexpected initialization-ablation manifest')
    job = json.loads(manifest.read_text())
    if (not os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_JOB_ACCOUNT') != job['account']
            or job['account'] not in ('aip-boyuwang', 'aip-yiweilu')
            or os.environ.get('SLURM_CLUSTER_NAME', os.environ.get('CC_CLUSTER')) != 'vulcan'
            or not torch.cuda.is_available() or torch.cuda.device_count() != 1):
        raise RuntimeError('Use the configured one-GPU Vulcan job')
    assert job['stage'] in ('generation', 'evaluation') and job['variants']
    assert all(variant in VARIANTS for variant in job['variants'])
    assert job['generation_args'] == generation_args()
    reference = reference_eval.reference_modules()
    common = reference['common']
    for relative, expected in job['code_hashes'].items():
        if common.file_hash(ROOT / relative) != expected:
            raise ValueError('Code changed after submission: ' + relative)
    teacher = ROOT / job['teacher']['path']
    assert common.file_hash(teacher) == job['teacher']['file_sha256']
    assert common.file_hash(teacher.parent / 'ssl_complete.json') == job['teacher']['marker_sha256']
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    data_root = Path(sys.argv[2]).resolve()
    for variant in job['variants']:
        if job['stage'] == 'generation':
            generate(job, variant, data_root, common)
        else:
            evaluate(job, variant, data_root, reference)


if __name__ == '__main__':
    main()
