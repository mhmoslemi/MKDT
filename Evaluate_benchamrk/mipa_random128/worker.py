"""Exact Random-baseline initialization, batch-128 MIPA, then standard SimCLR/probes."""
import fcntl
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
SYNTHETICS = OT / 'ours_syntehtic_random128'
SOURCES = ('CIFAR10', 'CIFAR100', 'TinyImageNet')
ITERATIONS = {0.2: 10000, 1: 8000, 2: 6000}
TARGETS = {source: tuple(t for t in (*SOURCES, 'Aircraft', 'Dogs') if t != source) for source in SOURCES}
sys.path.insert(0, str(OT))
import eval as reference_eval
import method1


def read(path):
    return json.loads(Path(path).read_text())


def generation_args(source, subset):
    assert source in SOURCES and subset in ITERATIONS
    return dict(dataset=source, percentage=subset, model='ResNet18BN', seed=0,
        Iteration=ITERATIONS[subset], lr_img=0.001, batch_real=128, temperature=0.2,
        disp_weight=0.0, pair_weight=0.0, num_random_networks=1, random_models='ResNet18BN',
        dsa=True, distill_aug_strategy='color_crop_cutout_flip_scale_rotate', distill_aug_mode='M',
        teacher_source='pretrained', teacher_seed=[1 if source == 'TinyImageNet' else 0],
        teacher_ssl='simclr', teacher_path=str(BENCH / 'full_ssl_checkpoints_1000ep_3seeds'), feature_batch=128)


def synthetic_path(source, subset):
    return SYNTHETICS / source / 'ResNet18BN' / ('pt%g' % subset) / (
        'res_IIC_%s_ResNet18BN_%gpercent_final.pt' % (source, subset))


def encoder_folder(source, subset):
    return HERE / 'results' / source / 'ResNet18BN' / ('pt%g' % subset) / 'simclr/ssl_seed_000'


def contract(job):
    return {key: job[key] for key in ('generation_args', 'selection', 'teacher', 'code_hashes')}


def initial_images(job, dataset, count, device):
    selection = job['selection']
    assert dataset.base.fingerprint == selection['source_fingerprint']
    indices = selection['indices']
    assert len(indices) == len(set(indices)) == count
    assert all(isinstance(i, int) and 0 <= i < len(dataset) for i in indices)
    return torch.stack([method1._images_from_batch(dataset[index]) for index in indices]).to(device)


def generation(job, data_root, common):
    source, subset = job['source'], job['subset']
    args = generation_args(source, subset)
    path = synthetic_path(source, subset)
    with common.exclusive_run(path.parent):
        config = path.parent / 'pipeline_config.json'
        if config.exists() and read(config) != contract(job):
            raise ValueError('Generation inputs changed; use a new output directory')
        common.atomic_json(config, contract(job))
        if path.exists():
            saved = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
            assert saved['run_args'] == args and saved['initialization'] == job['selection']
            del saved
        original_initializer = method1.deepcluster_initialize
        original_save = common.atomic_save
        original_print = method1.__dict__.get('print', print)
        previous_argv = sys.argv

        def initialize(dataset, count, device, *unused, **kwargs):
            images = initial_images(job, dataset, count, device)
            common.atomic_json(path.parent / 'initialization.json', dict(job['selection'],
                initial_images_sha256=hashlib.sha256(images.cpu().contiguous().numpy().tobytes()).hexdigest()))
            print('INITIALIZATION: exact saved Random seed-0 indices, %d images' % count, flush=True)
            return images

        def save(destination, value):
            if Path(destination).resolve() == path:
                assert value['run_args'] == args
                value = dict(value, initialization=job['selection'])
            original_save(destination, value)
            if Path(destination).resolve() == path:
                common.atomic_json(path.parent / 'generation_state.json', dict(
                    iteration=value['iteration'], complete=value['complete'], contract=contract(job)))

        def progress(*values, **kwargs):
            if len(values) == 1 and isinstance(values[0], str):
                hit = re.search(r'iter = (\d+)', values[0])
                if hit and int(hit[1]) % 200:
                    return
            print(*values, **kwargs)

        argv = [str(OT / 'method1.py')]
        for key, value in args.items():
            if isinstance(value, bool):
                argv.append('--' + (key if value else 'no-' + key))
            else:
                argv += ['--' + key] + [str(v) for v in (value if isinstance(value, list) else [value])]
        sys.argv = argv + ['--device', 'cuda', '--data_path', str(data_root), '--save_path', str(path.parent),
                          '--resume', '--save_every', '100', '--num_eval', '0']
        method1.deepcluster_initialize, common.atomic_save, method1.print = initialize, save, progress
        try:
            method1.main()
        finally:
            sys.argv = previous_argv
            method1.deepcluster_initialize, common.atomic_save, method1.print = original_initializer, original_save, original_print
        saved = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        assert saved['complete'] and saved['iteration'] == ITERATIONS[subset]
        assert saved['initialization'] == job['selection']


def summarize(job, reference, checkpoint, config, marker, target, labels):
    common, data = reference['common'], reference['data']
    source, subset = job['source'], job['subset']
    folder = encoder_folder(source, subset)
    protocol = config['signature']['protocol']
    train_set, test_set = data.RawDataset(job['data_root'], target, True), data.RawDataset(job['data_root'], target, False)
    samples = []
    for seed in range(5):
        output = folder / 'evaluations' / ('%s_lbl%d_probe%d.json' % (target, labels, seed))
        result = reference['evaluate'].linear_probe(checkpoint, config['identity'], protocol, source, target,
            train_set, test_set, labels, seed, output, workers=0, device='cuda')
        score = result['accuracy_percent']
        assert result['complete'] and result['encoder_unchanged'] and result['encoder_sha256'] == marker['encoder_sha256']
        assert math.isfinite(score) and 0 <= score <= 100
        samples.append(dict(ssl_seed=0, probe_seed=seed, accuracy_percent=score,
                            encoder_sha256=marker['encoder_sha256'], result_file=str(output)))
    scores = [row['accuracy_percent'] for row in samples]
    summary = dict(complete=True, experiment=HERE.name, source=source, target=target, subset=subset,
        model='ResNet18', kind='ours', ssl='simclr', ssl_epochs=2000, linear_epochs=100,
        labels=labels, ssl_seeds=[0], probe_seeds=list(range(5)), completed_probes=5, expected_probes=5,
        std_ddof=1, samples=samples, mean_percent=statistics.mean(scores), std_percent=statistics.stdev(scores),
        identity=config['identity'], generation_args=job['generation_args'], selection=job['selection'],
        checkpoint=str(checkpoint), table_scope='CVPR_tables/' + ('table.tex' if source == target else 'cifar_downstream.tex'))
    precision = 2 if source == target else 1
    summary['tex_value'] = '\\dsacc{%.*f}{%.*f}' % (precision, summary['mean_percent'], precision, summary['std_percent'])
    common.atomic_json(folder / ('summary_%s_lbl%d.json' % (target, labels)), summary)
    print('COMPLETE CELL:', source, subset, target, labels, summary['tex_value'], flush=True)


def ssl_and_main(job, reference):
    common, data = reference['common'], reference['data']
    source, subset = job['source'], job['subset']
    path = synthetic_path(source, subset)
    assert read(path.parent / 'pipeline_config.json') == contract(job)
    saved = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    assert saved['complete'] and saved['iteration'] == ITERATIONS[subset]
    assert saved['run_args'] == generation_args(source, subset) and saved['initialization'] == job['selection']
    images = saved['data'].detach().cpu().contiguous()
    size = 64 if source == 'TinyImageNet' else 32
    count = int(data.SOURCE_COUNTS[source] * subset / 100)
    assert images.shape == (count, 3, size, size) and torch.isfinite(images).all()
    assert saved['normalization'] == reference_eval.SYNTHETIC_NORMALIZATION[source]
    mean, std = (torch.tensor(v).view(1, 3, 1, 1) for v in saved['normalization'])
    pixels = images * std + mean
    synthetic = reference_eval.SyntheticImages(pixels.clamp(0, 1).contiguous())
    train_set = data.RawDataset(job['data_root'], source, True)
    assert train_set.fingerprint == job['selection']['source_fingerprint']
    protocol = read(BENCH / 'protocol.json')
    protocol['ssl'].update(subset_epochs=2000, checkpoint_every_epochs=500)
    protocol['subset_seeds'] = [0]
    assert protocol['ssl']['batch_size'] == 256 and protocol['evaluation']['epochs'] == 100
    signature = dict(experiment=HERE.name, source=source, model='ResNet18', subset=subset,
        selection_method='ours', ssl='simclr', protocol=protocol, source_fingerprint=train_set.fingerprint,
        generation_args=job['generation_args'], selection=job['selection'], teacher=job['teacher'],
        synthetic=dict(path=str(path), file_sha256=common.file_hash(path), iteration=saved['iteration'],
            normalization=saved['normalization'], images_sha256=hashlib.sha256(images.numpy().tobytes()).hexdigest(),
            initial_images_sha256=read(path.parent / 'initialization.json')['initial_images_sha256'],
            pixel_policy='denormalize_clamp_then_PIL_uint8', clipped_fraction=float(((pixels < 0) | (pixels > 1)).float().mean())),
        ssl_seeds=[0], probe_seeds=list(range(5)), code_hashes=common.code_hashes(), driver_code_hashes=job['code_hashes'])
    identity = dict(signature=common.digest_json(signature), seed=0, selection_method='ours')
    config = dict(signature=signature, identity=identity)
    folder = encoder_folder(source, subset)
    del saved, images, pixels, train_set
    with common.exclusive_run(folder):
        cfg = folder / 'config.json'
        if cfg.exists() and read(cfg) != config:
            raise ValueError('SSL inputs changed; refusing to mix results')
        common.atomic_json(cfg, config)

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
        checkpoint, trained = reference['train_ssl'].train(synthetic, protocol, source, 'ResNet18', 'simclr',
            0, folder, identity, workers=0, device='cuda')
        assert trained['epochs'] == 2000
        marker = dict(complete=True, identity=identity, epochs=2000, encoder_sha256=trained['encoder_sha256'],
                      checkpoint_file_sha256=common.file_hash(checkpoint))
        common.atomic_json(folder / 'ssl_complete.json', marker)
        del synthetic
        for labels in (1, 5):
            summarize(job, reference, checkpoint, config, marker, source, labels)
        common.atomic_json(folder / 'complete.json', dict(complete=True, identity=identity,
            generation_contract=contract(job), ssl_epochs=2000, probe_seeds=list(range(5)), labels=[1, 5]))


def downstream(job, reference):
    common = reference['common']
    source, subset, target, labels = (job[k] for k in ('source', 'subset', 'target', 'labels'))
    assert subset in (0.2, 1) and target in TARGETS[source] and labels in (1, 5)
    folder = encoder_folder(source, subset)
    with (folder / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        config, marker = read(folder / 'config.json'), read(folder / 'ssl_complete.json')
        signature = config['signature']
        assert common.digest_json(signature) == config['identity']['signature']
        assert signature['generation_args'] == job['generation_args'] and signature['selection'] == job['selection']
        assert signature['teacher'] == job['teacher'] and signature['driver_code_hashes'] == job['code_hashes']
        assert signature['source'] == source and signature['subset'] == subset and signature['ssl'] == 'simclr'
        assert signature['code_hashes'] == common.code_hashes()
        checkpoint = folder / 'encoder.pt'
        assert marker['complete'] and marker['epochs'] == 2000 and marker['identity'] == config['identity']
        assert marker['checkpoint_file_sha256'] == common.file_hash(checkpoint)
        encoder, _, trained = common.load_encoder(checkpoint, config['identity'], 'cpu')
        assert trained['epochs'] == 2000 and trained['encoder_sha256'] == marker['encoder_sha256']
        del encoder, trained
        with common.exclusive_run(folder / 'probe_locks' / ('%s_lbl%d' % (target, labels))):
            summarize(job, reference, checkpoint, config, marker, target, labels)


def main():
    manifest, data_root = (Path(p).resolve() for p in sys.argv[1:])
    assert manifest.is_relative_to(HERE / 'batches')
    job = read(manifest)
    assert job['source'] in SOURCES and job['subset'] in ITERATIONS
    assert job['stage'] in ('generation', 'ssl_main', 'downstream')
    expected_account = 'aip-yiweilu' if job['stage'] == 'downstream' else 'aip-boyuwang'
    if (not os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_JOB_ACCOUNT') != expected_account
            or job['account'] != expected_account
            or os.environ.get('SLURM_CLUSTER_NAME', os.environ.get('CC_CLUSTER')) != 'vulcan'
            or not torch.cuda.is_available() or torch.cuda.device_count() != 1):
        raise RuntimeError('Use the configured one-GPU Vulcan launcher/account')
    reference = reference_eval.reference_modules()
    common = reference['common']
    for relative, expected in job['code_hashes'].items():
        if common.file_hash(ROOT / relative) != expected:
            raise ValueError('Code changed since submission: ' + relative)
    assert job['generation_args'] == generation_args(job['source'], job['subset'])
    origin = ROOT / job['selection']['config']
    assert origin.is_relative_to(BENCH) and common.file_hash(origin) == job['selection']['config_sha256']
    previous = read(origin)
    assert previous['identity']['seed'] == 0 and previous['identity']['selection_method'] == 'random'
    assert previous['identity']['selection']['indices'] == job['selection']['indices']
    assert previous['signature']['source_fingerprint'] == job['selection']['source_fingerprint']
    teacher = ROOT / job['teacher']['path']
    assert common.file_hash(teacher) == job['teacher']['file_sha256']
    assert common.file_hash(teacher.parent / 'ssl_complete.json') == job['teacher']['marker_sha256']
    assert read(teacher.parent / 'ssl_complete.json')['complete']
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if job['stage'] != 'generation':
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    job['data_root'] = str(data_root)
    if job['stage'] == 'generation':
        generation(job, data_root, common)
    elif job['stage'] == 'ssl_main':
        ssl_and_main(job, reference)
    else:
        downstream(job, reference)


if __name__ == '__main__':
    main()
