"""CIFAR10 1% MIPA: vary only the real batch, then one 2k encoder and five probes."""
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

BATCHES = (128, 256, 512, 2048, 4096, 8192)


def main():
    manifest = Path(sys.argv[1]).resolve()
    if not manifest.is_relative_to(HERE / 'batches'):
        raise ValueError('Unexpected ablation manifest')
    job = json.loads(manifest.read_text())
    if (not os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_JOB_ACCOUNT') != job['account']
            or job['account'] not in ('aip-boyuwang', 'aip-yiweilu')
            or os.environ.get('SLURM_CLUSTER_NAME', os.environ.get('CC_CLUSTER')) != 'vulcan'
            or not torch.cuda.is_available() or torch.cuda.device_count() != 1):
        raise RuntimeError('Use the configured single-GPU Vulcan job')
    reference = reference_eval.reference_modules()
    common, data = reference['common'], reference['data']
    for relative, expected in job['code_hashes'].items():
        if common.file_hash(ROOT / relative) != expected:
            raise ValueError('Code changed after submission: ' + relative)
    teacher = ROOT / job['teacher']['path']
    if (common.file_hash(teacher) != job['teacher']['file_sha256']
            or common.file_hash(teacher.parent / 'ssl_complete.json') != job['teacher']['marker_sha256']):
        raise ValueError('Saved teacher changed after submission')
    args = job['generation_args']
    batch_size = args['batch_real']
    expected_args = dict(dataset='CIFAR10', percentage=1, model='ResNet18BN', seed=0,
        Iteration=8000, lr_img=0.001, batch_real=batch_size, temperature=0.2,
        disp_weight=0.0, pair_weight=0.0, num_random_networks=1, random_models='ResNet18BN',
        dsa=True, distill_aug_strategy='color_crop_cutout_flip_scale_rotate', distill_aug_mode='M',
        teacher_source='pretrained', teacher_seed=[0], teacher_ssl='simclr',
        teacher_path=str(BENCH / 'full_ssl_checkpoints_1000ep_3seeds'), feature_batch=128)
    if batch_size not in BATCHES or args != expected_args:
        raise ValueError('Not the requested CIFAR10 1% batch-size ablation')
    gen = OT / 'ours_syntehtic_batch_ablation/CIFAR10/ResNet18BN/pt1' / ('batch_%d' % batch_size)
    folder = HERE / 'results' / ('batch_%d' % batch_size) / 'simclr/ssl_seed_000'
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    data_root = Path(sys.argv[2]).resolve()
    with common.exclusive_run(HERE / ('sessions/batch_%d' % batch_size)):
        with common.exclusive_run(gen):
            contract = {key: job[key] for key in ('generation_args', 'teacher', 'code_hashes')}
            file = gen / 'pipeline_config.json'
            if file.exists() and json.loads(file.read_text()) != contract:
                raise ValueError('Distillation inputs changed; refusing to mix checkpoints')
            common.atomic_json(file, contract)
            argv = [str(OT / 'method1.py')]
            for key, value in args.items():
                if isinstance(value, bool):
                    argv.append('--' + (key if value else 'no-' + key))
                else:
                    argv += ['--' + key] + [str(v) for v in (value if isinstance(value, list) else [value])]
            argv += ['--device', 'cuda', '--data_path', str(data_root), '--save_path', str(gen),
                     '--resume', '--save_every', '100', '--num_eval', '0']

            def progress(*values, **kwargs):
                if len(values) == 1 and isinstance(values[0], str):
                    hit = re.search(r'iter = (\d+)', values[0])
                    if hit and int(hit[1]) % 200:
                        return
                print(*values, **kwargs)

            method1.print = progress
            previous = sys.argv
            sys.argv = argv
            try:
                method1.main()
            finally:
                sys.argv = previous
        path = gen / 'res_IIC_CIFAR10_ResNet18BN_1percent_final.pt'
        saved = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        if not saved['complete'] or saved['iteration'] != 8000 or saved['run_args'] != args:
            raise ValueError('The requested 8000-iteration distillation has not finished')
        images = saved['data'].detach().cpu().contiguous()
        if images.shape != (500, 3, 32, 32) or not images.is_floating_point() or not torch.isfinite(images).all():
            raise ValueError('Invalid CIFAR10 1% synthetic tensor')
        normalization = saved['normalization']
        mean, std = (torch.tensor(v).view(1, 3, 1, 1) for v in normalization)
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
            raise ValueError('Invalid saved synthetic normalization')
        pixels = images * std + mean
        synthetic = reference_eval.SyntheticImages(pixels.clamp(0, 1).contiguous())
        train_set = data.RawDataset(data_root, 'CIFAR10', True)
        test_set = data.RawDataset(data_root, 'CIFAR10', False)
        protocol = json.loads((BENCH / 'protocol.json').read_text())
        protocol['ssl'].update(subset_epochs=2000, checkpoint_every_epochs=500)
        protocol['subset_seeds'] = [0]
        if protocol['evaluation']['epochs'] != 100 or protocol['ssl']['batch_size'] != 256:
            raise ValueError('Keep the existing batch-256 SSL and 100-epoch linear recipe')
        signature = dict(experiment=HERE.name, source='CIFAR10', model='ResNet18', subset=1,
            ssl='simclr', selection_method='ours', generation_args=args, teacher=job['teacher'],
            protocol=protocol, source_fingerprint=train_set.fingerprint,
            synthetic=dict(path=str(path), file_sha256=common.file_hash(path), iteration=8000,
                initial_images_sha256=hashlib.sha256(saved['initial_data'].contiguous().numpy().tobytes()).hexdigest(),
                images_sha256=hashlib.sha256(images.numpy().tobytes()).hexdigest(), normalization=normalization,
                pixel_policy='denormalize_clamp_then_PIL_uint8',
                clipped_fraction=float(((pixels < 0) | (pixels > 1)).float().mean())),
            ssl_seeds=[0], probe_seeds=list(range(5)), labels=1,
            code_hashes=common.code_hashes(), driver_code_hashes=job['code_hashes'])
        identity = dict(signature=common.digest_json(signature), seed=0, selection_method='ours')
        config = dict(signature=signature, identity=identity)
        del saved, images, pixels
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        with common.exclusive_run(folder):
            file = folder / 'config.json'
            if file.exists() and json.loads(file.read_text()) != config:
                raise ValueError('Evaluation inputs changed; refusing to mix results')
            common.atomic_json(file, config)

            def ssl_progress(*values, **kwargs):
                if kwargs.get('file') is None and len(values) == 1 and isinstance(values[0], str) and values[0].startswith('SSL '):
                    prefix, line = values[0].split(': ', 1)
                    stats = json.loads(line)
                    if stats['epoch'] not in (1, 2000) and stats['epoch'] % 200:
                        return
                    stats.pop('projector_std', None)
                    stats.pop('seconds', None)
                    values = (prefix + ': ' + json.dumps(stats),)
                print(*values, **kwargs)

            reference['train_ssl'].print = ssl_progress
            checkpoint, trained = reference['train_ssl'].train(synthetic, protocol, 'CIFAR10', 'ResNet18',
                'simclr', 0, folder, identity, workers=0, device='cuda')
            if trained['epochs'] != 2000:
                raise ValueError('Not the completed 2000-epoch SimCLR encoder')
            common.atomic_json(folder / 'ssl_complete.json', dict(complete=True, identity=identity, epochs=2000,
                encoder_sha256=trained['encoder_sha256'], checkpoint_file_sha256=common.file_hash(checkpoint)))
            samples = []
            for seed in range(5):
                output = folder / 'evaluations' / ('CIFAR10_lbl1_probe%d.json' % seed)
                result = reference['evaluate'].linear_probe(checkpoint, identity, protocol, 'CIFAR10', 'CIFAR10',
                    train_set, test_set, 1, seed, output, workers=0, device='cuda')
                score = result['accuracy_percent']
                if (not result['complete'] or not result['encoder_unchanged']
                        or result['encoder_sha256'] != trained['encoder_sha256']
                        or not math.isfinite(score) or not 0 <= score <= 100):
                    raise ValueError('Invalid frozen-probe result')
                samples.append(dict(ssl_seed=0, probe_seed=seed, accuracy_percent=score, result_file=str(output)))
            scores = [s['accuracy_percent'] for s in samples]
            summary = dict(complete=True, experiment=HERE.name, source='CIFAR10', target='CIFAR10',
                model='ResNet18', kind='ours', subset=1, ssl='simclr', ssl_epochs=2000, linear_epochs=100,
                labels=1, generation_args=args, identity=identity, encoder_sha256=trained['encoder_sha256'],
                completed_probes=5, expected_probes=5, std_ddof=1, samples=samples,
                mean_percent=statistics.mean(scores), std_percent=statistics.stdev(scores))
            summary['tex_value'] = '\\dsacc{%.2f}{%.2f}' % (summary['mean_percent'], summary['std_percent'])
            common.atomic_json(folder / 'summary_lbl1.json', summary)
            print('COMPLETE batch %d:' % batch_size, summary['tex_value'], flush=True)


if __name__ == '__main__':
    main()
