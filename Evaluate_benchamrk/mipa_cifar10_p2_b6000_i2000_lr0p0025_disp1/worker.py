"""One separate CIFAR10 distillation, one reference SimCLR encoder, six probes per label budget."""
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


def main():
    manifest = Path(sys.argv[1]).resolve()
    if not manifest.is_relative_to(HERE / 'batches'):
        raise ValueError('Unexpected experiment manifest')
    job = json.loads(manifest.read_text())
    if (not os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_JOB_ACCOUNT') != job['account']
            or job['account'] != 'aip-boyuwang'
            or os.environ.get('SLURM_CLUSTER_NAME', os.environ.get('CC_CLUSTER')) != 'vulcan'
            or not torch.cuda.is_available() or torch.cuda.device_count() != 1):
        raise RuntimeError('Use the configured single-GPU Wang job on Vulcan')
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
    if (args['dataset'], args['percentage'], args['model'], args['batch_real'],
            args['Iteration'], args['lr_img'], args['disp_weight'], args['pair_weight'], args['teacher_seed']) != (
                'CIFAR10', 2, 'ResNet18BN', 6000, 2000, 0.0025, 1, 0, [0]):
        raise ValueError('Not the requested CIFAR10 displacement experiment')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    data_root = Path(sys.argv[2]).resolve()
    gen = ROOT / job['synthetic_directory']
    folder = HERE / 'results/CIFAR10/ResNet18BN/pt2/simclr/ssl_seed_000'
    with common.exclusive_run(HERE / 'session'):
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

            def generation_progress(*values, **kwargs):
                if len(values) == 1 and isinstance(values[0], str):
                    hit = re.search(r'iter = (\d+)', values[0])
                    if hit and int(hit[1]) % 200:
                        return
                print(*values, **kwargs)

            method1.print = generation_progress
            previous = sys.argv
            sys.argv = argv
            try:
                method1.main()
            finally:
                sys.argv = previous
        path = gen / 'res_IIC_CIFAR10_ResNet18BN_2percent_final.pt'
        saved = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        if not saved['complete'] or saved['iteration'] != 2000 or saved['run_args'] != args:
            raise ValueError('The requested 2000-iteration distillation has not finished')
        images = saved['data'].detach().cpu().contiguous()
        if images.shape != (1000, 3, 32, 32) or not images.is_floating_point() or not torch.isfinite(images).all():
            raise ValueError('Invalid CIFAR10 2% synthetic tensor')
        normalization = saved['normalization']
        mean, std = (torch.tensor(v).view(1, 3, 1, 1) for v in normalization)
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
            raise ValueError('Invalid synthetic normalization')
        pixels = images * std + mean
        synthetic = reference_eval.SyntheticImages(pixels.clamp(0, 1).contiguous())
        train_set, test_set = data.RawDataset(data_root, 'CIFAR10', True), data.RawDataset(data_root, 'CIFAR10', False)
        protocol = json.loads((BENCH / 'protocol.json').read_text())
        protocol['ssl'].update(subset_epochs=2000, checkpoint_every_epochs=500)
        protocol['subset_seeds'] = [0]
        if protocol['evaluation']['epochs'] != 100:
            raise ValueError('Keep the existing 100-epoch linear-probe recipe')
        signature = dict(experiment=HERE.name, source='CIFAR10', model='ResNet18', subset=2,
            ssl='simclr', selection_method='ours', generation_args=args, teacher=job['teacher'],
            protocol=protocol, source_fingerprint=train_set.fingerprint,
            synthetic=dict(path=str(path), file_sha256=common.file_hash(path), iteration=2000,
                images_sha256=hashlib.sha256(images.numpy().tobytes()).hexdigest(), normalization=normalization,
                pixel_policy='denormalize_clamp_then_PIL_uint8',
                clipped_fraction=float(((pixels < 0) | (pixels > 1)).float().mean())),
            ssl_seeds=[0], probe_seeds=list(range(6)), code_hashes=common.code_hashes(),
            driver_code_hashes=job['code_hashes'])
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
            cells = []
            for labels in (1, 5):
                samples = []
                for seed in range(6):
                    output = folder / 'evaluations' / ('CIFAR10_lbl%d_probe%d.json' % (labels, seed))
                    result = reference['evaluate'].linear_probe(checkpoint, identity, protocol, 'CIFAR10', 'CIFAR10',
                        train_set, test_set, labels, seed, output, workers=0, device='cuda')
                    score = result['accuracy_percent']
                    if (not result['complete'] or not result['encoder_unchanged']
                            or result['encoder_sha256'] != trained['encoder_sha256']
                            or not math.isfinite(score) or not 0 <= score <= 100):
                        raise ValueError('Invalid frozen-probe result')
                    samples.append(dict(ssl_seed=0, probe_seed=seed, accuracy_percent=score, result_file=str(output)))
                scores = [s['accuracy_percent'] for s in samples]
                summary = dict(complete=True, experiment=HERE.name, source='CIFAR10', target='CIFAR10', model='ResNet18',
                    kind='ours', subset=2, ssl='simclr', ssl_epochs=2000, linear_epochs=100, labels=labels,
                    generation_args=args, completed_probes=6, expected_probes=6, std_ddof=1, samples=samples,
                    mean_percent=statistics.mean(scores), std_percent=statistics.stdev(scores))
                summary['tex_value'] = '\\dsacc{%.2f}{%.2f}' % (summary['mean_percent'], summary['std_percent'])
                common.atomic_json(folder / ('summary_lbl%d.json' % labels), summary)
                cells.append(summary)
                print('COMPLETE CELL:', labels, summary['tex_value'], flush=True)
            common.atomic_json(HERE / 'summary.json', dict(complete=True, cells=cells))


if __name__ == '__main__':
    main()
