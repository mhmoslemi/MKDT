"""Continue MIPA in checkpointed segments, then one 2k SSL encoder and six probes."""
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
SYNTHETICS = OT / 'ours_syntehtic_b6000_i5000_lr0p0015'
SETTINGS = (('CIFAR10', 2), ('CIFAR100', 2), ('TinyImageNet', 0.2),
            ('TinyImageNet', 1), ('TinyImageNet', 2))


def read(path):
    return json.loads(Path(path).read_text())


def generation_args(source, subset):
    return dict(dataset=source, percentage=subset, model='ResNet18BN', seed=0,
                Iteration=5000, lr_img=0.0015, batch_real=6000, temperature=0.2,
                disp_weight=0.0, pair_weight=0.0, num_random_networks=1,
                random_models='ResNet18BN', dsa=True,
                distill_aug_strategy='color_crop_cutout_flip_scale_rotate', distill_aug_mode='M',
                teacher_source='pretrained', teacher_seed=[1 if source == 'TinyImageNet' else 0],
                teacher_ssl='simclr', teacher_path=str(BENCH / 'full_ssl_checkpoints_1000ep_3seeds'),
                feature_batch=128)


def synthetic_path(source, subset):
    return SYNTHETICS / source / 'ResNet18BN' / ('pt%g' % subset) / (
        'res_IIC_%s_ResNet18BN_%gpercent_final.pt' % (source, subset))


def generation(job, data_root):
    source, subset, stop = job['source'], job['subset'], job['stop_iteration']
    if stop not in (1000, 2000, 3000, 4000, 5000):
        raise ValueError('Invalid generation segment')
    args = generation_args(source, subset)
    path = synthetic_path(source, subset)
    path.parent.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(OT))
    import eval as generation_eval
    import method1
    common = generation_eval.reference_modules()['common']
    with (path.parent / '.pipeline.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        contract = dict(generation_args=args, code_hashes=job['code_hashes'], teacher=job['teacher'])
        config_file = path.parent / 'pipeline_config.json'
        if config_file.exists() and read(config_file) != contract:
            raise ValueError('Generation inputs/code changed; refusing to mix runs')
        common.atomic_json(config_file, contract)
        if path.exists():
            saved = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
            if saved.get('run_args') != args or not 0 <= saved['iteration'] <= 5000:
                raise ValueError('Not this experiment\'s resumable checkpoint')
            current = saved['iteration']
            del saved
            if current >= stop:
                print('SKIP generation segment already saved:', current, '>=', stop, flush=True)
                return
        elif stop != 1000:
            raise ValueError('Previous segment checkpoint is missing; will not restart')

        class SegmentFinished(Exception):
            pass

        original_save = common.atomic_save

        def save_segment(destination, value):
            original_save(destination, value)
            if Path(destination).resolve() != path:
                return
            if value['run_args'] != args:
                raise ValueError('Generation parameters differ from the requested experiment')
            iteration = value['iteration']
            common.atomic_json(path.parent / 'generation_state.json', dict(
                iteration=iteration, complete=value['complete'], generation_args=args,
                checkpoint=str(path), code_hashes=job['code_hashes']))
            if iteration >= stop and stop < 5000:
                print('SEGMENT SAVED: iteration %d/5000; Adam and RNG retained' % iteration, flush=True)
                raise SegmentFinished()

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
        argv += ['--device', 'cuda', '--data_path', str(data_root), '--save_path', str(path.parent),
                 '--resume', '--save_every', '100', '--num_eval', '0']
        common.atomic_save = save_segment
        method1.print = progress
        sys.argv = argv
        try:
            method1.main()
        except SegmentFinished:
            pass
        finally:
            common.atomic_save = original_save
        final = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        if final['iteration'] != stop or final['run_args'] != args or final['complete'] != (stop == 5000):
            raise ValueError('Generation segment did not save its requested boundary')
        if stop == 5000:
            for iteration in (1, 5000):
                if not (path.parent / ('vis_IIC_%s_ResNet18BN_%gpercent_iter%d.png' %
                                       (source, subset, iteration))).is_file():
                    raise ValueError('Final generation must retain first/final PNGs')


def evaluation(job, data_root):
    sys.path.insert(0, str(BENCH))
    from common import atomic_json, code_hashes, digest_json, exclusive_run, file_hash, load_encoder
    from data import ImageView, RawDataset, SOURCE_COUNTS
    from evaluate import linear_probe
    import train_ssl
    from torchvision.transforms.functional import to_pil_image

    source, subset = job['source'], job['subset']
    path = synthetic_path(source, subset)
    accepted = job.get('accepted_generation')
    if accepted and (file_hash(path) != accepted['file_sha256']
                     or read(path.parent / 'generation_state.json')['iteration'] != accepted['iteration']):
        raise ValueError('Stopped generation checkpoint changed after submission')
    saved = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    if saved['run_args'] != generation_args(source, subset):
        raise ValueError('Synthetic generation parameters differ from this experiment')
    if accepted:
        if not 0 < accepted['iteration'] < 5000 or saved['iteration'] != accepted['iteration'] or saved['complete']:
            raise ValueError('Not the explicitly accepted early-stopped checkpoint')
    elif not saved['complete'] or saved['iteration'] != 5000:
        raise ValueError('Generation must finish 5k iterations, or explicitly use --eval-stopped')
    images = saved['data'].detach().cpu().contiguous()
    size = 64 if source == 'TinyImageNet' else 32
    count = int(SOURCE_COUNTS[source] * subset / 100)
    if images.shape != (count, 3, size, size) or not images.is_floating_point() or not torch.isfinite(images).all():
        raise ValueError('Invalid synthetic image tensor')
    normalization = saved['normalization']
    mean, std = (torch.tensor(v).view(1, 3, 1, 1) for v in normalization)
    if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
        raise ValueError('Invalid saved synthetic normalization')
    pixels = images * std + mean
    synthetic = ImageView(torch.utils.data.TensorDataset(pixels.clamp(0, 1).contiguous(),
                          torch.full((count,), -1)), to_pil_image)
    train_set = RawDataset(data_root, source, True)
    protocol = read(BENCH / 'protocol.json')
    protocol['ssl'].update(subset_epochs=2000, checkpoint_every_epochs=500)
    protocol['subset_seeds'] = [0]
    if protocol['evaluation']['epochs'] != 100:
        raise ValueError('Frozen probes must use the existing 100-epoch reference recipe')
    experiment = HERE.name + ('_stopped' if accepted else '')
    signature = dict(experiment=experiment, source=source, model='ResNet18', subset=subset,
        ssl='simclr', selection_method='ours', protocol=protocol, source_fingerprint=train_set.fingerprint,
        synthetic=dict(path=str(path), file_sha256=file_hash(path), iteration=saved['iteration'],
            images_sha256=hashlib.sha256(images.numpy().tobytes()).hexdigest(), normalization=normalization,
            pixel_policy='denormalize_clamp_then_PIL_uint8',
            clipped_fraction=float(((pixels < 0) | (pixels > 1)).float().mean())),
        ssl_seeds=[0], probe_seeds=list(range(6)), code_hashes=code_hashes(), driver_code_hashes=job['code_hashes'])
    if accepted:
        signature['accepted_generation'] = accepted
    identity = dict(signature=digest_json(signature), seed=0, selection_method='ours')
    config = dict(signature=signature, identity=identity)
    folder = HERE / ('stopped_results' if accepted else 'results') / source / 'ResNet18BN' / ('pt%g' % subset) / 'simclr/ssl_seed_000'
    del saved, images, pixels
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if job['stage'] == 'ssl':
        with exclusive_run(folder):
            cfg = folder / 'config.json'
            if cfg.exists() and read(cfg) != config:
                raise ValueError('SSL configuration changed; refusing to mix results')
            atomic_json(cfg, config)

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

            train_ssl.print = progress
            checkpoint, trained = train_ssl.train(synthetic, protocol, source, 'ResNet18', 'simclr', 0,
                                                 folder, identity, workers=0, device='cuda')
            if trained['epochs'] != 2000:
                raise ValueError('SSL did not finish exactly 2000 epochs')
            atomic_json(folder / 'ssl_complete.json', dict(complete=True, identity=identity, epochs=2000,
                encoder_sha256=trained['encoder_sha256'], checkpoint_file_sha256=file_hash(checkpoint)))
        return
    labels = job['labels']
    if labels not in (1, 5):
        raise ValueError('Only main-table 1%/5% labeled probes are requested')
    with (folder / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        if read(folder / 'config.json') != config:
            raise ValueError('Probe does not match its saved SSL encoder')
        checkpoint = folder / 'encoder.pt'
        marker = read(folder / 'ssl_complete.json')
        if (not marker['complete'] or marker['identity'] != identity or marker['epochs'] != 2000
                or file_hash(checkpoint) != marker['checkpoint_file_sha256']):
            raise ValueError('Missing or changed completed 2k encoder')
        encoder, _, trained = load_encoder(checkpoint, identity, device='cpu')
        if trained['epochs'] != 2000 or trained['encoder_sha256'] != marker['encoder_sha256']:
            raise ValueError('Wrong frozen encoder')
        del encoder, trained
        test_set = RawDataset(data_root, source, False)
        with exclusive_run(folder / ('probe_locks/lbl%d' % labels)):
            samples = []
            for seed in range(6):
                output = folder / 'evaluations' / ('%s_lbl%d_probe%d.json' % (source, labels, seed))
                print('SKIP saved probe:' if output.exists() else 'RUN probe:', source, subset, labels, seed, flush=True)
                result = linear_probe(checkpoint, identity, protocol, source, source, train_set, test_set,
                                      labels, seed, output, workers=0, device='cuda')
                score = result['accuracy_percent']
                if (not result['complete'] or not result['encoder_unchanged']
                        or result['encoder_sha256'] != marker['encoder_sha256']
                        or not math.isfinite(score) or not 0 <= score <= 100):
                    raise ValueError('Invalid frozen-probe result')
                samples.append(dict(ssl_seed=0, probe_seed=seed, accuracy_percent=score,
                                    encoder_sha256=marker['encoder_sha256'], result_file=str(output)))
            scores = [s['accuracy_percent'] for s in samples]
            summary = dict(complete=True, experiment=experiment, source=source, target=source,
                model='ResNet18', kind='ours', subset=subset, ssl='simclr', ssl_epochs=2000,
                linear_epochs=100, labels=labels, ssl_seeds=[0], probe_seeds=list(range(6)),
                completed_probes=6, expected_probes=6, std_ddof=1, samples=samples,
                mean_percent=statistics.mean(scores), std_percent=statistics.stdev(scores))
            if accepted:
                summary['accepted_generation'] = accepted
            summary['tex_value'] = '\\dsacc{%.2f}{%.2f}' % (summary['mean_percent'], summary['std_percent'])
            atomic_json(folder / ('summary_lbl%d.json' % labels), summary)
            print('COMPLETE CELL:', source, subset, labels, summary['tex_value'], flush=True)


def main():
    manifest = Path(sys.argv[1]).resolve()
    if not manifest.is_relative_to(HERE / 'batches'):
        raise ValueError('Unexpected manifest')
    job = read(manifest)
    if (not os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_JOB_ACCOUNT') != job['account']
            or job['account'] not in ('aip-boyuwang', 'aip-yiweilu')
            or os.environ.get('SLURM_CLUSTER_NAME', os.environ.get('CC_CLUSTER')) != 'vulcan'
            or not torch.cuda.is_available() or torch.cuda.device_count() != 1):
        raise RuntimeError('Use the configured Wang/Lu single-GPU Vulcan job')
    if (job['source'], job['subset']) not in SETTINGS or job['stage'] not in ('generation', 'ssl', 'probe'):
        raise ValueError('Unrequested dataset, size or stage')
    for relative, expected in job['code_hashes'].items():
        actual = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError('Code changed since submission: ' + relative)
    teacher = ROOT / job['teacher']['checkpoint']
    teacher_marker = teacher.parent / 'ssl_complete.json'
    if (hashlib.sha256(teacher_marker.read_bytes()).hexdigest() != job['teacher']['marker_sha256']
            or not read(teacher_marker)['complete'] or not teacher.is_file()):
        raise ValueError('Saved teacher completion record changed or encoder missing')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if job['stage'] == 'generation':
        generation(job, Path(sys.argv[2]).resolve())
    else:
        evaluation(job, Path(sys.argv[2]).resolve())


if __name__ == '__main__':
    main()
