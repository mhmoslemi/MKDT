"""Evaluate finished method1 synthetic sets: two SSL seeds, three probes per seed.

Reuse the benchmark trainer/probe without changing existing benchmark identities.
One pass only; rerun to resume SSL, reuse finished probes, or pick up new sets.
"""
import argparse
import functools
import gc
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import statistics
import sys
import traceback

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent
sys.path.insert(0, str(BENCH))

import torch
from torchvision.transforms.functional import to_pil_image

from common import atomic_json, code_hashes, digest_json, exclusive_run, file_hash
from data import ImageView, RawDataset, SOURCE_COUNTS
from evaluate import linear_probe
from report import read, write_tsv
from run import evaluation_targets
from train_ssl import train
from worker import expected_protocol

SSL_SEEDS = (0, 1)
PROBE_SEEDS = (0, 1, 2)
METHODS = ('simclr', 'barlowtwins')
MODELS = ('ResNet18BN', 'VGG11BN', 'ConvNetBN')
SIZES = (0.2, 1, 2)


def load_synthetic(path, source, model, percentage):
    if not path.is_file():
        print('SKIP missing synthetic set:', path, flush=True)
        return None
    saved = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    target_iteration = {0.2: 10000, 1: 8000, 2: 6000}[percentage]
    if not saved.get('complete') or saved.get('iteration', -1) != target_iteration:
        print('SKIP unfinished synthetic set: %s (iteration %s/%d)' %
              (path, saved.get('iteration', '?'), target_iteration), flush=True)
        return None
    if (saved.get('dataset') != source or saved.get('model') != model
            or saved.get('percentage') != percentage):
        raise ValueError('Synthetic checkpoint dataset/model/size mismatch: %s' % path)
    images = saved['data'].detach().cpu().contiguous()
    size = 64 if source == 'TinyImageNet' else 32
    count = int(SOURCE_COUNTS[source] * percentage / 100)
    if (images.shape != (count, 3, size, size) or not images.is_floating_point()
            or not torch.isfinite(images).all()):
        raise ValueError('Invalid synthetic image tensor: %s' % path)
    normalization = saved['normalization']
    mean, std = (torch.tensor(values).reshape(1, 3, 1, 1) for values in normalization)
    if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
        raise ValueError('Invalid synthetic normalization: %s' % path)
    pixels = images * std + mean
    metadata = {'path': str(path.resolve()), 'iteration': saved['iteration'],
                'images_sha256': hashlib.sha256(images.numpy().tobytes()).hexdigest(),
                'dtype': str(images.dtype), 'normalization': normalization,
                'clipped_fraction': float(((pixels < 0) | (pixels > 1)).float().mean()),
                'pixel_policy': 'denormalize_clamp_then_PIL_uint8'}
    # Use the checkpoint's normalization, not the legacy benchmark synthetic statistics.
    tensors = torch.utils.data.TensorDataset(pixels.clamp(0, 1).contiguous(), torch.full((count,), -1))
    return ImageView(tensors, to_pil_image), metadata


def summarize(group, signature, source, model, percentage, method, targets):
    rows = []
    for target, labels in itertools.product(targets, (1, 5)):
        samples = []
        for ssl_seed, probe_seed in itertools.product(SSL_SEEDS, PROBE_SEEDS):
            folder = group / ('ssl_seed_%03d' % ssl_seed)
            path = folder / 'evaluations' / ('%s_lbl%d_probe%d.json' % (target, labels, probe_seed))
            if not path.is_file():
                continue
            result, marker = read(path), read(folder / 'ssl_complete.json')
            identity = {'signature': signature, 'seed': ssl_seed, 'selection_method': 'ours'}
            expected = result['expected']
            score = result['accuracy_percent']
            if (not result.get('complete') or not result.get('encoder_unchanged')
                    or expected['identity'] != identity or expected['seed'] != probe_seed
                    or expected['source'] != source or expected['target'] != target
                    or expected['label_percentage'] != labels
                    or marker['identity'] != identity or not marker.get('complete')
                    or result['encoder_sha256'] != marker['encoder_sha256']
                    or not math.isfinite(score) or not 0 <= score <= 100):
                raise ValueError('Incompatible probe result: %s' % path)
            samples.append({'ssl_seed': ssl_seed, 'probe_seed': probe_seed, 'accuracy_percent': score})
        scores = [sample['accuracy_percent'] for sample in samples]
        complete = len(scores) == len(SSL_SEEDS) * len(PROBE_SEEDS)
        if complete:
            assert {(s['ssl_seed'], s['probe_seed']) for s in samples} == set(itertools.product(SSL_SEEDS, PROBE_SEEDS))
        rows.append({'source': source, 'model': model, 'subset': percentage, 'ssl': method,
                     'target': target, 'labels': labels, 'completed_probes': len(scores),
                     'expected_probes': 6, 'complete': complete,
                     'mean_percent': statistics.mean(scores) if complete else None,
                     'std_percent': statistics.stdev(scores) if complete else None, 'samples': samples})
    atomic_json(group / 'summary.json', {'signature': signature, 'ssl_seeds': SSL_SEEDS,
                'probe_seeds': PROBE_SEEDS, 'std_ddof': 1,
                'complete': all(row['complete'] for row in rows), 'cells': rows})
    for row in rows:
        if row['complete']:
            print('CELL %s/%s/pt%g/%s -> %s lbl%d: %.2f +/- %.2f (6 probes)' %
                  (source, model, percentage, method, row['target'], row['labels'],
                   row['mean_percent'], row['std_percent']), flush=True)


def evaluate_set(args, synthetic, metadata, model, percentage, method, datasets, hashes):
    canonical = model.removesuffix('BN')
    protocol = expected_protocol('random', args.source, percentage)
    protocol['ssl']['checkpoint_every_epochs'] = 500
    protocol['subset_seeds'] = list(SSL_SEEDS)
    targets = evaluation_targets(args.source, canonical)
    train_set, _ = datasets(args.source)
    signature = {'protocol': protocol, 'source': args.source, 'model': canonical,
                 'subset': percentage, 'ssl': method, 'selection_method': 'ours',
                 'source_fingerprint': train_set.fingerprint, 'synthetic': metadata,
                 'ssl_seeds': SSL_SEEDS, 'probe_seeds': PROBE_SEEDS,
                 'code_hashes': hashes, 'driver_sha256': file_hash(Path(__file__))}
    signature_hash = digest_json(signature)
    group = args.output / args.source / model / ('pt%g' % percentage) / method
    failures = []
    with exclusive_run(group):
        for ssl_seed in SSL_SEEDS:
            folder = group / ('ssl_seed_%03d' % ssl_seed)
            folder.mkdir(exist_ok=True)
            identity = {'signature': signature_hash, 'seed': ssl_seed, 'selection_method': 'ours'}
            config = {'signature': signature, 'identity': identity}
            config_path = folder / 'config.json'
            if config_path.exists() and read(config_path) != json.loads(json.dumps(config)):
                raise ValueError('Evaluation inputs/code changed; use a fresh --output: %s' % folder)
            atomic_json(config_path, config)
            print('SSL CHECK %s/%s/pt%g/%s seed=%d, epochs=%d' %
                  (args.source, model, percentage, method, ssl_seed, protocol['ssl']['subset_epochs']), flush=True)
            try:
                checkpoint, saved = train(synthetic, protocol, args.source, canonical, method,
                                          ssl_seed, folder, identity, workers=0, device='cuda')
                atomic_json(folder / 'ssl_complete.json', {'complete': True, 'identity': identity,
                            'epochs': saved['epochs'], 'encoder_sha256': saved['encoder_sha256'],
                            'checkpoint_file_sha256': file_hash(checkpoint)})
            except Exception as error:
                failures.append('%s: %s' % (folder, error))
                traceback.print_exc()
                continue
            for target in targets:
                try:
                    target_train, target_test = datasets(target)
                except FileNotFoundError as error:
                    failures.append(str(error))
                    print('MISSING downstream data:', error, flush=True)
                    continue
                for labels, probe_seed in itertools.product((1, 5), PROBE_SEEDS):
                    path = folder / 'evaluations' / ('%s_lbl%d_probe%d.json' % (target, labels, probe_seed))
                    print('%s PROBE %s lbl%d, SSL seed=%d, probe seed=%d' %
                          ('REUSE' if path.exists() else 'RUN', target, labels, ssl_seed, probe_seed), flush=True)
                    try:
                        linear_probe(checkpoint, identity, protocol, args.source, target, target_train,
                                     target_test, labels, probe_seed, path, workers=0, device='cuda')
                        summarize(group, signature_hash, args.source, model, percentage, method, targets)
                    except Exception as error:
                        failures.append('%s: %s' % (path, error))
                        traceback.print_exc()
    return failures


def write_report(args):
    rows = []
    for model, percentage, method in itertools.product(MODELS, SIZES, METHODS):
        path = args.output / args.source / model / ('pt%g' % percentage) / method / 'summary.json'
        if path.is_file():
            rows.extend(read(path)['cells'])
    write_tsv(args.output / args.source / 'table_summary.tsv',
              ['source', 'model', 'subset', 'ssl', 'target', 'labels', 'completed_probes',
               'expected_probes', 'complete', 'mean_percent', 'std_percent'], rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', choices=('CIFAR10', 'CIFAR100', 'TinyImageNet'))
    parser.add_argument('--data-root', type=Path, default=BENCH.parent.parent / 'data')
    parser.add_argument('--synthetics', type=Path, default=BENCH.parent / 'OT-SSL-DD/ours_syntehtic')
    parser.add_argument('--output', type=Path, default=BENCH / 'ours_eval_results_interactive_2ssl3probes')
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        parser.error('Run inside your existing GPU allocation, not on the login node')
    if not torch.cuda.is_available():
        parser.error('CUDA is unavailable')
    args.output, args.synthetics, args.data_root = (path.resolve() for path in
                                                   (args.output, args.synthetics, args.data_root))
    if not args.output.is_relative_to(BENCH) or args.output == BENCH or args.output.is_relative_to(args.synthetics):
        parser.error('Use a dedicated evaluation output directory inside Evaluate_benchamrk')
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', '1')))
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    hashes = code_hashes()
    failures = []

    @functools.lru_cache(maxsize=7)
    def datasets(target):
        return RawDataset(args.data_root, target, True), RawDataset(args.data_root, target, False)

    print('Two SSL encoders per METHOD; three probes per encoder; six results per cell.', flush=True)
    print('SimCLR/Barlow, targets and 1%/5% label budgets are averaged separately.', flush=True)
    print('SSL resumes from the latest 500-epoch checkpoint. Completed probes are reused; an interrupted head restarts.', flush=True)
    with exclusive_run(args.output / args.source / 'interactive_session'):
        try:
            for model, percentage in itertools.product(MODELS, SIZES):
                path = args.synthetics / args.source / model / ('pt%g' % percentage) / (
                    'res_IIC_%s_%s_%gpercent_final.pt' % (args.source, model, percentage))
                try:
                    loaded = load_synthetic(path, args.source, model, percentage)
                    if loaded is None:
                        continue
                    synthetic, metadata = loaded
                    for method in METHODS:
                        try:
                            failures.extend(evaluate_set(args, synthetic, metadata, model, percentage,
                                                         method, datasets, hashes))
                        except Exception as error:
                            failures.append('%s/%s: %s' % (path, method, error))
                            traceback.print_exc()
                        finally:
                            write_report(args)
                            gc.collect()
                            torch.cuda.empty_cache()
                except Exception as error:
                    failures.append('%s: %s' % (path, error))
                    traceback.print_exc()
        except KeyboardInterrupt:
            print('Stopped. Rerun the same command to resume SSL and reuse completed probes.', flush=True)
            return 130
        finally:
            write_report(args)
            atomic_json(args.output / args.source / 'session.json', {'errors': failures})
            print('Results:', args.output / args.source, flush=True)
            print('Rerun to pick up newly completed synthetic sets; no automatic waiting.', flush=True)
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
