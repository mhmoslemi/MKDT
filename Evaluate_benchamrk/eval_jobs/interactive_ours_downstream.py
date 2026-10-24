"""Probe retained ours ResNet18 encoders on downstream datasets; never train SSL."""
import argparse
import fcntl
import functools
import gc
import itertools
import os
from pathlib import Path
import traceback

import interactive_ours as ours
from common import load_encoder
from interactive_ours_2k10k import SSL_EPOCHS


def saved_encoder(args, percentage, seed, hashes):
    folder = args.checkpoints / args.source / 'ResNet18BN' / ('pt%g' % percentage) / args.method / ('ssl_seed_%03d' % seed)
    missing = [name for name in ('config.json', 'encoder.pt', 'ssl_complete.json')
               if not (folder / name).is_file() or not (folder / name).stat().st_size]
    if missing:
        print('SKIP missing checkpoint:', folder, ', '.join(missing), flush=True)
        return None
    config, marker = ours.read(folder / 'config.json'), ours.read(folder / 'ssl_complete.json')
    signature, identity = config['signature'], config['identity']
    protocol = ours.expected_protocol('random', args.source, percentage)
    protocol['ssl'].update(subset_epochs=SSL_EPOCHS[args.method], checkpoint_every_epochs=500)
    protocol['subset_seeds'] = list(ours.SSL_SEEDS)
    expected = {'source': args.source, 'model': 'ResNet18', 'subset': percentage,
                'ssl': args.method, 'selection_method': 'ours', 'protocol': protocol,
                'ssl_seeds': list(ours.SSL_SEEDS), 'probe_seeds': list(ours.PROBE_SEEDS),
                'code_hashes': hashes, 'driver_sha256': ours.file_hash(Path(ours.__file__))}
    if (any(signature.get(key) != value for key, value in expected.items())
            or identity != {'signature': ours.digest_json(signature), 'seed': seed, 'selection_method': 'ours'}
            or not marker.get('complete') or marker['identity'] != identity
            or marker['epochs'] != SSL_EPOCHS[args.method]):
        raise ValueError('Wrong ours checkpoint configuration: %s' % folder)
    checkpoint = folder / 'encoder.pt'
    if ours.file_hash(checkpoint) != marker['checkpoint_file_sha256']:
        raise ValueError('Checkpoint file hash mismatch: %s' % checkpoint)
    encoder, _, saved = load_encoder(checkpoint, identity, device='cpu')
    metadata = {'source': args.source, 'model': 'ResNet18', 'method': args.method,
                'seed': seed, 'epochs': SSL_EPOCHS[args.method],
                'input_size': protocol['input_size'][args.source],
                'normalization': protocol['normalization'][args.source],
                'encoder_sha256': marker['encoder_sha256']}
    if any(saved.get(key) != value for key, value in metadata.items()):
        raise ValueError('Encoder metadata mismatch: %s' % checkpoint)
    del encoder, saved
    return checkpoint, config, marker


def write_report(args):
    with (args.output / args.source / '.report.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        ours.write_report(args)


def evaluate_seed(args, percentage, seed, targets, datasets, hashes, counts):
    loaded = saved_encoder(args, percentage, seed, hashes)
    if loaded is None:
        counts['missing_checkpoints'] += 1
        return []
    checkpoint, config, marker = loaded
    group = args.output / args.source / 'ResNet18BN' / ('pt%g' % percentage) / args.method
    folder = group / ('ssl_seed_%03d' % seed)
    folder.mkdir(parents=True, exist_ok=True)
    reference = {'checkpoint': str(checkpoint), 'checkpoint_file_sha256': marker['checkpoint_file_sha256'],
                 'driver_sha256': ours.file_hash(Path(__file__)), 'targets': targets, 'scope': 'downstream'}
    # Keep references to source encoders, not copies of their weights.
    for name, value in {'config.json': config, 'ssl_complete.json': {**marker, 'checkpoint': str(checkpoint)},
                        'checkpoint_reference.json': reference}.items():
        destination = folder / name
        if destination.exists() and ours.read(destination) != value:
            raise ValueError('Changed downstream inputs/code: %s' % destination)
        ours.atomic_json(destination, value)
    failures = []
    for target in targets:
        try:
            train_set, test_set = datasets(target)
        except FileNotFoundError as error:
            failures.append(str(error))
            print('SKIP missing downstream dataset:', error, flush=True)
            continue
        for labels, probe_seed in itertools.product((1, 5), ours.PROBE_SEEDS):
            output = folder / 'evaluations' / ('%s_lbl%d_probe%d.json' % (target, labels, probe_seed))
            reused = output.is_file()
            print('%s %s/pt%g/%s -> %s lbl%d, SSL seed=%d, probe seed=%d' %
                  ('REUSE' if reused else 'PROBE', args.source, percentage, args.method,
                   target, labels, seed, probe_seed), flush=True)
            try:
                result = ours.linear_probe(checkpoint, config['identity'], config['signature']['protocol'],
                                           args.source, target, train_set, test_set, labels, probe_seed,
                                           output, workers=0, device='cuda')
                if (not result.get('complete') or not result.get('encoder_unchanged')
                        or result['encoder_sha256'] != marker['encoder_sha256']):
                    raise ValueError('Probe used or changed a different encoder: %s' % output)
                counts['reused_probes' if reused else 'new_probes'] += 1
            except Exception as error:
                failures.append('%s: %s' % (output, error))
                traceback.print_exc()
            ours.summarize(group, config['identity']['signature'], args.source, 'ResNet18BN',
                           percentage, args.method, targets)
            write_report(args)
    if ours.file_hash(checkpoint) != marker['checkpoint_file_sha256']:
        raise ValueError('Source encoder changed during probing: %s' % checkpoint)
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', choices=('CIFAR10', 'CIFAR100', 'TinyImageNet'))
    parser.add_argument('method', choices=tuple(SSL_EPOCHS))
    parser.add_argument('--data-root', type=Path, default=ours.BENCH.parent.parent / 'data')
    parser.add_argument('--checkpoints', type=Path,
                        default=ours.BENCH / 'ours_eval_results_simclr2k_barlow10k_2ssl3probes_main')
    parser.add_argument('--output', type=Path,
                        default=ours.BENCH / 'ours_eval_results_simclr2k_barlow10k_2ssl3probes_downstream')
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or not ours.torch.cuda.is_available():
        parser.error('Run inside your existing GPU allocation, not on the login node')
    args.output, args.checkpoints, args.data_root = (p.resolve() for p in
                                                  (args.output, args.checkpoints, args.data_root))
    if (not args.output.is_relative_to(ours.BENCH) or args.output == ours.BENCH
            or args.output.is_relative_to(args.checkpoints) or args.checkpoints.is_relative_to(args.output)):
        parser.error('Keep downstream output separate from the saved encoder directory, inside Evaluate_benchamrk')
    ours.torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', '1')))
    ours.torch.set_num_interop_threads(1)
    ours.torch.backends.cuda.matmul.allow_tf32 = False
    ours.torch.backends.cudnn.allow_tf32 = False
    assert ours.SSL_SEEDS == (0, 1) and ours.PROBE_SEEDS == (0, 1, 2)
    targets = [target for target in ours.evaluation_targets(args.source, 'ResNet18') if target != args.source]
    assert len(targets) == len(set(targets)) == 6 and args.source not in targets
    (args.output / args.source).mkdir(parents=True, exist_ok=True)
    hashes, failures = ours.code_hashes(), []
    counts = {'new_probes': 0, 'reused_probes': 0, 'missing_checkpoints': 0}

    @functools.lru_cache(maxsize=6)
    def datasets(target):
        return ours.RawDataset(args.data_root, target, True), ours.RawDataset(args.data_root, target, False)

    print('DOWNSTREAM ONLY: %s/%s, ResNet18BN, sizes 0.2%%/1%%/2%%; no SSL training.' %
          (args.source, args.method), flush=True)
    print('Two saved SSL encoders x three 100-epoch probes per target/label cell. Targets:', targets, flush=True)
    print('Missing encoders are skipped. Completed probes are reused; an interrupted head restarts.', flush=True)
    with ours.exclusive_run(args.output / args.source / ('interactive_session_' + args.method)):
        try:
            for percentage in ours.SIZES:
                group = args.output / args.source / 'ResNet18BN' / ('pt%g' % percentage) / args.method
                with ours.exclusive_run(group):
                    for seed in ours.SSL_SEEDS:
                        try:
                            failures.extend(evaluate_seed(args, percentage, seed, targets, datasets, hashes, counts))
                        except Exception as error:
                            failures.append('pt%g/seed%d: %s' % (percentage, seed, error))
                            traceback.print_exc()
                        finally:
                            gc.collect()
                            ours.torch.cuda.empty_cache()
        except KeyboardInterrupt:
            print('Stopped. Rerun the same launcher to reuse completed probes.', flush=True)
            return 130
        finally:
            write_report(args)
            ours.atomic_json(args.output / args.source / ('session_%s.json' % args.method),
                             {**counts, 'errors': failures})
            print('SESSION:', counts, 'errors=%d' % len(failures), flush=True)
            print('Results:', args.output / args.source, flush=True)
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
