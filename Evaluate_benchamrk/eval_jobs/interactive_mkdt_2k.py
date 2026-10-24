"""MKDT SimCLR rerun: two 2,000-epoch encoders and three probes per encoder."""
import argparse
import functools
import gc
import itertools
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

from common import atomic_json, code_hashes, digest_json, exclusive_run, file_hash
from data import RawDataset, SyntheticDataset
from evaluate import linear_probe
from mkdt import expected_protocol
from report import read, write_tsv
from run import evaluation_targets, validate_protocol
from train_ssl import train

SSL_SEEDS = (0, 1)
PROBE_SEEDS = (0, 1, 2)
MODELS = ('ResNet18', 'VGG11', 'ConvNet')
SIZES = (1, 2)
SOURCES = ('CIFAR10', 'CIFAR100', 'TinyImageNet')


def protocol_for(source, percentage):
    protocol = expected_protocol(source, percentage)
    protocol['ssl'].update(subset_epochs=2000, checkpoint_every_epochs=500)
    protocol['subset_seeds'] = list(SSL_SEEDS)
    validate_protocol(protocol)
    assert protocol['evaluation']['epochs'] == 100
    return protocol


def synthetic_entry(root, source, model, percentage):
    generator = 'VGG11' if model == 'VGG11' else 'ConvNet'
    token, iteration = ('Tiny', 10000) if source == 'TinyImageNet' else (source, 5000)
    path = root / generator / source / f'{token}_{percentage}pct' / f'images_{iteration}.pt'
    return {'path': str(path), 'key': None, 'space': 'repository_normalized',
            'pixel_policy': 'clamp_to_unit_rgb', 'generator': generator}


def summarize(group, signature, targets):
    signature_hash = digest_json(signature)
    rows = []
    for target, labels in itertools.product(targets, (1, 5)):
        samples, fingerprints = [], set()
        for ssl_seed, probe_seed in itertools.product(SSL_SEEDS, PROBE_SEEDS):
            folder = group / f'ssl_seed_{ssl_seed:03d}'
            path = folder / 'evaluations' / f'{target}_lbl{labels}_probe{probe_seed}.json'
            if not path.is_file():
                continue
            config, marker, result = (read(folder / 'config.json'),
                                      read(folder / 'ssl_complete.json'), read(path))
            identity = {'signature': signature_hash, 'seed': ssl_seed, 'selection_method': 'mkdt'}
            expected, score = result['expected'], result['accuracy_percent']
            if (config != {'signature': signature, 'identity': identity}
                    or not marker.get('complete') or marker['identity'] != identity
                    or marker['epochs'] != 2000
                    or not result.get('complete') or not result.get('encoder_unchanged')
                    or result['encoder_sha256'] != marker['encoder_sha256']
                    or expected['identity'] != identity or expected['seed'] != probe_seed
                    or expected['source'] != signature['source'] or expected['target'] != target
                    or expected['label_percentage'] != labels
                    or expected['evaluation'] != signature['protocol']['evaluation']
                    or expected['evaluation_code_hashes'] != signature['code_hashes']
                    or not math.isfinite(score) or not 0 <= score <= 100):
                raise ValueError(f'Invalid MKDT probe result: {path}')
            fingerprints.add((expected['train_fingerprint'], expected['test_fingerprint']))
            if len(fingerprints) != 1:
                raise ValueError(f'Mixed target datasets: {target}')
            samples.append({'ssl_seed': ssl_seed, 'probe_seed': probe_seed, 'accuracy_percent': score})
        complete = len(samples) == 6
        if complete:
            assert {(s['ssl_seed'], s['probe_seed']) for s in samples} == set(
                itertools.product(SSL_SEEDS, PROBE_SEEDS))
        scores = [sample['accuracy_percent'] for sample in samples]
        rows.append({
            'source': signature['source'], 'model': signature['model'], 'method': 'mkdt',
            'subset': signature['subset'], 'ssl': 'simclr', 'target': target, 'labels': labels,
            'completed_probes': len(samples), 'expected_probes': 6, 'complete': complete,
            'mean_percent': statistics.mean(scores) if complete else None,
            'std_percent': statistics.stdev(scores) if complete else None, 'samples': samples})
    atomic_json(group / 'summary.json', {
        'signature': signature_hash, 'ssl_seeds': list(SSL_SEEDS), 'probe_seeds': list(PROBE_SEEDS),
        'std_ddof': 1, 'complete': all(row['complete'] for row in rows), 'cells': rows})
    return rows


def evaluate_set(args, model, percentage, datasets, hashes):
    protocol = protocol_for(args.source, percentage)
    entry = synthetic_entry(args.synthetic_root, args.source, model, percentage)
    if not Path(entry['path']).is_file():
        print('SKIP missing MKDT images:', entry['path'], flush=True)
        return []
    # Identical raw-tensor loading, generation normalization and PIL conversion
    # to the original MKDT worker; only the SSL horizon and repeat counts change.
    synthetic = SyntheticDataset(entry, args.source, percentage, protocol['input_size'][args.source])
    targets = evaluation_targets(args.source, model)
    signature = {
        'protocol': protocol, 'source': args.source, 'model': model, 'subset': percentage,
        'ssl': 'simclr', 'selection_method': 'mkdt',
        'source_fingerprint': datasets(args.source)[0].fingerprint,
        'synthetic': synthetic.metadata, 'input_scope': 'saved_synthetic_images_only',
        'ssl_seeds': list(SSL_SEEDS), 'probe_seeds': list(PROBE_SEEDS),
        'code_hashes': hashes, 'driver_sha256': file_hash(__file__)}
    group = args.output / args.source / model / 'mkdt' / f'pt{percentage}' / 'simclr'
    failures = []
    with exclusive_run(group):
        for ssl_seed in SSL_SEEDS:
            folder = group / f'ssl_seed_{ssl_seed:03d}'
            folder.mkdir(exist_ok=True)
            identity = {'signature': digest_json(signature), 'seed': ssl_seed, 'selection_method': 'mkdt'}
            config = {'signature': signature, 'identity': identity}
            if (folder / 'config.json').exists() and read(folder / 'config.json') != config:
                raise ValueError(f'MKDT inputs/code changed; do not mix evaluations: {folder}')
            atomic_json(folder / 'config.json', config)
            print(f'SSL CHECK {args.source}/{model}/pt{percentage}/simclr seed={ssl_seed}, epochs=2000', flush=True)
            try:
                checkpoint, saved = train(synthetic, protocol, args.source, model, 'simclr',
                                          ssl_seed, folder, identity, workers=0, device='cuda')
                if (saved['epochs'] != 2000 or not saved['complete']
                        or saved['source'] != args.source or saved['model'] != model
                        or saved['method'] != 'simclr' or saved['seed'] != ssl_seed):
                    raise ValueError(f'Wrong SSL encoder metadata: {checkpoint}')
                checksum = file_hash(checkpoint)
                marker = {'complete': True, 'identity': identity, 'epochs': saved['epochs'],
                          'encoder_sha256': saved['encoder_sha256'], 'checkpoint_file_sha256': checksum}
                marker_path = folder / 'ssl_complete.json'
                if marker_path.exists() and read(marker_path) != marker:
                    raise ValueError(f'Completed SSL checkpoint changed: {checkpoint}')
                atomic_json(marker_path, marker)
            except Exception as error:
                failures.append(f'{folder}: {error}')
                traceback.print_exc()
                continue
            for target in targets:
                try:
                    target_train, target_test = datasets(target)
                except FileNotFoundError as error:
                    failures.append(str(error))
                    print('SKIP missing downstream dataset:', error, flush=True)
                    continue
                for labels, probe_seed in itertools.product((1, 5), PROBE_SEEDS):
                    path = folder / 'evaluations' / f'{target}_lbl{labels}_probe{probe_seed}.json'
                    print(f'{"REUSE" if path.exists() else "PROBE"} {target} lbl={labels}, '
                          f'SSL seed={ssl_seed}, probe seed={probe_seed}', flush=True)
                    try:
                        result = linear_probe(
                            checkpoint, identity, protocol, args.source, target, target_train, target_test,
                            labels, probe_seed, path, workers=0, device='cuda')
                        if (not result.get('complete') or not result['encoder_unchanged']
                                or result['encoder_sha256'] != saved['encoder_sha256']):
                            raise ValueError(f'Probe used or changed a different encoder: {path}')
                        summarize(group, signature, targets)
                    except Exception as error:
                        failures.append(f'{path}: {error}')
                        traceback.print_exc()
            if file_hash(checkpoint) != checksum:
                raise ValueError(f'SSL checkpoint changed during probing: {checkpoint}')
    return failures


def write_report(args):
    rows = []
    for model, percentage in itertools.product(MODELS, SIZES):
        path = args.output / args.source / model / 'mkdt' / f'pt{percentage}' / 'simclr/summary.json'
        if path.is_file():
            rows.extend(read(path)['cells'])
    write_tsv(args.output / args.source / 'table_summary.tsv', [
        'source', 'model', 'method', 'subset', 'ssl', 'target', 'labels', 'completed_probes',
        'expected_probes', 'complete', 'mean_percent', 'std_percent'], rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', choices=SOURCES)
    parser.add_argument('--data-root', type=Path, default=BENCH.parent.parent / 'data')
    parser.add_argument('--synthetic-root', type=Path, default=os.environ.get(
        'MKDT_GENERATED_ROOT', BENCH.parent / 'benchmarks/mkdt_multidataset/results/'
        'batch_20261010T160046Z_Y6dqZS/synthetic/generated'))
    parser.add_argument('--output', type=Path, default=BENCH / 'mkdt_eval_results_simclr2k_2ssl3probes')
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():
        parser.error('Run inside your existing GPU allocation, not on the login node')
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        parser.error('Use one process and one GPU per launcher')
    args.output, args.synthetic_root, args.data_root = (
        path.resolve() for path in (args.output, args.synthetic_root, args.data_root))
    if (not args.output.is_relative_to(BENCH) or args.output == BENCH
            or args.output.is_relative_to(args.synthetic_root) or args.synthetic_root.is_relative_to(args.output)):
        parser.error('Use a separate evaluation directory inside Evaluate_benchamrk')
    if not args.synthetic_root.is_dir() or not args.data_root.is_dir():
        parser.error('Missing MKDT synthetic/source data; check the input roots')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    hashes, failures = code_hashes(), []

    @functools.lru_cache(maxsize=7)
    def datasets(target):
        return RawDataset(args.data_root, target, True), RawDataset(args.data_root, target, False)

    print('MKDT SimCLR only: 2,000 SSL epochs; two encoders x three 100-epoch probes per cell.', flush=True)
    print('Sizes 1%/2%; labels 1%/5%; main on all three BN networks; downstream on ResNet18 only.', flush=True)
    print('SSL resumes from training.pt (replaced every 500 epochs); finished probes are reused.', flush=True)
    print('An interrupted linear head restarts; old six-seed/long-epoch runs are not reused.', flush=True)
    with exclusive_run(args.output / args.source / 'interactive_session'):
        try:
            for model, percentage in itertools.product(MODELS, SIZES):
                try:
                    failures.extend(evaluate_set(args, model, percentage, datasets, hashes))
                except Exception as error:
                    failures.append(f'{args.source}/{model}/pt{percentage}: {error}')
                    traceback.print_exc()
                finally:
                    write_report(args)
                    gc.collect()
                    torch.cuda.empty_cache()
        except KeyboardInterrupt:
            print('Stopped. Rerun the same launcher to resume SSL and skip completed probes.', flush=True)
            return 130
        finally:
            write_report(args)
            atomic_json(args.output / args.source / 'session.json', {'errors': failures})
            print('Results:', args.output / args.source, flush=True)
    return int(bool(failures))


if __name__ == '__main__':
    raise SystemExit(main())
