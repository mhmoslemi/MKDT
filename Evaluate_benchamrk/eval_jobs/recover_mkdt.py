"""Recover missing MKDT SSL seeds and probe them in an existing GPU allocation."""
import argparse
import contextlib
import csv
from datetime import datetime, timezone
import fcntl
import functools
import gc
import os
from pathlib import Path
import subprocess
import sys
import traceback

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent
sys.path.insert(0, str(BENCH))

import torch

from common import (atomic_json, atomic_save, digest_json, exclusive_run,
                    file_hash, load_encoder, state_hash)
from data import RawDataset, SOURCE_COUNTS
from evaluate import linear_probe
from interactive_mkdt import Tee, evaluation_contract
from mkdt import SOURCES, evaluate_seed, expected_protocol, read
from report import report
from train_ssl import SSLModel

# Other timeouts were accepted as final; TinyImageNet 2% must finish 4,000.
ACCEPTED = {
    ('CIFAR10', 'ConvNet', 'simclr'): (1,),
    ('CIFAR10', 'VGG11', 'barlowtwins'): (1, 3),
    ('CIFAR100', 'ConvNet', 'simclr'): (5,),
    ('CIFAR100', 'ResNet18', 'simclr'): (3, 5),
    ('CIFAR100', 'ResNet18', 'barlowtwins'): (3, 5),
    ('CIFAR100', 'VGG11', 'simclr'): (1,),
    ('TinyImageNet', 'ConvNet', 'barlowtwins'): (0,),
    ('TinyImageNet', 'ResNet18', 'simclr'): (2,),
    ('TinyImageNet', 'VGG11', 'barlowtwins'): (3,),
}


def accepted_timeout(unit):
    subset = 1 if unit['source'] == 'TinyImageNet' else 2
    return (unit['subset'] == subset and unit['seed'] in ACCEPTED.get(
        (unit['source'], unit['model'], unit['ssl']), ()))


def train_missing(unit, args):
    folder = args.checkpoints / unit['relative']
    if ((folder / 'config.json').exists() and not (folder / 'encoder.pt').exists()
            and not (folder / 'training.pt').exists()):
        raise FileNotFoundError(f'{folder}/training.pt is missing; refusing to restart saved training')
    with (BENCH / 'mkdt_ssl_jobs/tasks.tsv').open() as stream:
        tasks = [row['task'] for row in csv.DictReader(stream, delimiter='\t')
                 if row['source'] == unit['source'] and row['model'] == unit['model']
                 and int(row['subset']) == unit['subset'] and row['ssl'] == unit['ssl']
                 and int(row['seed_start']) <= unit['seed']
                 < int(row['seed_start']) + int(row['seed_count'])]
    if len(tasks) != 1:
        raise ValueError(f'Expected one original training task for {unit["relative"]}')
    action = 'RESUME SSL' if (folder / 'training.pt').exists() else 'TRAIN SSL'
    print(action, unit['relative'], flush=True)
    # The original worker retains its exact signature, augmentation, LR horizon,
    # RNG/optimizer resume, 500-epoch saves, and synthetic image conversion.
    subprocess.run([
        sys.executable, '-u', str(BENCH / 'mkdt_ssl_jobs/worker.py'), tasks[0],
        str(args.data_root), str(args.synthetic_root), str(args.checkpoints)], check=True)


def accepted_encoder(folder, output, unit, protocol, hashes, worker_hash):
    if not accepted_timeout(unit):
        raise ValueError('This seed has no accepted-timeout exception')
    config = read(folder / 'config.json')
    signature, identity = config['signature'], config['identity']
    if (digest_json(signature) != identity['signature']
            or signature['protocol'] != protocol or signature['code_hashes'] != hashes
            or signature['ssl_worker_sha256'] != worker_hash
            or signature.get('ssl_worker') != 'mkdt_ssl_jobs/worker.py'
            or signature['selection_method'] != 'mkdt' or identity['selection_method'] != 'mkdt'
            or identity['seed'] != unit['seed']
            or identity['selection'] != signature['synthetic']
            or signature.get('input_scope') != 'saved_synthetic_images_only'
            or any(signature[key] != unit[key] for key in ('source', 'model', 'subset', 'ssl'))):
        raise ValueError(f'Wrong MKDT training configuration: {folder}')
    rolling_path = folder / 'training.pt'
    rolling_hash = file_hash(rolling_path)
    rolling = torch.load(rolling_path, map_location='cpu', weights_only=True)
    requested = protocol['ssl']['subset_epochs']
    batch = min(identity['selection']['count'], protocol['ssl']['batch_size'])
    steps_per_epoch = identity['selection']['count'] // batch
    if (rolling['identity'] != identity or not requested - 500 <= rolling['epoch'] < requested
            or rolling['steps_per_epoch'] != steps_per_epoch
            or rolling['steps'] != rolling['epoch'] * steps_per_epoch
            or identity['selection']['count'] != int(SOURCE_COUNTS[unit['source']] * unit['subset'] / 100)):
        raise ValueError(f'Accepted timeout is not a final-500-epochs snapshot: {rolling_path}')
    model = SSLModel(unit['model'], protocol['input_size'][unit['source']],
                     protocol['methods'][unit['ssl']])
    model.load_state_dict(rolling['model'], strict=True)
    encoder = model.encoder.cpu()
    checkpoint = output / f'accepted_encoder_epoch{rolling["epoch"]}.pt'
    record = {
        'accepted_epoch': rolling['epoch'], 'requested_epoch': requested,
        'reason': 'user_accepted_MKDT_timeout_as_final',
        'source_checkpoint': str(rolling_path), 'source_checkpoint_sha256': rolling_hash,
        'derived_checkpoint': str(checkpoint), 'recovery_driver_sha256': file_hash(__file__)}
    record_path = output / 'accepted_rolling_checkpoint.json'
    if record_path.exists() and read(record_path) != record:
        raise ValueError('Accepted timeout checkpoint or recovery code changed')
    saved = {
        'kind': 'ssl_reference_encoder', 'complete': True, 'identity': identity,
        'model': unit['model'], 'input_size': protocol['input_size'][unit['source']],
        'source': unit['source'], 'method': unit['ssl'], 'seed': unit['seed'],
        'epochs': rolling['epoch'], 'requested_epochs': requested,
        'steps': rolling['steps'], 'effective_batch': batch,
        'accepted_partial': True, 'acceptance_reason': record['reason'],
        'encoder': encoder.state_dict(), 'encoder_sha256': state_hash(encoder),
        'normalization': protocol['normalization'][unit['source']]}
    if not checkpoint.exists():
        atomic_save(checkpoint, saved)
    checked, _, actual = load_encoder(checkpoint, identity, 'cpu')
    del checked
    if any(actual.get(key) != saved[key] for key in saved if key != 'encoder'):
        raise ValueError('Derived accepted encoder metadata or weights differ')
    if file_hash(rolling_path) != rolling_hash:
        raise ValueError('Rolling checkpoint changed during extraction')
    atomic_json(record_path, record)
    print(f'USE ACCEPTED {unit["relative"]}: saved epoch {saved["epochs"]}/{requested}', flush=True)
    return config, saved, checkpoint, record


def probe_accepted(unit, args, root, datasets, hashes, drivers, worker_hash):
    source, output = args.checkpoints / unit['relative'], root / unit['relative']
    protocol = expected_protocol(unit['source'], unit['subset'])
    with exclusive_run(source), exclusive_run(output):
        config, saved, checkpoint, record = accepted_encoder(
            source, output, unit, protocol, hashes, worker_hash)
        if datasets(unit['source'])[0].fingerprint != config['signature']['source_fingerprint']:
            raise ValueError('Source training split differs from MKDT training')
        if (output / 'config.json').exists() and read(output / 'config.json') != config:
            raise ValueError('Existing evaluation configuration differs')
        atomic_json(output / 'config.json', config)
        checksum, results = file_hash(checkpoint), {}
        for cell in unit['cells']:
            target, label = cell['target'], int(cell['labels'])
            path = output / 'evaluations' / f'{target}_lbl{label}.json'
            result = linear_probe(
                checkpoint, config['identity'], protocol, unit['source'], target,
                *datasets(target), label, unit['seed'], path, 0, 'cuda')
            if (not result.get('complete') or not result['encoder_unchanged']
                    or result['encoder_sha256'] != saved['encoder_sha256']):
                raise ValueError(f'Probe used or changed a different encoder: {path}')
            results[path.name] = file_hash(path)
        if file_hash(checkpoint) != checksum:
            raise ValueError('Accepted encoder changed during evaluation')
        atomic_json(output / 'complete.json', {
            'complete': True, 'identity': config['identity'],
            'encoder_sha256': saved['encoder_sha256'], 'checkpoint_file_sha256': checksum,
            'checkpoint_retained': True, 'checkpoint': str(checkpoint),
            'evaluations': [[cell['target'], int(cell['labels'])] for cell in unit['cells']],
            'evaluation_code_hashes': hashes, 'driver_hashes': drivers,
            'ssl_epochs': saved['epochs'], 'accepted_rolling_checkpoint': record,
            'result_hashes': results})


def run(args, root):
    plan, hashes, drivers, worker_hash, contract = evaluation_contract(args)
    policy = {'recovery_driver_sha256': file_hash(__file__),
              'tinyimagenet_2_percent': 'resume_original_recipe_to_4000'}
    with (root / '.plan.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for name, value in (('plan.json', contract), ('recovery_policy.json', policy)):
            path = root / name
            if path.exists() and read(path) != value:
                raise ValueError(f'Existing MKDT contract differs: {path}')
            atomic_json(path, value)
    progress = report(root, plan, hashes, drivers)
    if progress['invalid_encoders']:
        raise ValueError(f'Invalid existing MKDT results: {progress["invalid_encoders"]}')

    @functools.lru_cache(maxsize=7)
    def datasets(target):
        return RawDataset(args.data_root, target, True), RawDataset(args.data_root, target, False)

    counts = {'already_complete': 0, 'evaluated_encoders': 0}
    failures = []
    interrupted = False
    units = [unit for job in plan['jobs'] for unit in job['units'] if unit['source'] == args.source]
    print(f'MKDT recovery: {args.source}; {len(units)} encoders, sizes 1% and 2%, six seeds.', flush=True)
    print('Original SSL recipe; 100-epoch probes at 1%/5% labels; downstream on ResNet18 only.', flush=True)
    try:
        for unit in units:
            output = root / unit['relative']
            if (output / 'complete.json').exists():
                counts['already_complete'] += 1
                print('SKIP already evaluated:', unit['relative'], flush=True)
                continue
            try:
                folder = args.checkpoints / unit['relative']
                if accepted_timeout(unit) and not (folder / 'encoder.pt').exists():
                    probe_accepted(unit, args, root, datasets, hashes, drivers, worker_hash)
                else:
                    if not all((folder / name).is_file() for name in (
                            'config.json', 'encoder.pt', 'ssl_complete.json')):
                        train_missing(unit, args)
                    evaluate_seed(unit, plan, root, datasets, hashes, drivers, worker_hash)
                counts['evaluated_encoders'] += 1
            except Exception as error:
                failures.append({'encoder': unit['relative'], 'error': str(error)})
                traceback.print_exc()
            finally:
                gc.collect()
                torch.cuda.empty_cache()
            progress = report(root, plan, hashes, drivers)
    except KeyboardInterrupt:
        interrupted = True
        print('Stopped. Rerun the same script to resume SSL and skip completed probes.', flush=True)
    finally:
        progress = report(root, plan, hashes, drivers)
        atomic_json(root / 'logs' / f'recovery_{args.source}.json', {
            **counts, 'failures': failures, 'interrupted': interrupted, 'progress': progress})
        print('SESSION:', {**counts, 'errors': len(failures), 'interrupted': interrupted}, flush=True)
        print('COVERAGE:', progress, flush=True)
        print('Results:', root, flush=True)
    return 130 if interrupted else int(bool(failures or progress['invalid_encoders']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', choices=SOURCES, required=True)
    parser.add_argument('--data-root', type=Path, default=BENCH.parent.parent / 'data')
    parser.add_argument('--synthetic-root', type=Path, default=os.environ.get(
        'MKDT_GENERATED_ROOT', BENCH.parent / 'benchmarks/mkdt_multidataset/results/'
        'batch_20261010T160046Z_Y6dqZS/synthetic/generated'))
    parser.add_argument('--checkpoints', type=Path,
                        default=BENCH / 'mkdt_ssl_checkpoints_cifar8k6k_tiny5k4k_6seeds')
    parser.add_argument('--output', type=Path, default=BENCH / 'mkdt_eval_results_interactive_6seeds')
    args = parser.parse_args()
    if (not os.environ.get('SLURM_JOB_ID')
            or os.environ.get('SLURM_CLUSTER_NAME', os.environ.get('CC_CLUSTER')) != 'killarney'):
        parser.error('Run inside your existing Killarney GPU allocation')
    if not torch.cuda.is_available():
        parser.error('CUDA unavailable; no CPU fallback')
    args.data_root, args.synthetic_root = args.data_root.resolve(), args.synthetic_root.resolve()
    args.checkpoints, root = args.checkpoints.resolve(), args.output.resolve()
    if (not root.is_relative_to(BENCH) or root == BENCH
            or root.is_relative_to(args.checkpoints) or args.checkpoints.is_relative_to(root)
            or not args.checkpoints.is_relative_to(BENCH) or args.checkpoints == BENCH):
        parser.error('Use separate checkpoint/result directories inside Evaluate_benchamrk')
    if not all(path.is_dir() for path in (args.data_root, args.synthetic_root, args.checkpoints)):
        parser.error('Missing source data, MKDT synthetic images, or checkpoint root')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    (root / 'logs').mkdir(parents=True, exist_ok=True)
    session = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + f'_{os.getpid()}'
    log_path = root / 'logs' / f'recovery_{args.source}_{session}.log'
    with exclusive_run(root / f'interactive_session_{args.source}'), log_path.open('a', buffering=1) as log:
        with contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, log)):
            print('Session log:', log_path, flush=True)
            return run(args, root)


if __name__ == '__main__':
    raise SystemExit(main())
