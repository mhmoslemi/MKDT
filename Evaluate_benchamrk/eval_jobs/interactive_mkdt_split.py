"""Evaluate one dataset and one MKDT table scope in an existing GPU allocation.

Main and downstream processes share the normal MKDT result directory. Existing
probe JSONs and full completion markers are reused; missing checkpoints are
skipped. A full completion marker is created once both scopes are available.
"""
import argparse
import contextlib
from datetime import datetime, timezone
import fcntl
import functools
import gc
import json
import os
from pathlib import Path
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent
sys.path.insert(0, str(BENCH))

import torch

from common import atomic_json, exclusive_run, file_hash
from data import RawDataset
from evaluate import linear_probe
from interactive_mkdt import Tee, evaluation_contract
from mkdt import SOURCES, evaluate_seed, expected_protocol, read
from report import report
from worker import validate_checkpoint


@contextlib.contextmanager
def blocking_output(folder):
    """Serialize main/downstream work on the same encoder output."""
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def scoped_cells(unit, scope):
    if scope == 'main':
        return [cell for cell in unit['cells'] if cell['target'] == unit['source']]
    return [cell for cell in unit['cells'] if cell['target'] != unit['source']]


def evaluate_scope(unit, cells, scope, plan, root, datasets, hashes,
                   driver_hashes, worker_hash):
    source = Path(plan['checkpoint_root']) / unit['relative']
    output = root / unit['relative']
    protocol = expected_protocol(unit['source'], unit['subset'])
    marker_path = output / f'complete_{scope}.json'
    with blocking_output(output), exclusive_run(source):
        config, saved, checksum = validate_checkpoint(
            source, unit, 'mkdt', protocol, hashes, worker_hash)
        if datasets(unit['source'])[0].fingerprint != config['signature']['source_fingerprint']:
            raise ValueError('Source training split differs from the MKDT training split')
        if config['signature'].get('input_scope') != 'saved_synthetic_images_only':
            raise ValueError('Checkpoint is not from the saved MKDT synthetic images')
        identity = config['identity']
        if (output / 'config.json').exists() and read(output / 'config.json') != config:
            raise ValueError('Refusing to mix checkpoint configurations in one output directory')
        atomic_json(output / 'config.json', config)
        requested = [[cell['target'], int(cell['labels'])] for cell in cells]
        marker = {
            'complete': True, 'scope': scope, 'identity': identity,
            'encoder_sha256': saved['encoder_sha256'],
            'checkpoint_file_sha256': checksum, 'checkpoint_retained': True,
            'checkpoint': str(source / 'encoder.pt'), 'evaluations': requested,
            'evaluation_code_hashes': hashes, 'driver_hashes': driver_hashes,
            'ssl_epochs': saved['epochs']}
        prior = read(marker_path) if marker_path.exists() else None
        if prior and any(prior.get(key) != value for key, value in marker.items()):
            raise ValueError(f'Completed {scope} evaluation provenance differs from this run')

        result_hashes = {}
        for cell in cells:
            target, label = cell['target'], int(cell['labels'])
            path = output / 'evaluations' / f'{target}_lbl{label}.json'
            if prior and (not path.exists()
                          or file_hash(path) != prior['result_hashes'][path.name]):
                raise ValueError(f'Completed result was changed or removed: {path}')
            train_set, test_set = datasets(target)
            result = linear_probe(
                source / 'encoder.pt', identity, protocol, unit['source'], target,
                train_set, test_set, label, unit['seed'], path, 0, 'cuda')
            if (not result.get('complete') or not result['encoder_unchanged']
                    or result['encoder_sha256'] != saved['encoder_sha256']):
                raise ValueError(f'Probe changed or used a different encoder: {path}')
            result_hashes[path.name] = file_hash(path)
        if file_hash(source / 'encoder.pt') != checksum:
            raise ValueError('MKDT SSL checkpoint changed during evaluation')
        atomic_json(marker_path, {**marker, 'result_hashes': result_hashes})
    print(f'{scope.upper()} SCOPE EVALUATED', unit['relative'], flush=True)


def run(args, root, session, plan, hashes, driver_hashes, worker_hash):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; run this on your allocated GPU node')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    @functools.lru_cache(maxsize=7)
    def datasets(target):
        return RawDataset(args.data_root, target, True), RawDataset(args.data_root, target, False)

    selected = []
    for job in plan['jobs']:
        for unit in job['units']:
            if unit['source'] != args.source:
                continue
            cells = scoped_cells(unit, args.scope)
            if cells:
                selected.append((unit, cells))
    selected.sort(key=lambda item: (
        item[0]['source'], item[0]['model'], item[0]['subset'],
        item[0]['ssl'], item[0]['seed']))
    expected = 72 if args.scope == 'main' else 24
    if len(selected) != expected:
        raise ValueError(f'Expected {expected} {args.scope} MKDT encoders, found {len(selected)}')

    counts = {'evaluated_scopes': 0, 'already_complete': 0,
              'missing_checkpoint': 0, 'deferred_finalization': 0}
    failures = []
    interrupted = False
    started = time.monotonic()
    print(f'Sequential MKDT {args.scope} evaluation for {args.source}: '
          f'{expected} possible encoders; missing inputs are skipped.', flush=True)
    print('100-epoch frozen linear probes; completed probe JSONs are validated and reused.',
          flush=True)
    try:
        for unit, cells in selected:
            folder = args.checkpoints / unit['relative']
            missing = [name for name in ('config.json', 'encoder.pt', 'ssl_complete.json')
                       if not (folder / name).is_file() or not (folder / name).stat().st_size]
            if missing:
                counts['missing_checkpoint'] += 1
                print('SKIP missing checkpoint:', unit['relative'], ', '.join(missing), flush=True)
                continue
            output = root / unit['relative']
            full_marker = output / 'complete.json'
            scope_marker = output / f'complete_{args.scope}.json'
            saved_probes = sum(
                (output / 'evaluations' / f"{cell['target']}_lbl{cell['labels']}.json").is_file()
                for cell in cells)
            print(f'CHECK {unit["relative"]}: {saved_probes}/{len(cells)} '
                  f'{args.scope} probes already saved', flush=True)
            try:
                if full_marker.is_file():
                    evaluate_seed(unit, plan, root, datasets, hashes,
                                  driver_hashes, worker_hash)
                    counts['already_complete'] += 1
                    print('SKIP already fully evaluated:', unit['relative'], flush=True)
                    continue
                scope_done = scope_marker.is_file()
                evaluate_scope(unit, cells, args.scope, plan, root, datasets,
                               hashes, driver_hashes, worker_hash)
                counts['already_complete' if scope_done else 'evaluated_scopes'] += 1
                all_results = all(
                    (output / 'evaluations' /
                     f"{cell['target']}_lbl{cell['labels']}.json").is_file()
                    for cell in unit['cells'])
                if all_results:
                    try:
                        evaluate_seed(unit, plan, root, datasets, hashes,
                                      driver_hashes, worker_hash)
                        print('FULL ENCODER EVALUATION COMPLETE:', unit['relative'], flush=True)
                    except RuntimeError as error:
                        if 'Another process is working in' not in str(error):
                            raise
                        counts['deferred_finalization'] += 1
                        print('DEFER full marker; counterpart process is active:',
                              unit['relative'], flush=True)
            except RuntimeError as error:
                if 'Another process is working in' in str(error):
                    counts['deferred_finalization'] += 1
                    print('SKIP busy; counterpart process is active:',
                          unit['relative'], flush=True)
                    continue
                failures.append({'encoder': unit['relative'], 'error': str(error)})
                print('ERROR:', unit['relative'], flush=True)
                traceback.print_exc()
            except Exception as error:
                failures.append({'encoder': unit['relative'], 'error': str(error)})
                print('ERROR:', unit['relative'], flush=True)
                traceback.print_exc()
            finally:
                gc.collect()
                torch.cuda.empty_cache()
            if (counts['evaluated_scopes'] + counts['already_complete']) % 10 == 0:
                report(root, plan, hashes, driver_hashes)
    except KeyboardInterrupt:
        interrupted = True
        print('\nStopped. Completed probe files are retained; rerun the same command.', flush=True)
    finally:
        progress = report(root, plan, hashes, driver_hashes)
        atomic_json(root / 'logs' / f'{session}.json', {
            'source': args.source, 'scope': args.scope, **counts,
            'failures': failures, 'interrupted': interrupted,
            'seconds': time.monotonic() - started, 'progress': progress,
            'split_driver_sha256': file_hash(__file__)})
        print('SESSION:', json.dumps({
            'source': args.source, 'scope': args.scope, **counts,
            'errors': len(failures), 'interrupted': interrupted}), flush=True)
        print('COVERAGE:', json.dumps(progress), flush=True)
        print('Results:', root, flush=True)
    return 130 if interrupted else (1 if failures or progress['invalid_encoders'] else 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', choices=SOURCES, required=True)
    parser.add_argument('--scope', choices=('main', 'downstream'), required=True)
    parser.add_argument('--data-root', type=Path, default=BENCH.parent.parent / 'data')
    parser.add_argument('--checkpoints', type=Path,
                        default=BENCH / 'mkdt_ssl_checkpoints_cifar8k6k_tiny5k4k_6seeds')
    parser.add_argument('--output', type=Path,
                        default=BENCH / 'mkdt_eval_results_interactive_6seeds')
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        parser.error('Run on your allocated GPU node, not the login node')
    if os.environ.get('SLURM_CLUSTER_NAME', os.environ.get('CC_CLUSTER')) != 'killarney':
        parser.error('This runner is for the Killarney MKDT checkpoints')
    args.data_root, args.checkpoints = args.data_root.resolve(), args.checkpoints.resolve()
    root = args.output.resolve()
    if not root.is_relative_to(BENCH) or root == BENCH or root.is_relative_to(args.checkpoints):
        parser.error('Use a separate output directory inside Evaluate_benchamrk, '
                     'outside the checkpoint directory')
    if not args.data_root.is_dir() or not args.checkpoints.is_dir():
        parser.error('Missing data/checkpoint root; check --data-root and --checkpoints')
    root.mkdir(parents=True, exist_ok=True)
    (root / 'logs').mkdir(exist_ok=True)

    plan, hashes, driver_hashes, worker_hash, contract = evaluation_contract(args)
    with (root / '.plan.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        contract_path = root / 'plan.json'
        if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
            raise ValueError('Evaluation inputs/code changed; do not mix results. '
                             'Use a different --output directory.')
        atomic_json(contract_path, contract)

    session = (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
               + f'_{args.source}_{args.scope}_{os.getpid()}')
    log_path = root / 'logs' / f'{session}.log'
    lock_name = f'interactive_session_{args.source}_{args.scope}'
    with exclusive_run(root / lock_name), log_path.open('a', buffering=1) as log:
        with contextlib.redirect_stdout(Tee(sys.stdout, log)), \
                contextlib.redirect_stderr(Tee(sys.stderr, log)):
            print('Session log:', log_path, flush=True)
            try:
                return run(args, root, session, plan, hashes, driver_hashes, worker_hash)
            except Exception:
                traceback.print_exc()
                return 1


if __name__ == '__main__':
    raise SystemExit(main())
