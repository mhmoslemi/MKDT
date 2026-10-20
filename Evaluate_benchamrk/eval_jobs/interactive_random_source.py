"""Evaluate one random-subset source dataset inside an existing Killarney GPU allocation.

This changes scheduling only. Probe/checkpoint validation and result provenance stay
identical to interactive_random.py, so all source runners safely share its output.
"""
import argparse
import contextlib
from datetime import datetime, timezone
import fcntl
import functools
import gc
import hashlib
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

from common import atomic_json, code_hashes, exclusive_run, file_hash
from data import RawDataset
from interactive_random import Tee
from report import report
from worker import build_plan, evaluate_seed

SOURCES = ('CIFAR10', 'CIFAR100', 'TinyImageNet')


def evaluation_contract(args):
    """Build the exact contract used by the original all-source runner."""
    plan = build_plan('random', args.checkpoints, args.data_root)
    for job in plan['jobs']:
        for unit in job['units']:
            unit['cells'] = sorted(unit['cells'], key=lambda cell: (
                cell['target'] != unit['source'], cell['target'], int(cell['labels'])))
    hashes = code_hashes()
    driver_hashes = {str(path.relative_to(BENCH)): file_hash(path) for path in (
        HERE / 'interactive_random.py', HERE / 'worker.py', HERE / 'report.py',
        HERE / 'random_tasks.tsv', BENCH / 'table_coverage.tsv', BENCH / 'protocol.json')}
    worker_text = (BENCH / 'random_ssl.sh').read_text().split("<<'PY'\n", 1)[1].split("\nPY\n", 1)[0]
    worker_hash = hashlib.sha256(worker_text.encode()).hexdigest()
    contract = {'plan': plan, 'code_hashes': hashes, 'driver_hashes': driver_hashes,
                'ssl_worker_sha256': worker_hash}
    return plan, hashes, driver_hashes, worker_hash, contract


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

    units = sorted(
        (unit for job in plan['jobs'] for unit in job['units'] if unit['source'] == args.source),
        key=lambda unit: (unit['model'], unit['subset'], unit['ssl'], unit['seed']))
    if len(units) != 180:
        raise ValueError(f'Expected 180 random encoders for {args.source}, found {len(units)}')

    counts = {'evaluated_encoders': 0, 'already_complete': 0, 'missing_checkpoint': 0}
    failures = []
    interrupted = False
    started = time.monotonic()
    print(f'Sequential random evaluation for {args.source}: 180 possible encoders; '
          'missing inputs are skipped.', flush=True)
    print('100-epoch frozen linear probes, SGD lr=0.1 momentum=0.9 wd=0, '
          'batch<=256, cosine LR.', flush=True)
    print('Labels: 1% and 5%; main on all 3 backbones, downstream on ResNet18 only.', flush=True)
    print('Completed probe JSONs are validated and reused.', flush=True)
    try:
        for unit in units:
            folder = args.checkpoints / unit['relative']
            missing = [name for name in ('config.json', 'encoder.pt', 'ssl_complete.json')
                       if not (folder / name).is_file() or not (folder / name).stat().st_size]
            if missing:
                counts['missing_checkpoint'] += 1
                print('SKIP missing checkpoint:', unit['relative'], ', '.join(missing), flush=True)
                continue
            output = root / unit['relative']
            complete_before = (output / 'complete.json').is_file()
            saved_probes = sum(
                (output / 'evaluations' / f"{cell['target']}_lbl{cell['labels']}.json").is_file()
                for cell in unit['cells'])
            print(f'CHECK {unit["relative"]}: {saved_probes}/{len(unit["cells"])} '
                  'probes already saved', flush=True)
            try:
                evaluate_seed(unit, plan, root, datasets, hashes, driver_hashes, worker_hash)
                counts['already_complete' if complete_before else 'evaluated_encoders'] += 1
                print('SKIP already evaluated:' if complete_before else 'DONE:',
                      unit['relative'], flush=True)
            except Exception as error:
                failures.append({'encoder': unit['relative'], 'error': str(error)})
                print('ERROR:', unit['relative'], flush=True)
                traceback.print_exc()
            finally:
                gc.collect()
                torch.cuda.empty_cache()
            if (counts['evaluated_encoders'] + counts['already_complete']) % 10 == 0:
                report(root, plan, hashes, driver_hashes)
    except KeyboardInterrupt:
        interrupted = True
        print('\nStopped. Completed probe files are retained; rerun the same command.', flush=True)
    finally:
        progress = report(root, plan, hashes, driver_hashes)
        atomic_json(root / 'logs' / f'{session}.json', {
            'source': args.source, **counts, 'failures': failures,
            'interrupted': interrupted, 'seconds': time.monotonic() - started,
            'scheduler_sha256': file_hash(Path(__file__)), 'progress': progress})
        print('SESSION:', json.dumps({'source': args.source, **counts,
                                      'errors': len(failures),
                                      'interrupted': interrupted}), flush=True)
        print('COVERAGE:', json.dumps(progress), flush=True)
        print('Results:', root, flush=True)
        print('Re-run to pick up checkpoints that finished after they were skipped. '
              'No automatic waiting.', flush=True)
    return 130 if interrupted else (1 if failures or progress['invalid_encoders'] else 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', choices=SOURCES)
    parser.add_argument('--data-root', type=Path, default=BENCH.parent.parent / 'data')
    parser.add_argument('--checkpoints', type=Path,
                        default=BENCH / 'random_ssl_checkpoints_cifar10k8k6k_tiny6k5k4k')
    parser.add_argument('--output', type=Path,
                        default=BENCH / 'random_eval_results_interactive_10seeds')
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        parser.error('Run on your allocated GPU node, not the login node')
    if os.environ.get('SLURM_CLUSTER_NAME', os.environ.get('CC_CLUSTER')) != 'killarney':
        parser.error('This runner is for the Killarney random checkpoints')
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
               + f'_{args.source}_{os.getpid()}')
    log_path = root / 'logs' / f'{session}.log'
    with exclusive_run(root / f'interactive_session_{args.source}'), \
            log_path.open('a', buffering=1) as log:
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
