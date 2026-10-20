"""Evaluate ready six-seed MKDT encoders inside an existing Killarney allocation.

With --source, evaluate one source dataset. Without it, evaluate all three.
Missing checkpoints are skipped; valid completed probes are reused.
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

from common import atomic_json, code_hashes, exclusive_run, file_hash
from data import RawDataset
from mkdt import SOURCES, build_plan, evaluate_seed
from report import report


class Tee:
    def __init__(self, terminal, log):
        self.terminal, self.log = terminal, log

    def write(self, text):
        self.terminal.write(text)
        self.log.write(text)
        self.flush()
        return len(text)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def evaluation_contract(args):
    plan = build_plan(args.checkpoints, args.data_root)
    hashes = code_hashes()
    paths = (
        HERE / 'interactive_mkdt.py', HERE / 'mkdt.py', HERE / 'worker.py',
        HERE / 'report.py', BENCH / 'mkdt_ssl_jobs/worker.py',
        BENCH / 'table_coverage.tsv', BENCH / 'protocol.json')
    driver_hashes = {str(path.relative_to(BENCH)): file_hash(path) for path in paths}
    worker_hash = file_hash(BENCH / 'mkdt_ssl_jobs/worker.py')
    contract = {
        'plan': plan, 'code_hashes': hashes, 'driver_hashes': driver_hashes,
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
        (unit for job in plan['jobs'] for unit in job['units']
         if args.source is None or unit['source'] == args.source),
        key=lambda unit: (
            unit['source'], unit['model'], unit['subset'], unit['ssl'], unit['seed']))
    expected = 216 if args.source is None else 72
    if len(units) != expected:
        raise ValueError(f'Expected {expected} MKDT encoders, found {len(units)}')

    scope = args.source or 'all datasets'
    counts = {'evaluated_encoders': 0, 'already_complete': 0, 'missing_checkpoint': 0}
    failures = []
    interrupted = False
    started = time.monotonic()
    print(f'Sequential MKDT evaluation for {scope}: {expected} possible encoders; '
          'missing inputs are skipped.', flush=True)
    print('Valid MKDT sizes: 1% and 2%; six SSL seeds per cell.', flush=True)
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
            'progress': progress})
        print('SESSION:', json.dumps({
            'source': args.source, **counts, 'errors': len(failures),
            'interrupted': interrupted}), flush=True)
        print('COVERAGE:', json.dumps(progress), flush=True)
        print('Results:', root, flush=True)
        print('Re-run to pick up checkpoints that finished after they were skipped. '
              'No automatic waiting.', flush=True)
    return 130 if interrupted else (1 if failures or progress['invalid_encoders'] else 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', choices=SOURCES)
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

    scope = args.source or 'all'
    session = (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
               + f'_{scope}_{os.getpid()}')
    log_path = root / 'logs' / f'{session}.log'
    with exclusive_run(root / f'interactive_session_{scope}'), \
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
