"""TinyImageNet full-data evaluation with the explicitly accepted timeout rules.

Reuse the original driver and its completed probes unchanged. Record this extension
in plan_tiny_final.json; original plan.json remains the contract for the reused runs.
VGG11/SimCLR uses two encoders with two probe seeds each, never four SSL seeds.
"""
import argparse
import contextlib
from datetime import datetime, timezone
import functools
import gc
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
import traceback

import interactive_full as full
from common import atomic_json, code_hashes, exclusive_run, file_hash
from data import RawDataset
from evaluate import linear_probe
from report import report as base_report, write_tsv


BENCH = full.BENCH
VGG_GROUP = 'TinyImageNet/VGG11/full/pt100/simclr'
VGG_SEEDS = (0, 2)
ACCEPTED_TINY = {
    ('TinyImageNet', model, ssl, seed): {
        'accepted_epoch': 950, 'requested_epoch': 1000,
        'reason': 'user_accepted_TinyImageNet_timeout_checkpoint_as_final',
    }
    for model, ssl, seed in (
        ('ConvNet', 'barlowtwins', 0), ('ConvNet', 'barlowtwins', 2),
        ('ResNet18', 'barlowtwins', 0), ('ResNet18', 'barlowtwins', 2),
        ('ResNet18', 'simclr', 0), ('ResNet18', 'simclr', 2),
        ('VGG11', 'barlowtwins', 1), ('VGG11', 'simclr', 2))
}
# Only this TinyImageNet entry point installs these user-approved exceptions.
full.ACCEPTED_ROLLING.update(ACCEPTED_TINY)


def repeat_seed(encoder_seed):
    return 10000 + encoder_seed


def build_plan(checkpoints, data_root):
    plan = full.build_plan('TinyImageNet', checkpoints, data_root)
    job = plan['jobs'][0]
    job['units'] = [u for u in job['units']
                    if u['group'] != VGG_GROUP or u['seed'] in VGG_SEEDS]
    job['probes'] = sum(len(u['cells']) * (2 if u['group'] == VGG_GROUP else 1)
                        for u in job['units'])
    if len(job['units']) != 17 or job['probes'] != 110:
        raise ValueError('Expected 17 encoders and 110 probes for the accepted TinyImageNet plan')
    return plan


def repeat_marker(base, probe_seed, extension_hashes):
    return {**{key: base[key] for key in (
        'identity', 'encoder_sha256', 'checkpoint_file_sha256', 'checkpoint',
        'evaluations', 'evaluation_code_hashes', 'driver_hashes', 'ssl_epochs',
        'accepted_rolling_checkpoint')},
        'complete': True, 'probe_repeat': 1, 'probe_seed': probe_seed,
        'encoder_seed': base['identity']['seed'],
        'extension_driver_hashes': extension_hashes}


def evaluate_repeat(unit, root, datasets, extension_hashes):
    output = root / unit['relative']
    with exclusive_run(output):
        base = full.read(output / 'complete.json')
        config = full.read(output / 'config.json')
        identity, checkpoint = config['identity'], Path(base['checkpoint'])
        expected = repeat_marker(base, repeat_seed(unit['seed']), extension_hashes)
        folder = output / 'probe_repeat_001'
        prior = full.read(folder / 'complete.json') if (folder / 'complete.json').exists() else None
        if prior and any(prior.get(k) != v for k, v in expected.items()):
            raise ValueError('Repeated probe provenance differs from this run')
        if file_hash(checkpoint) != base['checkpoint_file_sha256']:
            raise ValueError('Checkpoint changed before repeated probing')
        result_hashes = {}
        for cell in unit['cells']:
            target, labels = cell['target'], int(cell['labels'])
            path = folder / 'evaluations' / f'{target}_lbl{labels}.json'
            if prior and (not path.exists() or file_hash(path) != prior['result_hashes'][path.name]):
                raise ValueError(f'Repeated probe result was changed: {path}')
            train_set, test_set = datasets(target)
            result = linear_probe(checkpoint, identity, config['signature']['protocol'],
                unit['source'], target, train_set, test_set, labels,
                repeat_seed(unit['seed']), path, 0, 'cuda')
            if (not result.get('complete') or not result['encoder_unchanged']
                    or result['encoder_sha256'] != base['encoder_sha256']):
                raise ValueError('Repeated probe used or changed a different encoder')
            result_hashes[path.name] = file_hash(path)
        if file_hash(checkpoint) != base['checkpoint_file_sha256']:
            raise ValueError('Checkpoint changed during repeated probing')
        atomic_json(folder / 'complete.json', {**expected, 'result_hashes': result_hashes})


def repeated_scores(unit, root, hashes, driver_hashes, extension_hashes):
    """Validate repeat provenance before allowing any repeated score into a cell."""
    output = root / unit['relative']
    folder = output / 'probe_repeat_001'
    if not (folder / 'complete.json').exists():
        return {}
    base = full.read(output / 'complete.json')
    marker = full.read(folder / 'complete.json')
    expected_marker = repeat_marker(base, repeat_seed(unit['seed']), extension_hashes)
    if (any(marker.get(k) != v for k, v in expected_marker.items())
            or base['evaluation_code_hashes'] != hashes or base['driver_hashes'] != driver_hashes):
        raise ValueError('Invalid repeated probe completion marker')
    config = full.read(output / 'config.json')
    scores = {}
    for cell in unit['cells']:
        name = f"{cell['target']}_lbl{cell['labels']}.json"
        path = folder / 'evaluations' / name
        row, original = full.read(path), full.read(output / 'evaluations' / name)
        wanted = {**original['expected'], 'seed': repeat_seed(unit['seed'])}
        # The independent probe seed also selects an independent labeled subset.
        wanted['labeled_indices'] = row['expected']['labeled_indices']
        score = row['accuracy_percent']
        if (not row.get('complete') or not row['encoder_unchanged']
                or row['expected'] != wanted
                or row['encoder_sha256'] != base['encoder_sha256']
                or row['expected']['identity'] != config['identity']
                or file_hash(path) != marker['result_hashes'][name]
                or not math.isfinite(score) or not 0 <= score <= 100):
            raise ValueError(f'Invalid repeated probe result: {path}')
        scores[(cell['target'], cell['labels'])] = row
    return scores


def report(root, plan, hashes, driver_hashes, extension_hashes):
    status = base_report(root, plan, hashes, driver_hashes)
    summaries = full.rows(root / 'table_summary.tsv')
    checklist = full.rows(root / 'checklist.tsv')
    timings = full.rows(root / 'probe_times.tsv')
    for row in checklist + timings:
        row.update(encoder_seed=row['seed'], probe_repeat=0, probe_seed=row['seed'])
    group_summary = full.read(root / VGG_GROUP / 'summary.json')
    extra_probes = 0
    validated = {}
    for unit in plan['jobs'][0]['units']:
        if unit['group'] != VGG_GROUP:
            continue
        # Primary runs were already checked by the common report, including hashes.
        primary_valid = all(unit['seed'] in row['seeds'] for row in group_summary['results'])
        scores, state = {}, 'pending'
        if primary_valid:
            try:
                scores = repeated_scores(unit, root, hashes, driver_hashes, extension_hashes)
                state = 'complete' if scores else 'pending'
            except Exception as error:
                state = 'invalid'
                status['invalid_encoders'].append({'encoder': unit['relative'] + '/probe_repeat_001',
                                                   'error': str(error)})
        validated[unit['seed']] = scores
        extra_probes += len(scores)
        for cell in unit['cells']:
            result = scores.get((cell['target'], cell['labels']))
            entry = {**cell, 'task': 'TinyImageNet', 'seed': unit['seed'],
                     'encoder_seed': unit['seed'], 'probe_repeat': 1,
                     'probe_seed': repeat_seed(unit['seed'])}
            checklist.append({**entry, 'status': state,
                'accuracy_percent': '' if result is None else result['accuracy_percent'],
                'result': str(root / unit['relative'] / 'probe_repeat_001' / 'evaluations'
                              / f"{cell['target']}_lbl{cell['labels']}.json")})
            if result is not None:
                timings.append({**entry, 'seconds': result['seconds']})
    for row in group_summary['results']:
        samples = [{'encoder_seed': seed, 'probe_repeat': 0, 'probe_seed': seed,
                    'accuracy_percent': score}
                   for seed, score in zip(row['seeds'], row['accuracies_percent'])]
        for seed in VGG_SEEDS:
            repeated = validated[seed].get((row['target'], row['labels']))
            if repeated is not None:
                samples.append({'encoder_seed': seed, 'probe_repeat': 1,
                    'probe_seed': repeat_seed(seed), 'accuracy_percent': repeated['accuracy_percent']})
        complete = {(r['encoder_seed'], r['probe_repeat']) for r in samples} == {
            (seed, repeat) for seed in VGG_SEEDS for repeat in (0, 1)}
        row.update(complete=complete, expected_seeds=2, completed_probe_runs=len(samples),
                   expected_probe_runs=4, probe_repeats_per_encoder=2,
                   probe_runs=samples, mean_percent='', std_percent='', latex='')
        if complete:
            values = [r['accuracy_percent'] for r in samples]
            mean, std = statistics.mean(values), statistics.stdev(values)
            row.update(mean_percent=mean, std_percent=std, latex=f'\\dsacc{{{mean:.2f}}}{{{std:.2f}}}')
        for summary in summaries:
            if (summary['model'], summary['ssl'], summary['target'], summary['labels']) == (
                    'VGG11', 'simclr', row['target'], row['labels']):
                summary.update({k: row[k] for k in ('complete', 'expected_seeds', 'completed_seeds',
                    'mean_percent', 'std_percent', 'latex', 'completed_probe_runs',
                    'expected_probe_runs', 'probe_repeats_per_encoder')})
    group_summary.update(complete=all(r['complete'] for r in group_summary['results']),
        encoder_seeds=list(VGG_SEEDS), probe_repeats_per_encoder=2,
        aggregation='mean and sample SD across four probes from two SSL encoders',
        extension_driver_hashes=extension_hashes)
    atomic_json(root / VGG_GROUP / 'summary.json', group_summary)
    write_tsv(root / 'table_summary.tsv', list(summaries[0]) + [
        'completed_probe_runs', 'expected_probe_runs', 'probe_repeats_per_encoder'], summaries)
    write_tsv(root / 'checklist.tsv', list(checklist[0]), checklist)
    write_tsv(root / 'probe_times.tsv', list(timings[0]) if timings else [
        'task', 'source', 'model', 'method', 'subset', 'ssl', 'seed', 'target', 'labels',
        'seconds', 'encoder_seed', 'probe_repeat', 'probe_seed'], timings)
    status.update(completed_probes=status['completed_probes'] + extra_probes,
        completed_table_cells=sum(str(r['complete']) == 'True' for r in summaries),
        excluded_encoder='TinyImageNet/VGG11/full/pt100/simclr/seed_001',
        vgg11_simclr_encoder_seeds=list(VGG_SEEDS), vgg11_simclr_probe_runs_per_cell=4)
    status['complete'] = (status['completed_encoders'] == status['expected_encoders']
                          and status['completed_probes'] == status['expected_probes']
                          and status['completed_table_cells'] == status['expected_table_cells']
                          and not status['invalid_encoders'])
    atomic_json(root / 'progress.json', status)
    return status


def run(args, root, session):
    if not full.torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; use your allocated GPU node')
    full.torch.set_num_threads(1)
    full.torch.set_num_interop_threads(1)
    full.torch.backends.cuda.matmul.allow_tf32 = False
    full.torch.backends.cudnn.allow_tf32 = False
    hashes = code_hashes()
    worker_hash = file_hash(BENCH / 'full_ssl_jobs' / 'worker.py')
    driver_hashes = {str(p.relative_to(BENCH)): file_hash(p) for p in (
        full.HERE / 'interactive_full.py', full.HERE / 'report.py',
        BENCH / 'full_ssl_jobs' / 'worker.py', BENCH / 'full_ssl_jobs' / 'tasks.tsv',
        BENCH / 'table_coverage.tsv', BENCH / 'protocol.json')}
    legacy = {'plan': full.build_plan('TinyImageNet', args.checkpoints, args.data_root),
              'code_hashes': hashes, 'driver_hashes': driver_hashes,
              'ssl_worker_sha256': worker_hash}
    if (root / 'plan.json').exists() and full.read(root / 'plan.json') != legacy:
        raise ValueError('Original evaluation inputs/code changed; cannot reuse existing probes')
    plan = build_plan(args.checkpoints, args.data_root)
    extension_hashes = {'eval_jobs/interactive_full_tiny.py': file_hash(__file__)}
    contract = {'base_contract': legacy, 'plan': plan,
        'accepted_rolling': [{'source': s, 'model': m, 'ssl': ssl, 'seed': seed, **rule}
                            for (s, m, ssl, seed), rule in ACCEPTED_TINY.items()],
        'repeated_probe_seeds': {str(s): [s, repeat_seed(s)] for s in VGG_SEEDS},
        'extension_driver_hashes': extension_hashes}
    path = root / 'plan_tiny_final.json'
    if path.exists() and full.read(path) != contract:
        raise ValueError('TinyImageNet final-evaluation contract changed; use another --output')
    atomic_json(root / 'plan.json', legacy)
    atomic_json(path, contract)

    @functools.lru_cache(maxsize=7)
    def datasets(target):
        return RawDataset(args.data_root, target, True), RawDataset(args.data_root, target, False)

    counts = {'evaluated_encoders': 0, 'already_complete': 0, 'missing_checkpoint': 0}
    failures, interrupted, started = [], False, time.monotonic()
    print('TinyImageNet full data: 17 encoders, 110 probes, 36 table cells.', flush=True)
    print('Accepted timeout snapshots: epoch 950. Excluded: VGG11/SimCLR/seed_001.', flush=True)
    print('VGG11/SimCLR: encoder seeds 0 and 2, two probe seeds each; four runs per label fraction.', flush=True)
    print('100-epoch frozen linear probes at 1%/5% labels; downstream on ResNet18 only.', flush=True)
    print('Completed probes are validated and reused; missing checkpoints are skipped.', flush=True)
    try:
        for unit in plan['jobs'][0]['units']:
            folder = args.checkpoints / unit['relative']
            final = all((folder / name).is_file() and (folder / name).stat().st_size
                        for name in ('config.json', 'encoder.pt', 'ssl_complete.json'))
            rolling = full.accepted_rolling(unit) and all(
                (folder / name).is_file() and (folder / name).stat().st_size
                for name in ('config.json', 'training.pt'))
            if not (final or rolling):
                counts['missing_checkpoint'] += 1
                print('SKIP missing checkpoint:', unit['relative'], flush=True)
                continue
            if not final:
                print('USE accepted rolling checkpoint:', unit['relative'], 'epoch 950/1000', flush=True)
            output = root / unit['relative']
            done = (output / 'complete.json').exists() and (unit['group'] != VGG_GROUP or
                       (output / 'probe_repeat_001' / 'complete.json').exists())
            try:
                full.evaluate_seed(unit, plan, root, datasets, hashes, driver_hashes, worker_hash)
                if unit['group'] == VGG_GROUP:
                    print('CHECK second probe run:', unit['relative'],
                          'probe seed', repeat_seed(unit['seed']), flush=True)
                    evaluate_repeat(unit, root, datasets, extension_hashes)
                counts['already_complete' if done else 'evaluated_encoders'] += 1
                print('SKIP already evaluated:' if done else 'DONE:', unit['relative'], flush=True)
            except Exception as error:
                failures.append({'encoder': unit['relative'], 'error': str(error)})
                print('ERROR:', unit['relative'], flush=True)
                traceback.print_exc()
            finally:
                gc.collect()
                full.torch.cuda.empty_cache()
            report(root, plan, hashes, driver_hashes, extension_hashes)
    except KeyboardInterrupt:
        interrupted = True
        print('Stopped; saved probes will be reused on rerun.', flush=True)
    finally:
        progress = report(root, plan, hashes, driver_hashes, extension_hashes)
        atomic_json(root / 'logs' / f'{session}.json', {**counts, 'failures': failures,
            'interrupted': interrupted, 'seconds': time.monotonic() - started,
            'extension_driver_hashes': extension_hashes, 'progress': progress})
        print('SESSION:', json.dumps({**counts, 'errors': len(failures), 'interrupted': interrupted}), flush=True)
        print('COVERAGE:', json.dumps(progress), flush=True)
        print('Results:', root, flush=True)
    return 130 if interrupted else int(bool(failures or progress['invalid_encoders']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=BENCH.parent.parent / 'data')
    parser.add_argument('--checkpoints', type=Path,
                        default=BENCH / 'full_ssl_checkpoints_1000ep_3seeds')
    parser.add_argument('--output', type=Path,
                        default=BENCH / 'full_eval_results_interactive_3seeds_TinyImageNet')
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get(
            'SLURM_CLUSTER_NAME', os.environ.get('CC_CLUSTER')) != 'vulcan':
        parser.error('Run this on your allocated Vulcan GPU node')
    args.data_root, args.checkpoints, root = (p.resolve() for p in (
        args.data_root, args.checkpoints, args.output))
    if (not root.is_relative_to(BENCH) or root == BENCH or root.is_relative_to(args.checkpoints)
            or args.checkpoints.is_relative_to(root)):
        parser.error('Use a separate evaluation directory inside Evaluate_benchamrk')
    if not args.checkpoints.is_dir() or not args.data_root.is_dir():
        parser.error('Missing data or checkpoint root')
    (root / 'logs').mkdir(parents=True, exist_ok=True)
    session = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + f'_tiny_final_{os.getpid()}'
    log_path = root / 'logs' / f'{session}.log'
    with exclusive_run(root / 'interactive_session'), log_path.open('a', buffering=1) as log:
        with contextlib.redirect_stdout(full.Tee(sys.stdout, log)), \
                contextlib.redirect_stderr(full.Tee(sys.stderr, log)):
            print('Session log:', log_path, flush=True)
            try:
                return run(args, root, session)
            except Exception:
                traceback.print_exc()
                return 1


if __name__ == '__main__':
    raise SystemExit(main())
