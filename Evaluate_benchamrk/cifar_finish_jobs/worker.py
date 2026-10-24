"""One SSL seed per Killarney job; continue saved training and finish main probes."""
import argparse
import contextlib
import copy
import csv
import fcntl
import functools
import itertools
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent
sys.path[:0] = [str(BENCH / 'eval_jobs'), str(BENCH)]

import torch

from common import (atomic_json, atomic_save, code_hashes, digest_json,
                    exclusive_run, file_hash, load_encoder)
from data import RawDataset, SyntheticDataset
from evaluate import linear_probe
import interactive_mkdt_2k as mkdt
import interactive_ours as ours
import interactive_random_source as random_eval
from train_ssl import train


def read(path):
    return json.loads(Path(path).read_text())


def inside_bench(value):
    path = (BENCH / value).resolve()
    if not path.is_relative_to(BENCH) or path == BENCH:
        raise ValueError(f'Output/checkpoint path must stay inside Evaluate_benchamrk: {path}')
    return path


def load_task(manifest, task):
    with Path(manifest).open() as stream:
        rows = [row for row in csv.DictReader(stream, delimiter='\t') if row['task'] == task]
    if len(rows) != 1:
        raise ValueError('Unknown or duplicate seed job')
    row = rows[0]
    row['seed'], row['epochs'] = int(row['seed']), int(row['epochs'])
    percentage = float(row['subset'])
    row['subset'] = int(percentage) if percentage.is_integer() else percentage
    if (row['model'] not in mkdt.MODELS or row['ssl'] not in ('simclr', 'barlowtwins')
            or row['subset'] not in (0.2, 1, 2) or row['scope'] != 'main'):
        raise ValueError('Unsupported model/method/size/scope')
    if row['kind'] == 'random':
        expected = {0.2: 10000, 1: 8000, 2: 6000}[row['subset']]
        valid = row['source'] == 'CIFAR100' and 0 <= row['seed'] < 10
    else:
        expected = (3000 if row['subset'] == 2 else 2000) if row['ssl'] == 'simclr' else 10000
        valid = (row['kind'] in ('ours', 'mkdt') and row['source'] == 'CIFAR10'
                 and row['seed'] in (0, 1) and not (row['kind'] == 'mkdt' and row['subset'] == 0.2))
    if not valid or row['epochs'] != expected:
        raise ValueError('Seed job does not match the requested CIFAR table schedule')
    for key in ('checkpoint_dir', 'output_dir'):
        if key == 'output_dir' and row[key] == '-':
            row[key] = row['checkpoint_dir']
        row[key] = inside_bench(row[key])
    row['parent_dir'] = None if row['parent_dir'] == '-' else inside_bench(row['parent_dir'])
    return row


@contextlib.contextmanager
def group_reader(group):
    """Parallel seeds share this lock; existing interactive group writers cannot."""
    group.mkdir(parents=True, exist_ok=True)
    with (group / '.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f'An interactive run owns {group}; stop that run before these seed jobs') from error
        yield


def copy_continuation(parent, folder, config, dataset_count):
    if (folder / 'training.pt').exists() or (folder / 'encoder.pt').exists():
        return
    if not parent.is_dir() or not (parent / 'config.json').is_file():
        print('FRESH SSL: no earlier training checkpoint/config exists:', parent, flush=True)
        return
    with group_reader(parent.parent), exclusive_run(parent):
        checkpoint = parent / 'training.pt'
        if not checkpoint.is_file():
            if (parent / 'encoder.pt').exists() or (parent / 'ssl_complete.json').exists():
                raise FileNotFoundError(f'Cannot continue encoder-only weights: sync {checkpoint}; '
                                        'the projector, optimizer and RNG state are required')
            print('FRESH SSL: no saved training state exists:', parent, flush=True)
            return
        old = read(parent / 'config.json')
        source, target = old['signature'], config['signature']
        if (digest_json(source) != old['identity']['signature']
                or old['identity']['seed'] != config['identity']['seed']
                or old['identity']['selection_method'] != config['identity']['selection_method']
                or any(source[key] != target[key] for key in (
                    'source', 'model', 'subset', 'ssl', 'source_fingerprint', 'synthetic', 'code_hashes'))):
            raise ValueError(f'Continuation checkpoint uses different images/seed/model/code: {parent}')
        source_protocol = copy.deepcopy(source['protocol'])
        source_protocol['ssl']['subset_epochs'] = target['protocol']['ssl']['subset_epochs']
        source_protocol['subset_seeds'] = target['protocol']['subset_seeds']
        if source_protocol != target['protocol']:
            raise ValueError('Continuation may change only the SSL horizon and repeat count')
        checksum = file_hash(checkpoint)
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        batch = min(dataset_count, target['protocol']['ssl']['batch_size'])
        steps_per_epoch = dataset_count // batch
        if (state['identity'] != old['identity']
                or not 0 < state['epoch'] <= source['protocol']['ssl']['subset_epochs']
                or state['epoch'] >= target['protocol']['ssl']['subset_epochs']
                or state['steps_per_epoch'] != steps_per_epoch
                or state['steps'] != state['epoch'] * steps_per_epoch
                or not all(key in state for key in ('model', 'optimizer', 'rng'))):
            raise ValueError('Invalid continuation epoch, step count or full SSL state')
        marker_path = parent / 'ssl_complete.json'
        if (parent / 'encoder.pt').exists() and marker_path.exists():
            if read(marker_path)['epochs'] != state['epoch']:
                raise ValueError('Rolling state is older than the completed parent encoder')
        record = {'source_checkpoint': str(checkpoint), 'source_checkpoint_sha256': checksum,
                  'source_identity': old['identity'], 'source_requested_epochs': source['protocol']['ssl']['subset_epochs'],
                  'resumed_epoch': state['epoch'], 'target_epochs': target['protocol']['ssl']['subset_epochs'],
                  'target_identity': config['identity'],
                  'schedule': 'preserve earlier history; use target-horizon cosine for remaining updates'}
        record_path = folder / 'continuation.json'
        if record_path.exists() and read(record_path) != record:
            raise ValueError('Saved continuation provenance changed')
        if file_hash(checkpoint) != checksum:
            raise ValueError('Parent training state changed during import')
        # Only the destination identity changes. All encoder/projector weights,
        # optimizer momentum, global step and RNG state are copied unchanged.
        state['identity'] = config['identity']
        atomic_json(record_path, record)
        atomic_save(folder / 'training.pt', state)
        print(f'CONTINUE saved epoch {record["resumed_epoch"]} -> {record["target_epochs"]}: {checkpoint}', flush=True)


def synthetic_config(task, args, datasets):
    source, model, percentage = task['source'], task['model'], task['subset']
    if task['kind'] == 'ours':
        path = (BENCH.parent / 'OT-SSL-DD/ours_syntehtic' / source / (model + 'BN')
                / f'pt{percentage}' / f'res_IIC_{source}_{model}BN_{percentage}percent_final.pt')
        loaded = ours.load_synthetic(path, source, model + 'BN', percentage)
        if loaded is None:
            raise FileNotFoundError(f'Missing or unfinished synthetic dataset: {path}')
        synthetic, metadata = loaded
        protocol = ours.expected_protocol('random', source, percentage)
        driver_hash = file_hash(ours.__file__)
    else:
        protocol = mkdt.protocol_for(source, percentage)
        entry = mkdt.synthetic_entry(args.synthetic_root, source, model, percentage)
        synthetic = SyntheticDataset(entry, source, percentage, protocol['input_size'][source])
        metadata, driver_hash = synthetic.metadata, file_hash(mkdt.__file__)
    protocol['ssl'].update(subset_epochs=task['epochs'], checkpoint_every_epochs=500)
    protocol['subset_seeds'] = [0, 1]
    signature = {'protocol': protocol, 'source': source, 'model': model, 'subset': percentage,
                 'ssl': task['ssl'], 'selection_method': task['kind'],
                 'source_fingerprint': datasets(source)[0].fingerprint, 'synthetic': metadata,
                 'ssl_seeds': [0, 1], 'probe_seeds': [0, 1, 2],
                 'code_hashes': code_hashes(), 'driver_sha256': driver_hash}
    if task['kind'] == 'mkdt':
        signature['input_scope'] = 'saved_synthetic_images_only'
    if task['parent_dir'] is not None:
        signature.update(driver_sha256=file_hash(__file__), base_driver_sha256=driver_hash,
                         continuation_policy='continue_full_SSL_state_to_requested_horizon')
    identity = {'signature': digest_json(signature), 'seed': task['seed'], 'selection_method': task['kind']}
    return synthetic, {'signature': signature, 'identity': identity}


def summarize_synthetic(task, config):
    group = task['output_dir'].parent
    with (group / '.summary.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        rows = []
        for labels in (1, 5):
            samples, fingerprints = [], set()
            for ssl_seed, probe_seed in itertools.product((0, 1), (0, 1, 2)):
                folder = group / f'ssl_seed_{ssl_seed:03d}'
                path = folder / 'evaluations' / f'{task["source"]}_lbl{labels}_probe{probe_seed}.json'
                if not path.exists():
                    continue
                result, saved_config, marker = read(path), read(folder / 'config.json'), read(folder / 'ssl_complete.json')
                identity = {'signature': config['identity']['signature'], 'seed': ssl_seed,
                            'selection_method': task['kind']}
                expected = result['expected']
                if (saved_config != {'signature': config['signature'], 'identity': identity}
                        or not marker['complete'] or marker['epochs'] != task['epochs']
                        or marker['identity'] != identity or not result['complete']
                        or not result['encoder_unchanged'] or result['encoder_sha256'] != marker['encoder_sha256']
                        or expected['identity'] != identity or expected['seed'] != probe_seed
                        or expected['source'] != task['source'] or expected['target'] != task['source']
                        or expected['label_percentage'] != labels
                        or expected['evaluation'] != config['signature']['protocol']['evaluation']
                        or expected['evaluation_code_hashes'] != config['signature']['code_hashes']
                        or not math.isfinite(result['accuracy_percent'])
                        or not 0 <= result['accuracy_percent'] <= 100):
                    raise ValueError(f'Incompatible main probe: {path}')
                fingerprints.add((expected['train_fingerprint'], expected['test_fingerprint']))
                if len(fingerprints) != 1:
                    raise ValueError('Main probes use different train/test datasets')
                samples.append({'ssl_seed': ssl_seed, 'probe_seed': probe_seed,
                                'accuracy_percent': result['accuracy_percent']})
            scores = [sample['accuracy_percent'] for sample in samples]
            complete = len(scores) == 6
            rows.append({'source': task['source'], 'model': task['model'], 'method': task['kind'],
                         'subset': task['subset'], 'ssl': task['ssl'], 'target': task['source'], 'labels': labels,
                         'completed_probes': len(scores), 'expected_probes': 6, 'complete': complete,
                         'mean_percent': statistics.mean(scores) if complete else None,
                         'std_percent': statistics.stdev(scores) if complete else None, 'samples': samples})
        atomic_json(group / 'summary.json', {'signature': config['identity']['signature'],
                    'ssl_seeds': [0, 1], 'probe_seeds': [0, 1, 2], 'std_ddof': 1,
                    'complete': all(row['complete'] for row in rows), 'cells': rows})


def run_synthetic(task, args, datasets):
    synthetic, config = synthetic_config(task, args, datasets)
    folder, group = task['checkpoint_dir'], task['checkpoint_dir'].parent
    with group_reader(group), exclusive_run(folder):
        config_path = folder / 'config.json'
        if config_path.exists() and read(config_path) != config:
            raise ValueError(f'Existing training identity/config changed: {folder}')
        atomic_json(config_path, config)
        if task['parent_dir'] is not None:
            copy_continuation(task['parent_dir'], folder, config, len(synthetic))
        protocol, identity = config['signature']['protocol'], config['identity']
        checkpoint, saved = train(synthetic, protocol, task['source'], task['model'], task['ssl'],
                                  task['seed'], folder, identity, workers=0, device='cuda')
        if not saved['complete'] or saved['epochs'] != task['epochs']:
            raise ValueError('SSL did not finish the requested horizon')
        checksum = file_hash(checkpoint)
        marker = {'complete': True, 'identity': identity, 'epochs': saved['epochs'],
                  'encoder_sha256': saved['encoder_sha256'], 'checkpoint_file_sha256': checksum}
        marker_path = folder / 'ssl_complete.json'
        if marker_path.exists() and read(marker_path) != marker:
            raise ValueError('Saved final SSL checkpoint changed')
        atomic_json(marker_path, marker)
        train_set, test_set = datasets(task['source'])
        for labels, probe_seed in itertools.product((1, 5), (0, 1, 2)):
            path = folder / 'evaluations' / f'{task["source"]}_lbl{labels}_probe{probe_seed}.json'
            print('REUSE' if path.exists() else 'PROBE', path, flush=True)
            result = linear_probe(checkpoint, identity, protocol, task['source'], task['source'],
                                  train_set, test_set, labels, probe_seed, path, 0, 'cuda')
            if result['encoder_sha256'] != saved['encoder_sha256'] or not result['encoder_unchanged']:
                raise ValueError('Probe used or changed a different encoder')
            summarize_synthetic(task, config)
        if file_hash(checkpoint) != checksum:
            raise ValueError('Checkpoint changed during evaluation')


def run_random(task, args, datasets):
    root = BENCH / 'random_eval_results_interactive_10seeds'
    checkpoints = BENCH / 'random_ssl_checkpoints_cifar10k8k6k_tiny6k5k4k'
    settings = argparse.Namespace(checkpoints=checkpoints, data_root=args.data_root)
    plan, hashes, drivers, worker_hash, contract = random_eval.evaluation_contract(settings)
    if read(root / 'plan.json') != contract:
        raise ValueError('Random evaluation plan/code differs from the saved runs')
    units = [unit for job in plan['jobs'] for unit in job['units']
             if unit['source'] == task['source'] and unit['model'] == task['model']
             and unit['subset'] == task['subset'] and unit['ssl'] == task['ssl'] and unit['seed'] == task['seed']]
    if len(units) != 1:
        raise ValueError('Cannot identify the original random SSL seed')
    unit = units[0]
    filled = random_eval.filled_random_cells(task['source'])
    pending = [cell for cell in unit['cells'] if cell['target'] == task['source']
               and random_eval.cell_key(cell) not in filled]
    if not pending:
        print('SKIP: requested Random main-table cells are already filled.', flush=True)
        return
    folder = checkpoints / unit['relative']
    if not (folder / 'encoder.pt').is_file():
        subprocess.run(['bash', str(BENCH / 'random_ssl.sh'), '--worker', str(BENCH),
                        task['source'], task['model'], str(task['subset']), task['ssl'],
                        str(args.venv), str(args.data_root), str(checkpoints),
                        str(task['seed']), str(task['epochs']), '1'], check=True)
    # Recover a crash between the final encoder write and its completion marker.
    with exclusive_run(folder):
        config = read(folder / 'config.json')
        encoder, _, saved = load_encoder(folder / 'encoder.pt', config['identity'], 'cpu')
        del encoder
        if saved['epochs'] != task['epochs']:
            raise ValueError('Random encoder epoch budget differs')
        marker_path = folder / 'ssl_complete.json'
        if not marker_path.exists():
            atomic_json(marker_path, {**{k: v for k, v in saved.items() if k != 'encoder'},
                        'checkpoint_file_sha256': file_hash(folder / 'encoder.pt')})
    random_eval.evaluate_pending_seed(unit, pending, plan, root, datasets, hashes, drivers, worker_hash)


def write_report(manifest):
    directory = HERE / 'reports'
    directory.mkdir(exist_ok=True)
    with (directory / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with Path(manifest).open() as stream:
            tasks = list(csv.DictReader(stream, delimiter='\t'))
        rows, seen = [], set()
        for task in tasks:
            if task['kind'] == 'random':
                continue
            path = inside_bench(task['checkpoint_dir'] if task['output_dir'] == '-' else task['output_dir']).parent / 'summary.json'
            if path in seen or not path.exists():
                continue
            seen.add(path)
            rows.extend({**row, 'method': task['kind'], 'model': task['model']}
                        for row in read(path)['cells'] if row['source'] == row['target'])
        ours.write_tsv(directory / 'cifar10_main_summary.tsv', [
            'source', 'model', 'method', 'subset', 'ssl', 'target', 'labels',
            'completed_probes', 'expected_probes', 'complete', 'mean_percent', 'std_percent'], rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task')
    parser.add_argument('--manifest', type=Path, default=HERE / 'tasks.tsv')
    parser.add_argument('--data-root', type=Path, default=BENCH.parent.parent / 'data')
    parser.add_argument('--synthetic-root', type=Path, default=BENCH.parent / 'benchmarks/mkdt_multidataset/results/'
                        'batch_20261010T160046Z_Y6dqZS/synthetic/generated')
    parser.add_argument('--venv', type=Path, required=True)
    args = parser.parse_args()
    if (not os.environ.get('SLURM_JOB_ID')
            or os.environ.get('SLURM_CLUSTER_NAME', os.environ.get('CC_CLUSTER')) != 'killarney'
            or not torch.cuda.is_available()):
        parser.error('Run in the submitted Killarney GPU job')
    if os.environ.get('SLURM_JOB_ACCOUNT', 'aip-yiweilu') != 'aip-yiweilu':
        parser.error('These jobs must use the Lu account: aip-yiweilu')
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        parser.error('Use one process and one GPU per seed job')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    args.data_root, args.synthetic_root, args.venv = (path.resolve() for path in
                                                    (args.data_root, args.synthetic_root, args.venv))
    task = load_task(args.manifest, args.task)

    @functools.lru_cache(maxsize=1)
    def datasets(source):
        return RawDataset(args.data_root, source, True), RawDataset(args.data_root, source, False)

    print('ONE SEED / MAIN ONLY:', {k: task[k] for k in ('kind', 'source', 'model', 'subset', 'ssl', 'seed', 'epochs')}, flush=True)
    with exclusive_run(HERE / 'runtime' / task['task']):
        try:
            if task['kind'] == 'random':
                run_random(task, args, datasets)
            else:
                run_synthetic(task, args, datasets)
            atomic_json(HERE / 'runtime' / task['task'] / 'complete.json', {
                'complete': True, 'task': task['task'], 'worker_sha256': file_hash(__file__),
                'epochs': task['epochs'], 'checkpoint_dir': str(task['checkpoint_dir']),
                'output_dir': str(task['output_dir'])})
        finally:
            write_report(args.manifest)


if __name__ == '__main__':
    main()
