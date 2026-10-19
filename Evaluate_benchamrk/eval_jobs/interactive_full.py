"""Evaluate ready full-data encoders for one source inside an existing GPU allocation.

Missing checkpoints are skipped. Valid completed probes are reused on every rerun.
This driver never trains SSL, submits jobs, waits for checkpoints, or deletes files.
"""
import argparse
import contextlib
import csv
from datetime import datetime, timezone
import functools
import gc
import itertools
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

from common import (atomic_json, atomic_save, code_hashes, digest_json,
                    exclusive_run, file_hash, load_encoder, state_hash)
from data import RawDataset, SOURCE_COUNTS
from evaluate import linear_probe
from report import report
from run import evaluation_targets, validate_protocol
from train_ssl import SSLModel


SOURCES = ('CIFAR10', 'CIFAR100', 'TinyImageNet')
MODELS = ('ConvNet', 'VGG11', 'ResNet18')
METHODS = ('simclr', 'barlowtwins')
SEEDS = (0, 1, 2)
ACCEPTED_ROLLING = {
    ('CIFAR10', 'ResNet18', 'barlowtwins', 1): {
        'accepted_epoch': 950,
        'requested_epoch': 1000,
        'reason': 'user_accepted_timeout_after_epoch_999_log',
    },
}


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    with Path(path).open() as stream:
        return list(csv.DictReader(stream, delimiter='\t'))


def expected_protocol():
    protocol = read(BENCH / 'protocol.json')
    # These are the per-launcher overrides used by full_ssl_jobs/worker.py.
    protocol['full_seeds'] = list(SEEDS)
    protocol['ssl']['checkpoint_every_epochs'] = 50
    validate_protocol(protocol)
    if (protocol['ssl']['full_epochs'] != 1000
            or protocol['ssl']['batch_size'] != 256
            or protocol['evaluation']['epochs'] != 100):
        raise ValueError('Unexpected full-data SSL or linear-probe protocol')
    return protocol


def accepted_rolling(unit):
    return ACCEPTED_ROLLING.get((
        unit['source'], unit['model'], unit['ssl'], unit['seed']))


def build_plan(source, checkpoint_root, data_root):
    cells = [row for row in rows(BENCH / 'table_coverage.tsv')
             if row['source'] == source and row['method'] == 'full'
             and row['disabled'] == 'False']
    groups = {}
    for cell in cells:
        key = f"{source}/{cell['model']}/full/pt100/{cell['ssl']}"
        group = groups.setdefault(key, {'group': key, 'source': source,
            'model': cell['model'], 'subset': 100, 'ssl': cell['ssl'], 'cells': []})
        group['cells'].append(cell)
    expected_groups = {f'{source}/{model}/full/pt100/{method}'
                       for model, method in itertools.product(MODELS, METHODS)}
    if set(groups) != expected_groups or len(cells) != 36:
        raise ValueError(f'Full-data table coverage is incomplete for {source}')
    for group in groups.values():
        actual = [(cell['target'], int(cell['labels'])) for cell in group['cells']]
        wanted = set(itertools.product(evaluation_targets(source, group['model']), (1, 5)))
        if len(actual) != len(set(actual)) or set(actual) != wanted:
            raise ValueError(f'Missing or duplicate full-data table cell: {group["group"]}')
        group['cells'].sort(key=lambda cell: (
            cell['target'] != source, cell['target'], int(cell['labels'])))
    units = []
    for key in sorted(groups):
        group = groups[key]
        for seed in SEEDS:
            units.append({**group, 'seed': seed,
                'relative': f'{key}/seed_{seed:03d}'})
    job = {'task': source, 'units': units,
           'probes': sum(len(unit['cells']) for unit in units)}
    if len(units) != 18 or job['probes'] != 108:
        raise ValueError('Expected 18 encoders and 108 probes per source')
    return {'method': 'full', 'source': source, 'seeds': list(SEEDS),
            'checkpoint_root': str(Path(checkpoint_root).resolve()),
            'data_root': str(Path(data_root).resolve()),
            'jobs': [job], 'groups': list(groups.values())}


def validate_config(folder, unit, protocol, hashes, worker_hash):
    config = read(folder / 'config.json')
    signature, identity = config['signature'], config['identity']
    if (digest_json(signature) != identity['signature']
            or signature['protocol'] != protocol
            or signature['code_hashes'] != hashes
            or signature['ssl_worker_sha256'] != worker_hash
            or signature.get('ssl_worker') != 'full_ssl_jobs/worker.py'
            or signature['selection_method'] != 'full'
            or identity['selection_method'] != 'full'
            or identity['selection'] != {'scope': 'full'}
            or identity['seed'] != unit['seed']
            or signature['source'] != unit['source']
            or signature['model'] != unit['model']
            or signature['ssl'] != unit['ssl']
            or signature['subset'] != 100
            or signature['input_scope'] != 'entire_source_training_split'
            or signature['image_count'] != SOURCE_COUNTS[unit['source']]):
        raise ValueError(f'Wrong full-data checkpoint recipe, code, or seed: {folder}')
    return config


def validate_checkpoint(folder, unit, protocol, hashes, worker_hash):
    config = validate_config(folder, unit, protocol, hashes, worker_hash)
    identity = config['identity']
    checkpoint = folder / 'encoder.pt'
    marker = read(folder / 'ssl_complete.json')
    checksum = file_hash(checkpoint)
    if (not marker.get('complete') or marker['identity'] != identity
            or marker['epochs'] != 1000
            or marker['checkpoint_file_sha256'] != checksum):
        raise ValueError(f'Incomplete or changed full-data checkpoint: {folder}')
    encoder, _, saved = load_encoder(checkpoint, identity, 'cpu')
    del encoder
    if (not saved['complete'] or saved['identity'] != identity
            or saved['epochs'] != 1000
            or saved['encoder_sha256'] != marker['encoder_sha256']
            or saved['source'] != unit['source']
            or saved['model'] != unit['model']
            or saved['method'] != unit['ssl']
            or saved['seed'] != unit['seed']
            or saved['effective_batch'] != 256
            or saved['steps'] != 1000 * (SOURCE_COUNTS[unit['source']] // 256)
            or saved['input_size'] != protocol['input_size'][unit['source']]
            or saved['normalization'] != protocol['normalization'][unit['source']]):
        raise ValueError(f'Full-data encoder metadata mismatch: {folder}')
    return config, saved, checksum, checkpoint, None


def derive_accepted_encoder(folder, output, unit, protocol, hashes, worker_hash):
    """Materialize the one explicitly accepted rolling encoder for probing only."""
    acceptance = accepted_rolling(unit)
    if not acceptance:
        raise FileNotFoundError(f'No accepted rolling checkpoint exception: {folder}')
    config = validate_config(folder, unit, protocol, hashes, worker_hash)
    identity = config['identity']
    rolling_path = folder / 'training.pt'
    rolling_hash = file_hash(rolling_path)
    derived = output / f"accepted_encoder_epoch{acceptance['accepted_epoch']}.pt"
    record_path = output / 'accepted_rolling_checkpoint.json'
    expected_record = {**acceptance, 'source_checkpoint': str(rolling_path),
        'source_checkpoint_sha256': rolling_hash,
        'derived_checkpoint': str(derived)}
    if derived.exists():
        if record_path.exists() and read(record_path) != expected_record:
            raise ValueError('Accepted rolling-checkpoint provenance changed')
        encoder, _, saved = load_encoder(derived, identity, 'cpu')
        del encoder
    else:
        rolling = torch.load(rolling_path, map_location='cpu', weights_only=True)
        steps_per_epoch = SOURCE_COUNTS[unit['source']] // protocol['ssl']['batch_size']
        if (rolling['identity'] != identity
                or rolling['epoch'] != acceptance['accepted_epoch']
                or rolling['steps_per_epoch'] != steps_per_epoch
                or rolling['steps'] != acceptance['accepted_epoch'] * steps_per_epoch):
            raise ValueError('Accepted rolling checkpoint is not the recorded epoch-950 state')
        model = SSLModel(unit['model'], protocol['input_size'][unit['source']],
                         protocol['methods'][unit['ssl']])
        model.load_state_dict(rolling['model'], strict=True)
        encoder = model.encoder.cpu()
        saved = {'kind': 'ssl_reference_encoder', 'complete': True,
            'identity': identity, 'model': unit['model'],
            'input_size': protocol['input_size'][unit['source']],
            'source': unit['source'], 'method': unit['ssl'], 'seed': unit['seed'],
            'epochs': acceptance['accepted_epoch'],
            'requested_epochs': acceptance['requested_epoch'],
            'steps': rolling['steps'], 'effective_batch': protocol['ssl']['batch_size'],
            'accepted_partial': True, 'acceptance_reason': acceptance['reason'],
            'encoder': encoder.state_dict(), 'encoder_sha256': state_hash(encoder),
            'normalization': protocol['normalization'][unit['source']]}
        atomic_save(derived, saved)
        checked, _, checked_saved = load_encoder(derived, identity, 'cpu')
        del checked
        if checked_saved['encoder_sha256'] != saved['encoder_sha256']:
            raise ValueError('Derived accepted encoder failed strict reload')
    if (not saved.get('accepted_partial')
            or saved['epochs'] != acceptance['accepted_epoch']
            or saved['requested_epochs'] != acceptance['requested_epoch']
            or saved['acceptance_reason'] != acceptance['reason']):
        raise ValueError('Derived encoder lacks the accepted partial-run metadata')
    atomic_json(record_path, expected_record)
    return config, saved, file_hash(derived), derived, expected_record


def evaluate_seed(unit, plan, root, datasets, hashes, driver_hashes, worker_hash):
    source = Path(plan['checkpoint_root']) / unit['relative']
    output = root / unit['relative']
    protocol = expected_protocol()
    with exclusive_run(output), exclusive_run(source):
        if ((source / 'encoder.pt').is_file()
                and (source / 'ssl_complete.json').is_file()):
            config, saved, checksum, checkpoint, acceptance = validate_checkpoint(
                source, unit, protocol, hashes, worker_hash)
        else:
            config, saved, checksum, checkpoint, acceptance = derive_accepted_encoder(
                source, output, unit, protocol, hashes, worker_hash)
        if datasets(unit['source'])[0].fingerprint != config['signature']['source_fingerprint']:
            raise ValueError('Source training split differs from the full-data SSL split')
        identity = config['identity']
        if (output / 'config.json').exists() and read(output / 'config.json') != config:
            raise ValueError('Refusing to mix checkpoint configurations in one output directory')
        atomic_json(output / 'config.json', config)
        requested = [[cell['target'], int(cell['labels'])] for cell in unit['cells']]
        marker = {'complete': True, 'identity': identity,
            'encoder_sha256': saved['encoder_sha256'],
            'checkpoint_file_sha256': checksum, 'checkpoint_retained': True,
            'checkpoint': str(checkpoint), 'evaluations': requested,
            'evaluation_code_hashes': hashes, 'driver_hashes': driver_hashes,
            'ssl_epochs': saved['epochs'], 'accepted_rolling_checkpoint': acceptance}
        prior = read(output / 'complete.json') if (output / 'complete.json').exists() else None
        if prior and any(prior.get(key) != value for key, value in marker.items()):
            raise ValueError('Completed full-data evaluation provenance differs from this run')
        result_hashes = {}
        for target, label in requested:
            path = output / 'evaluations' / f'{target}_lbl{label}.json'
            if prior and (not path.exists()
                          or file_hash(path) != prior['result_hashes'][path.name]):
                raise ValueError(f'Completed result was changed or removed: {path}')
            train_set, test_set = datasets(target)
            result = linear_probe(checkpoint, identity, protocol,
                unit['source'], target, train_set, test_set, label,
                unit['seed'], path, 0, 'cuda')
            if (not result.get('complete') or not result['encoder_unchanged']
                    or result['encoder_sha256'] != saved['encoder_sha256']):
                raise ValueError(f'Probe changed or used a different encoder: {path}')
            result_hashes[path.name] = file_hash(path)
        if file_hash(checkpoint) != checksum:
            raise ValueError('Full-data evaluation checkpoint changed during evaluation')
        atomic_json(output / 'complete.json', {**marker, 'result_hashes': result_hashes})
    print('ENCODER EVALUATED', unit['relative'], flush=True)


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


def run(args, root, session):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; run this on your allocated GPU node')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    plan = build_plan(args.source, args.checkpoints, args.data_root)
    hashes = code_hashes()
    worker_hash = file_hash(BENCH / 'full_ssl_jobs' / 'worker.py')
    driver_hashes = {str(path.relative_to(BENCH)): file_hash(path) for path in (
        HERE / 'interactive_full.py', HERE / 'report.py',
        BENCH / 'full_ssl_jobs' / 'worker.py',
        BENCH / 'full_ssl_jobs' / 'tasks.tsv',
        BENCH / 'table_coverage.tsv', BENCH / 'protocol.json')}
    contract = {'plan': plan, 'code_hashes': hashes,
                'driver_hashes': driver_hashes,
                'ssl_worker_sha256': worker_hash}
    contract_path = root / 'plan.json'
    if contract_path.exists() and read(contract_path) != contract:
        raise ValueError('Evaluation inputs/code changed; use a different --output directory')
    atomic_json(contract_path, contract)

    @functools.lru_cache(maxsize=7)
    def datasets(target):
        return (RawDataset(args.data_root, target, True),
                RawDataset(args.data_root, target, False))

    units = plan['jobs'][0]['units']
    counts = {'evaluated_encoders': 0, 'already_complete': 0,
              'missing_checkpoint': 0}
    failures = []
    interrupted = False
    started = time.monotonic()
    print(f'Sequential full-data evaluation for {args.source}: '
          '18 possible encoders; missing inputs are skipped.', flush=True)
    print('100-epoch frozen linear probes, SGD lr=0.1 momentum=0.9 wd=0, '
          'batch<=256, cosine LR.', flush=True)
    print('Labels: 1% and 5%; main on all 3 backbones, downstream on ResNet18 only.',
          flush=True)
    print('Interrupted heads restart; valid completed probe JSONs are reused.', flush=True)
    try:
        for unit in units:
            folder = args.checkpoints / unit['relative']
            final_ready = all((folder / name).is_file()
                              and (folder / name).stat().st_size
                              for name in ('config.json', 'encoder.pt', 'ssl_complete.json'))
            rolling_ready = (accepted_rolling(unit) is not None
                             and all((folder / name).is_file()
                                     and (folder / name).stat().st_size
                                     for name in ('config.json', 'training.pt')))
            missing = [] if final_ready or rolling_ready else [
                name for name in ('config.json', 'encoder.pt', 'ssl_complete.json')
                if not (folder / name).is_file() or not (folder / name).stat().st_size]
            if missing:
                counts['missing_checkpoint'] += 1
                print('SKIP missing checkpoint:', unit['relative'], ', '.join(missing),
                      flush=True)
                continue
            if rolling_ready and not final_ready:
                acceptance = accepted_rolling(unit)
                print('USE accepted rolling checkpoint:', unit['relative'],
                      f"epoch {acceptance['accepted_epoch']}/"
                      f"{acceptance['requested_epoch']}", flush=True)
            output = root / unit['relative']
            complete_before = (output / 'complete.json').is_file()
            saved_probes = sum((output / 'evaluations'
                / f"{cell['target']}_lbl{cell['labels']}.json").is_file()
                for cell in unit['cells'])
            print(f'CHECK {unit["relative"]}: {saved_probes}/{len(unit["cells"])} '
                  'probes already saved', flush=True)
            try:
                evaluate_seed(unit, plan, root, datasets, hashes,
                              driver_hashes, worker_hash)
                field = 'already_complete' if complete_before else 'evaluated_encoders'
                counts[field] += 1
                print('SKIP already evaluated:' if complete_before else 'DONE:',
                      unit['relative'], flush=True)
            except Exception as error:
                failures.append({'encoder': unit['relative'], 'error': str(error)})
                print('ERROR:', unit['relative'], flush=True)
                traceback.print_exc()
            finally:
                gc.collect()
                torch.cuda.empty_cache()
            if (counts['evaluated_encoders'] + counts['already_complete']) % 3 == 0:
                report(root, plan, hashes, driver_hashes)
    except KeyboardInterrupt:
        interrupted = True
        print('\nStopped. Completed probes are retained; rerun the same command.',
              flush=True)
    finally:
        progress = report(root, plan, hashes, driver_hashes)
        atomic_json(root / 'logs' / f'{session}.json', {
            **counts, 'failures': failures, 'interrupted': interrupted,
            'seconds': time.monotonic() - started, 'progress': progress})
        print('SESSION:', json.dumps({**counts, 'errors': len(failures),
              'interrupted': interrupted}), flush=True)
        print('COVERAGE:', json.dumps(progress), flush=True)
        print('Results:', root, flush=True)
        print('Rerun to pick up newly finished checkpoints. No automatic waiting.',
              flush=True)
    return 130 if interrupted else (1 if failures or progress['invalid_encoders'] else 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', choices=SOURCES)
    parser.add_argument('--data-root', type=Path, default=BENCH.parent.parent / 'data')
    parser.add_argument('--checkpoints', type=Path,
                        default=BENCH / 'full_ssl_checkpoints_1000ep_3seeds')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        parser.error('Run on your allocated GPU node, not the login node')
    if os.environ.get('SLURM_CLUSTER_NAME', os.environ.get('CC_CLUSTER')) != 'vulcan':
        parser.error('Full-data checkpoints and evaluation belong on Vulcan')
    args.data_root = args.data_root.resolve()
    args.checkpoints = args.checkpoints.resolve()
    root = (args.output or BENCH / f'full_eval_results_interactive_3seeds_{args.source}').resolve()
    if (not root.is_relative_to(BENCH) or root == BENCH
            or root.is_relative_to(args.checkpoints)):
        parser.error('Use a separate output directory inside Evaluate_benchamrk')
    if not args.data_root.is_dir() or not args.checkpoints.is_dir():
        parser.error('Missing data/checkpoint root; check --data-root and --checkpoints')
    root.mkdir(parents=True, exist_ok=True)
    (root / 'logs').mkdir(exist_ok=True)
    session = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + f'_{os.getpid()}'
    log_path = root / 'logs' / f'{session}.log'
    with exclusive_run(root / 'interactive_session'), log_path.open('a', buffering=1) as log:
        with contextlib.redirect_stdout(Tee(sys.stdout, log)), \
                contextlib.redirect_stderr(Tee(sys.stderr, log)):
            print('Session log:', log_path, flush=True)
            try:
                return run(args, root, session)
            except Exception:
                traceback.print_exc()
                return 1


if __name__ == '__main__':
    raise SystemExit(main())
