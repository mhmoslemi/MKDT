"""One saved SSL seed and its missing ResNet18BN main probes; unchanged benchmark code."""
import argparse
import functools
import importlib.util
import itertools
import math
import os
from pathlib import Path
import statistics
import subprocess

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent
spec = importlib.util.spec_from_file_location('main_finish_base', BENCH / 'cifar_finish_jobs/worker.py')
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
original_config = base.synthetic_config


def synthetic_config(task, args, datasets):
    if task['kind'] != 'mkdt' or task['subset'] != 0.2:
        return original_config(task, args, datasets)
    protocol = base.ours.expected_protocol('random', task['source'], 0.2)
    protocol['ssl'].update(subset_epochs=task['epochs'], checkpoint_every_epochs=500)
    protocol['subset_seeds'] = [0, 1]
    entry = base.mkdt.synthetic_entry(args.synthetic_root, task['source'], 'ResNet18', 0.2)
    marker = base.read(Path(entry['path']).parent / 'complete_0.2.json')
    count, iterations = (200, 10000) if task['source'] == 'TinyImageNet' else (100, 5000)
    if (not marker['complete'] or marker['count'] != count or marker['iterations'] != iterations
            or base.file_hash(entry['path']) != marker['files'][f'images_{iterations}.pt']):
        raise ValueError('MKDT 0.2% generation is unfinished or its images changed')
    synthetic = base.SyntheticDataset(entry, task['source'], 0.2, protocol['input_size'][task['source']])
    signature = {'protocol': protocol, 'source': task['source'], 'model': 'ResNet18',
        'subset': 0.2, 'ssl': task['ssl'], 'selection_method': 'mkdt',
        'source_fingerprint': datasets(task['source'])[0].fingerprint, 'synthetic': synthetic.metadata,
        'ssl_seeds': [0, 1], 'probe_seeds': [0, 1, 2], 'code_hashes': base.code_hashes(),
        'driver_sha256': base.file_hash(base.mkdt.__file__), 'input_scope': 'saved_synthetic_images_only'}
    return synthetic, {'signature': signature, 'identity': {
        'signature': base.digest_json(signature), 'seed': task['seed'], 'selection_method': 'mkdt'}}


def summarize(task, config):
    group = task['output_dir'].parent
    with (group / '.summary.lock').open('a') as lock:
        base.fcntl.flock(lock, base.fcntl.LOCK_EX)
        destination = group / 'summary.json'
        old = base.read(destination) if destination.exists() else {}
        if old.get('signature', config['identity']['signature']) != config['identity']['signature']:
            raise ValueError('Saved summary belongs to different synthetic inputs')
        rows = [r for r in old.get('cells', []) if r['target'] != task['source']]
        for labels in (1, 5):
            samples, fingerprints = [], set()
            for seed, probe in itertools.product((0, 1), (0, 1, 2)):
                folder = group / f'ssl_seed_{seed:03d}'
                path = folder / 'evaluations' / f'{task["source"]}_lbl{labels}_probe{probe}.json'
                if not path.exists():
                    continue
                result, saved_config, marker = (base.read(path), base.read(folder / 'config.json'),
                                                base.read(folder / 'ssl_complete.json'))
                identity = {**config['identity'], 'seed': seed}
                e, score = result['expected'], result['accuracy_percent']
                if (saved_config != {'signature': config['signature'], 'identity': identity}
                        or not marker['complete'] or marker['identity'] != identity or marker['epochs'] != task['epochs']
                        or not result['complete'] or not result['encoder_unchanged']
                        or result['encoder_sha256'] != marker['encoder_sha256'] or e['identity'] != identity
                        or (e['source'], e['target'], e['label_percentage'], e['seed']) != (
                            task['source'], task['source'], labels, probe)
                        or e['evaluation'] != config['signature']['protocol']['evaluation']
                        or e['evaluation_code_hashes'] != config['signature']['code_hashes']
                        or not math.isfinite(score) or not 0 <= score <= 100):
                    raise ValueError(f'Incompatible main probe: {path}')
                fingerprints.add((e['train_fingerprint'], e['test_fingerprint']))
                if len(fingerprints) != 1:
                    raise ValueError('Mixed main train/test datasets')
                samples.append({'ssl_seed': seed, 'probe_seed': probe, 'accuracy_percent': score})
            scores = [r['accuracy_percent'] for r in samples]
            complete = len(scores) == 6
            rows.append({**{k: task[k] for k in ('source', 'model', 'subset', 'ssl')},
                'method': task['kind'], 'target': task['source'], 'labels': labels,
                'completed_probes': len(scores), 'expected_probes': 6, 'complete': complete,
                'mean_percent': statistics.mean(scores) if complete else None,
                'std_percent': statistics.stdev(scores) if complete else None, 'samples': samples})
        base.atomic_json(destination, {**old, 'signature': config['identity']['signature'],
            'ssl_seeds': [0, 1], 'probe_seeds': [0, 1, 2], 'std_ddof': 1,
            'complete': all(r['complete'] for r in rows), 'cells': rows})


def run_random(task, args, datasets):
    settings = argparse.Namespace(checkpoints=BENCH / 'random_ssl_checkpoints_cifar10k8k6k_tiny6k5k4k',
                                  data_root=args.data_root)
    plan, hashes, drivers, worker_hash, contract = base.random_eval.evaluation_contract(settings)
    root = BENCH / 'random_eval_results_interactive_10seeds'
    if base.read(root / 'plan.json') != contract:
        raise ValueError('Saved Random evaluation contract changed')
    units = [u for job in plan['jobs'] for u in job['units']
             if all(u[k] == task[k] for k in ('source', 'model', 'subset', 'ssl', 'seed'))]
    if len(units) != 1:
        raise ValueError('Cannot identify the original Random seed')
    unit = units[0]
    if (settings.checkpoints / unit['relative']).resolve() != task['checkpoint_dir']:
        raise ValueError('Unexpected Random checkpoint path')
    folder = task['checkpoint_dir']
    if not (folder / 'encoder.pt').is_file():
        subprocess.run(['bash', str(BENCH / 'random_ssl.sh'), '--worker', str(BENCH),
            task['source'], task['model'], str(task['subset']), task['ssl'], str(args.venv),
            str(args.data_root), str(settings.checkpoints), str(task['seed']), str(task['epochs']), '1'], check=True)
    with base.exclusive_run(folder):
        config = base.read(folder / 'config.json')
        encoder, _, saved = base.load_encoder(folder / 'encoder.pt', config['identity'], 'cpu')
        del encoder
        if not (folder / 'ssl_complete.json').exists():
            base.atomic_json(folder / 'ssl_complete.json', {**{k: v for k, v in saved.items() if k != 'encoder'},
                'checkpoint_file_sha256': base.file_hash(folder / 'encoder.pt')})
    pending = [c for c in unit['cells'] if c['target'] == task['source'] and int(c['labels']) in task['labels']]
    base.random_eval.evaluate_pending_seed(unit, pending, plan, root, datasets, hashes, drivers, worker_hash)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('venv', type=Path)
    parser.add_argument('data_root', type=Path)
    parser.add_argument('synthetic_root', type=Path)
    args = parser.parse_args()
    if (not os.environ.get('SLURM_JOB_ID')
            or os.environ.get('SLURM_CLUSTER_NAME', os.environ.get('CC_CLUSTER')) != 'killarney'
            or not base.torch.cuda.is_available()):
        parser.error('Use this launcher in a Killarney GPU job')
    if not args.manifest.resolve().is_relative_to(HERE / 'batches'):
        parser.error('Unexpected main-table job manifest')
    job = base.read(args.manifest)
    account = 'aip-boyuwang' if job['seconds'] < 3600 else 'aip-yiweilu'
    if os.environ.get('SLURM_JOB_ACCOUNT') != account:
        parser.error(f'This job must use account {account}')
    task = job['unit']
    expected = {0.2: 6000, 1: 5000, 2: 4000}[task['subset']] if task['kind'] == 'random' else (
        2000 if task['ssl'] == 'simclr' else 10000)
    if (task['source'] != 'TinyImageNet' or task['model'] != 'ResNet18' or task['scope'] != 'main'
            or task['kind'] not in ('random', 'ours', 'mkdt') or task['subset'] not in (0.2, 1, 2)
            or task['ssl'] not in ('simclr', 'barlowtwins') or task['epochs'] != expected
            or (task['subset'] == 2 and task['ssl'] == 'barlowtwins')
            or task['seed'] not in range(10 if task['kind'] == 'random' else 2)
            or not task['labels'] or not set(task['labels']).issubset({1, 5}) or task['parent_dir'] is not None):
        raise ValueError('Task is outside the remaining main-table scope')
    for k in ('checkpoint_dir', 'output_dir'):
        task[k] = base.inside_bench(task[k])
    args.data_root, args.synthetic_root, args.venv = (p.resolve() for p in
                                                    (args.data_root, args.synthetic_root, args.venv))
    base.torch.set_num_threads(1)
    base.torch.set_num_interop_threads(1)
    base.torch.backends.cuda.matmul.allow_tf32 = False
    base.torch.backends.cudnn.allow_tf32 = False

    @functools.lru_cache(maxsize=1)
    def datasets(source):
        return base.RawDataset(args.data_root, source, True), base.RawDataset(args.data_root, source, False)

    base.synthetic_config = synthetic_config
    base.summarize_synthetic = summarize
    print('MAIN ONLY:', job['name'], task['epochs'], 'SSL epochs; 100-epoch frozen probes', flush=True)
    with base.exclusive_run(HERE / 'runtime' / job['name']):
        if task['kind'] == 'random':
            run_random(task, args, datasets)
        else:
            base.run_synthetic(task, args, datasets)
    print('COMPLETE:', job['name'], flush=True)


if __name__ == '__main__':
    main()
