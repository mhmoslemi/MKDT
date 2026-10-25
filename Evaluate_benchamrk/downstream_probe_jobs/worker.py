"""Run existing-reference three-probe bundles, optionally five per job; never train SSL."""
import argparse
import contextlib
import fcntl
import functools
import importlib.util
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent
sys.path[:0] = [str(BENCH / 'eval_jobs'), str(BENCH)]
import interactive_ours as ours
from common import load_encoder

spec = importlib.util.spec_from_file_location('random_downstream', BENCH / 'random_downstream_jobs/worker.py')
random = importlib.util.module_from_spec(spec)
spec.loader.exec_module(random)


def inside(value):
    folder = (BENCH / value).resolve()
    if not folder.is_relative_to(BENCH) or folder == BENCH:
        raise ValueError('Checkpoint/result path must stay inside Evaluate_benchamrk')
    return folder


def run(job, args):
    unit, probes = job['unit'], job['probes']
    source, subset, method, seed = (unit[key] for key in ('source', 'subset', 'ssl', 'seed'))
    if (unit['kind'] not in ('ours', 'random') or source not in ('CIFAR10', 'CIFAR100', 'TinyImageNet')
            or subset not in (0.2, 1, 2) or method not in ours.METHODS or len(probes) != 3
            or seed not in range(2 if unit['kind'] == 'ours' else 10)):
        raise ValueError('Invalid three-probe unit')
    targets = [name for name in ours.evaluation_targets(source, 'ResNet18') if name != source]
    for probe in probes:
        if (probe['target'] not in targets or probe['labels'] not in (1, 5)
                or probe['seed'] not in (range(3) if unit['kind'] == 'ours' else (seed,))):
            raise ValueError('Invalid downstream target, labels or probe seed')
    if len({(p['target'], p['labels'], p['seed']) for p in probes}) != 3:
        raise ValueError('Duplicate probes')
    omitted = set(ours.read(HERE / 'omitted_cells.json'))
    probes = [p for p in probes if '|'.join(map(str, (source, 'ResNet18', unit['kind'],
              subset, method, p['target'], p['labels']))) not in omitted]
    if not probes:
        print('SKIP cells marked --:', job['name'], flush=True)
        return
    folder, output = inside(unit['checkpoint_dir']), inside(unit['output_dir'])
    checkpoint = folder / 'encoder.pt'
    if not checkpoint.is_file():
        print('SKIP missing encoder; no SSL fallback:', checkpoint, flush=True)
        return
    with contextlib.ExitStack() as locks:
        for directory in sorted({folder, folder.parent, output, output.parent}):
            locks.enter_context(random.reader(directory))
        config, marker = ours.read(folder / 'config.json'), ours.read(folder / 'ssl_complete.json')
        signature, identity = config['signature'], config['identity']
        if (ours.digest_json(signature) != identity['signature'] or signature['code_hashes'] != ours.code_hashes()
                or signature['source'] != source or signature['model'] != 'ResNet18'
                or signature['selection_method'] != unit['kind'] or signature['subset'] != subset
                or signature['ssl'] != method or identity['seed'] != seed
                or not marker['complete'] or marker['identity'] != identity or marker['epochs'] != unit['epochs']):
            raise ValueError('Saved checkpoint configuration/identity mismatch')
        if unit['kind'] == 'ours':
            epochs = 10000 if method == 'barlowtwins' else 1000 if subset == 2 else 2000
            if unit['epochs'] != epochs:
                raise ValueError('Ours 2% SimCLR must use 1k; other SimCLR 2k; Barlow 10k')
            reference = ours.expected_protocol('random', source, subset)
            reference['ssl'].update(subset_epochs=epochs, checkpoint_every_epochs=500)
            reference['subset_seeds'] = [0, 1]
            if signature['protocol'] != reference:
                raise ValueError('Ours checkpoint differs from the reference recipe')
        else:
            route = ours.read(inside(unit['route']))
            if any(route[key] != unit[key] for key in route):
                raise ValueError('Random encoder route changed')
            random.check_config(config, route)
        protocol = signature['protocol']
        if (protocol['ssl']['subset_epochs'] != unit['epochs'] or protocol['evaluation']['epochs'] != 100
                or protocol['evaluation'] != ours.read(BENCH / 'protocol.json')['evaluation']):
            raise ValueError('Wrong SSL horizon or frozen-probe recipe')
        checksum = ours.file_hash(checkpoint)
        if checksum != marker['checkpoint_file_sha256']:
            raise ValueError('Checkpoint file differs from completion marker')
        encoder, _, saved = load_encoder(checkpoint, identity, 'cpu')
        del encoder
        expected = {'source': source, 'model': 'ResNet18', 'method': method, 'seed': seed,
                    'epochs': unit['epochs'], 'input_size': protocol['input_size'][source],
                    'normalization': protocol['normalization'][source], 'encoder_sha256': marker['encoder_sha256']}
        if any(saved.get(key) != value for key, value in expected.items()):
            raise ValueError('Encoder metadata mismatch')
        with ours.exclusive_run(output / '.metadata'):
            values = {'config.json': config}
            if unit['kind'] == 'ours':
                values.update({'ssl_complete.json': {**marker, 'checkpoint': str(checkpoint)},
                    'checkpoint_reference.json': {'checkpoint': str(checkpoint), 'scope': 'downstream',
                    'checkpoint_file_sha256': checksum, 'driver_sha256': ours.file_hash(__file__), 'targets': targets}})
            for name, value in values.items():
                file = output / name
                if file.exists() and ours.read(file) != value:
                    raise ValueError(f'Refusing to mix results: {file}')
                ours.atomic_json(file, value)

        @functools.lru_cache(maxsize=1)
        def datasets(target):
            return ours.RawDataset(args.data_root, target, True), ours.RawDataset(args.data_root, target, False)

        print('DOWNSTREAM ONLY:', unit, 'probes:', probes, flush=True)
        for probe in probes:
            target, labels, probe_seed = (probe[key] for key in ('target', 'labels', 'seed'))
            suffix = f'_probe{probe_seed}' if unit['kind'] == 'ours' else ''
            path = output / 'evaluations' / f'{target}_lbl{labels}{suffix}.json'
            lock = output / '.probe_locks' / f'{target}_lbl{labels}{suffix}'
            with ours.exclusive_run(lock):
                train_set, test_set = datasets(target)
                result = ours.linear_probe(checkpoint, identity, protocol, source, target,
                    train_set, test_set, labels, probe_seed, path, workers=0, device='cuda')
                if (not result['complete'] or not result['encoder_unchanged']
                        or result['encoder_sha256'] != marker['encoder_sha256']):
                    raise ValueError('Incomplete probe or changed encoder')
            if unit['kind'] == 'random':
                random.summarize(route, target, labels)
            else:
                with (output.parent / '.summary.lock').open('a') as lock:
                    fcntl.flock(lock, fcntl.LOCK_EX)
                    ours.summarize(output.parent, identity['signature'], source, 'ResNet18BN', subset, method, targets)
        if ours.file_hash(checkpoint) != checksum:
            raise ValueError('Encoder changed during probing')
    print('COMPLETE:', job['name'], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('data_root', type=Path)
    args = parser.parse_args()
    if (os.environ.get('SLURM_CLUSTER_NAME', os.environ.get('CC_CLUSTER')) != 'killarney'
            or os.environ.get('SLURM_JOB_ACCOUNT') != 'aip-boyuwang'
            or not os.environ.get('SLURM_JOB_ID') or not ours.torch.cuda.is_available()):
        parser.error('Use a Wang-account GPU job on Killarney')
    if not args.manifest.resolve().is_relative_to(HERE / 'batches'):
        parser.error('Unexpected job manifest')
    ours.torch.set_num_threads(1)
    ours.torch.set_num_interop_threads(1)
    ours.torch.backends.cuda.matmul.allow_tf32 = False
    ours.torch.backends.cudnn.allow_tf32 = False
    job = ours.read(args.manifest)
    bundles = job.get('bundles', [job])
    if len(bundles) not in (1, 5):
        raise ValueError('Expected one or five three-probe bundles')
    for bundle in bundles:
        run(bundle, args)


if __name__ == '__main__':
    main()
