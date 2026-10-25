"""One Random/ResNet18BN downstream cell and one seed per GPU job."""
import argparse
import contextlib
import copy
import fcntl
import json
import math
import os
from pathlib import Path
import statistics
import sys

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent
sys.path.insert(0, str(BENCH))

import torch

from common import (atomic_json, code_hashes, digest_json, exclusive_run,
                    file_hash, load_encoder)
from data import RawDataset, SOURCE_COUNTS, select_subset
from evaluate import linear_probe
from run import validate_protocol
from train_ssl import train

HASHES = code_hashes()
REFERENCE = json.loads((BENCH / 'protocol.json').read_text())


def read(path):
    return json.loads(Path(path).read_text())


def inside(value):
    path = (BENCH / value).resolve()
    if not path.is_relative_to(BENCH) or path == BENCH:
        raise ValueError('Checkpoint and result paths must stay inside Evaluate_benchamrk')
    return path


def route_name(source, subset, ssl, seed):
    return f'{source}_p{float(subset):g}_{ssl}_s{seed}'.replace('.', 'p')


@contextlib.contextmanager
def reader(folder):
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / '.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f'Stop the other training/interactive writer using {folder}') from error
        yield


def check_config(config, route):
    signature, identity = config['signature'], config['identity']
    reference = copy.deepcopy(REFERENCE)
    reference['ssl'].update(subset_epochs=route['epochs'],
        checkpoint_every_epochs=signature['protocol']['ssl']['checkpoint_every_epochs'])
    if (digest_json(signature) != identity['signature']
            or signature['code_hashes'] != HASHES or signature['protocol'] != reference
            or signature['source'] != route['source'] or signature['model'] != 'ResNet18'
            or signature['subset'] != route['subset'] or signature['ssl'] != route['ssl']
            or signature['selection_method'] != 'random'
            or identity['selection_method'] != 'random' or identity['seed'] != route['seed']
            or signature['protocol']['ssl']['subset_epochs'] != route['epochs']):
        raise ValueError('Saved encoder configuration differs from the requested random seed')
    validate_protocol(signature['protocol'])


def ensure_encoder(route, data_root):
    folder = inside(route['checkpoint_dir'])
    if route['mode'] == 'fallback' and not (folder / 'encoder.pt').is_file():
        with exclusive_run(folder):
            protocol = read(BENCH / 'protocol.json')
            protocol['ssl'].update(subset_epochs=route['epochs'], checkpoint_every_epochs=500)
            validate_protocol(protocol)
            base = RawDataset(data_root, route['source'], True)
            if len(base) != SOURCE_COUNTS[route['source']]:
                raise ValueError('Unexpected source training split')
            selected, selection = select_subset(base, 'random', route['subset'], route['seed'],
                protocol, HERE / 'selections', 0, 'cuda')
            signature = {'protocol': protocol, 'code_hashes': code_hashes(),
                         'source': route['source'], 'source_fingerprint': base.fingerprint,
                         'model': 'ResNet18', 'selection_method': 'random',
                         'ssl': route['ssl'], 'subset': route['subset'],
                         'driver_sha256': file_hash(__file__)}
            identity = {'signature': digest_json(signature), 'seed': route['seed'],
                        'selection': selection, 'selection_method': 'random'}
            config = {'identity': identity, 'signature': signature}
            path = folder / 'config.json'
            if path.exists() and read(path) != config:
                raise ValueError('Refusing to overwrite a different fallback SSL configuration')
            atomic_json(path, config)
            print('TRAIN/RESUME missing encoder:', folder, 'epochs:', route['epochs'], flush=True)
            train(selected, protocol, route['source'], 'ResNet18', route['ssl'],
                  route['seed'], folder, identity, 0, 'cuda')
    config = read(folder / 'config.json')
    check_config(config, route)
    checkpoint = folder / 'encoder.pt'
    encoder, _, saved = load_encoder(checkpoint, config['identity'], 'cpu')
    del encoder
    saved = {key: value for key, value in saved.items() if key != 'encoder'}
    protocol = config['signature']['protocol']
    if (saved['epochs'] != route['epochs'] or saved['seed'] != route['seed']
            or saved['source'] != route['source'] or saved['model'] != 'ResNet18'
            or saved['method'] != route['ssl']
            or saved['input_size'] != protocol['input_size'][route['source']]
            or saved['normalization'] != protocol['normalization'][route['source']]):
        raise ValueError('Saved encoder metadata mismatch')
    checksum = file_hash(checkpoint)
    marker = folder / 'ssl_complete.json'
    if marker.exists():
        prior = read(marker)
        if (not prior['complete'] or prior['identity'] != config['identity']
                or prior['epochs'] != saved['epochs']
                or prior['encoder_sha256'] != saved['encoder_sha256']
                or prior['checkpoint_file_sha256'] != checksum):
            raise ValueError('Saved encoder and completion marker disagree')
    else:
        atomic_json(marker, {**saved, 'checkpoint_file_sha256': checksum})
    return config, saved, checksum


def valid_probe(result, config, marker, route, target, labels):
    expected = result.get('expected', {})
    score = result.get('accuracy_percent')
    return (result.get('complete') is True and result.get('encoder_unchanged') is True
            and result.get('encoder_sha256') == marker.get('encoder_sha256')
            and marker.get('complete') is True and marker.get('epochs') == route['epochs']
            and marker.get('identity') == config['identity']
            and expected.get('identity') == config['identity']
            and expected.get('source') == route['source'] and expected.get('target') == target
            and expected.get('seed') == route['seed'] and expected.get('label_percentage') == labels
            and expected.get('evaluation') == config['signature']['protocol']['evaluation']
            and expected.get('evaluation_code_hashes') == HASHES
            and isinstance(score, (int, float)) and math.isfinite(score) and 0 <= score <= 100)


def summarize(route, target, labels):
    group = HERE / 'summaries' / route['source'] / f"pt{route['subset']:g}" / route['ssl']
    group.mkdir(parents=True, exist_ok=True)
    with (group / f'.{target}_lbl{labels}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        samples, fingerprints = [], set()
        for seed in range(10):
            choice = HERE / 'routes' / (route_name(route['source'], route['subset'], route['ssl'], seed) + '.json')
            if not choice.exists():
                continue
            other = read(choice)
            folder, output = inside(other['checkpoint_dir']), inside(other['output_dir'])
            result = output / 'evaluations' / f'{target}_lbl{labels}.json'
            if not result.exists():
                continue
            config, marker, row = read(folder / 'config.json'), read(folder / 'ssl_complete.json'), read(result)
            check_config(config, other)
            if not valid_probe(row, config, marker, other, target, labels):
                raise ValueError(f'Invalid saved downstream probe: {result}')
            fingerprints.add((row['expected']['train_fingerprint'], row['expected']['test_fingerprint']))
            if len(fingerprints) > 1:
                raise ValueError('Downstream seeds use different target dataset files')
            samples.append({'seed': seed, 'accuracy_percent': row['accuracy_percent'],
                            'ssl_epochs': other['epochs'], 'result': str(result)})
        complete = len(samples) == 10
        scores = [row['accuracy_percent'] for row in samples]
        mean = statistics.mean(scores) if complete else None
        std = statistics.stdev(scores) if complete else None
        atomic_json(group / f'{target}_lbl{labels}.json', {
            'source': route['source'], 'model': 'ResNet18', 'method': 'random',
            'subset': route['subset'], 'ssl': route['ssl'], 'target': target, 'labels': labels,
            'complete': complete, 'completed_seeds': len(samples), 'expected_seeds': 10,
            'mean_percent': mean, 'std_percent': std, 'std_ddof': 1, 'samples': samples,
            'latex': f'\\dsacc{{{mean:.2f}}}{{{std:.2f}}}' if complete else ''})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('route', type=Path)
    parser.add_argument('target', choices=('CIFAR10', 'CIFAR100', 'TinyImageNet', 'Aircraft', 'CUB2011', 'Dogs', 'Flowers'))
    parser.add_argument('labels', type=int, choices=(1, 5))
    parser.add_argument('data_root', type=Path)
    args = parser.parse_args()
    if (not os.environ.get('SLURM_JOB_ID')
            or os.environ.get('SLURM_CLUSTER_NAME', os.environ.get('CC_CLUSTER')) != 'killarney'
            or not torch.cuda.is_available()):
        raise RuntimeError('Run on the allocated Killarney GPU, not a login node')
    if not args.route.resolve().is_relative_to(HERE / 'routes'):
        raise ValueError('Unexpected encoder route file')
    route = read(args.route)
    if (route['source'] not in ('CIFAR100', 'TinyImageNet')
            or route['subset'] not in ((2,) if route['source'] == 'CIFAR100' else (0.2, 1, 2))
            or route['ssl'] not in ('simclr', 'barlowtwins') or route['seed'] not in range(10)
            or args.target == route['source'] or route['mode'] not in ('existing', 'fallback')
            or (route['mode'] == 'fallback' and route['epochs'] != (2000 if route['ssl'] == 'simclr' else 10000))):
        raise ValueError('Unexpected Random downstream seed job')
    if args.route.name != route_name(route['source'], route['subset'], route['ssl'], route['seed']) + '.json':
        raise ValueError('Encoder route filename and contents disagree')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    output = inside(route['output_dir'])
    with exclusive_run(output / '.probe_locks' / f'{args.target}_lbl{args.labels}'):
        config, saved, checksum = ensure_encoder(route, args.data_root)
        checkpoint = inside(route['checkpoint_dir']) / 'encoder.pt'
        with reader(output), reader(checkpoint.parent):
            path = output / 'config.json'
            if path.exists() and read(path) != config:
                raise ValueError('Refusing to mix encoders in a probe output directory')
            atomic_json(path, config)
            train_set = RawDataset(args.data_root, args.target, True)
            test_set = RawDataset(args.data_root, args.target, False)
            result = linear_probe(checkpoint, config['identity'], config['signature']['protocol'],
                route['source'], args.target, train_set, test_set, args.labels, route['seed'],
                output / 'evaluations' / f'{args.target}_lbl{args.labels}.json', 0, 'cuda')
            if not valid_probe(result, config, saved, route, args.target, args.labels):
                raise ValueError('Probe failed provenance validation')
            if file_hash(checkpoint) != checksum:
                raise ValueError('Encoder changed during the linear probe')
            summarize(route, args.target, args.labels)


if __name__ == '__main__':
    main()
