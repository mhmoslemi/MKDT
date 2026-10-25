"""Finish saved Tiny ImageNet Random encoders' main probes and Aircraft seed 9."""
import argparse
import contextlib
import functools
import os
from pathlib import Path

import worker as probes

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('data_root', type=Path)
    args = parser.parse_args()
    if (not os.environ.get('SLURM_JOB_ID')
            or os.environ.get('SLURM_CLUSTER_NAME', os.environ.get('CC_CLUSTER')) != 'killarney'
            or os.environ.get('SLURM_JOB_ACCOUNT') != 'aip-boyuwang'
            or not probes.torch.cuda.is_available()):
        parser.error('Use a Wang-account GPU job on Killarney')
    if not args.manifest.resolve().is_relative_to(HERE / 'finish_tiny_batches'):
        parser.error('Unexpected job manifest')
    job = probes.read(args.manifest)
    if not 1 <= len(job['probes']) <= 15:
        raise ValueError('Expected at most fifteen sequential probes')
    probes.torch.set_num_threads(1)
    probes.torch.set_num_interop_threads(1)
    probes.torch.backends.cuda.matmul.allow_tf32 = False
    probes.torch.backends.cudnn.allow_tf32 = False

    @functools.lru_cache(maxsize=1)
    def datasets(target):
        return (probes.RawDataset(args.data_root, target, True),
                probes.RawDataset(args.data_root, target, False))

    seen = set()
    for request in job['probes']:
        route_file = probes.inside(request['route'])
        if not route_file.is_relative_to(HERE / 'routes'):
            raise ValueError('Unexpected encoder route')
        route = probes.read(route_file)
        target, labels = request['target'], request['labels']
        subset, method, seed = route['subset'], route['ssl'], route['seed']
        main_probe = (target == 'TinyImageNet' and labels in (1, 5)
                      and not (subset == 2 and method == 'barlowtwins'))
        aircraft_probe = (target == 'Aircraft' and labels == 1 and subset == 2
                          and method == 'simclr' and seed == 9)
        if (route['source'] != 'TinyImageNet' or route['mode'] != 'existing'
                or subset not in (0.2, 1, 2) or method not in ('simclr', 'barlowtwins')
                or seed not in range(10) or route['epochs'] != {0.2: 6000, 1: 5000, 2: 4000}[subset]
                or not (main_probe or aircraft_probe)
                or route_file.name != probes.route_name('TinyImageNet', subset, method, seed) + '.json'):
            raise ValueError('Probe is outside the requested missing Random cells')
        identifier = (str(route_file), target, labels)
        if identifier in seen:
            raise ValueError('Duplicate probe in manifest')
        seen.add(identifier)
        folder, output = probes.inside(route['checkpoint_dir']), probes.inside(route['output_dir'])
        if not all((folder / name).is_file() for name in ('encoder.pt', 'config.json', 'ssl_complete.json')):
            print('SKIP missing encoder; no SSL training:', folder, flush=True)
            continue
        with contextlib.ExitStack() as locks:
            for directory in sorted({folder, folder.parent, output, output.parent}):
                locks.enter_context(probes.reader(directory))
            locks.enter_context(probes.exclusive_run(output / '.probe_locks' / f'{target}_lbl{labels}'))
            # Existing mode forbids the shared helper's SSL fallback.
            config, saved, checksum = probes.ensure_encoder(route, args.data_root)
            config_file = output / 'config.json'
            if config_file.exists() and probes.read(config_file) != config:
                raise ValueError('Refusing to mix encoder configurations')
            probes.atomic_json(config_file, config)
            train_set, test_set = datasets(target)
            destination = output / 'evaluations' / f'{target}_lbl{labels}.json'
            print('REUSE' if destination.exists() else 'PROBE', route_file.stem, target, labels, flush=True)
            result = probes.linear_probe(folder / 'encoder.pt', config['identity'], config['signature']['protocol'],
                'TinyImageNet', target, train_set, test_set, labels, seed, destination, 0, 'cuda')
            if not probes.valid_probe(result, config, saved, route, target, labels):
                raise ValueError('Probe provenance validation failed')
            if probes.file_hash(folder / 'encoder.pt') != checksum:
                raise ValueError('Encoder changed during probing')
            probes.summarize(route, target, labels)
    print('COMPLETE:', job['name'], flush=True)


if __name__ == '__main__':
    main()
