"""Feed official SAS indices to the unchanged random-baseline trainer and probes."""
import argparse
import gc
import json
import math
import os
from pathlib import Path
import pickle
import statistics
import sys

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent
sys.path.insert(0, str(BENCH))
import torch
from common import atomic_json, code_hashes, digest_json, exclusive_run, file_hash
from data import RawDataset
from evaluate import linear_probe
from train_ssl import train

DATA = Path('/scratch/mmoslem3/data')
SELECTION = Path('/scratch/mmoslem3/sas_official_standalone/r18_300/selections')
REFERENCE = BENCH / 'cifar_small_simclr2k_1ssl5probes/results'


def read(path):
    return json.loads(Path(path).read_text())


def validate_indices(indices, count):
    if (not isinstance(indices, list) or len(indices) != count or count < 2
            or any(type(i) is not int or not 0 <= i < 50000 for i in indices)
            or len(set(indices)) != count):
        raise ValueError('Invalid SAS indices')


def prepare(source_filter, percentage_filter):
    validate_indices([0, 49999], 2)
    for invalid in ([1, 1], [-1, 2], [0, 50000], [False, 2], [0]):
        try:
            validate_indices(invalid, 2)
        except ValueError:
            continue
        raise AssertionError('Invalid indices accepted')
    protocol = read(BENCH / 'protocol.json')
    protocol['ssl'].update(subset_epochs=2000, checkpoint_every_epochs=500)
    protocol['subset_seeds'] = [0]
    hashes = code_hashes()
    for source in (source_filter,):
        dataset = RawDataset(DATA, source, True)
        for percentage in (percentage_filter,):
            ref_pct = '0.2' if percentage == '0.2' else '1'
            ref = REFERENCE / source / 'ResNet18/random' / f'pt{ref_pct}/simclr/ssl_seed_000'
            previous = read(ref / 'config.json')
            assert protocol == previous['signature']['protocol'], 'Random-baseline recipe changed'
            assert hashes == previous['signature']['code_hashes'], 'Shared evaluation code changed'
            assert dataset.fingerprint == previous['signature']['source_fingerprint']
            folder = SELECTION / source.lower() / f'{percentage}pct'
            complete = read(folder / 'complete.json')
            index_file = folder / f'{source.lower()}_{percentage}pct_indices.pkl'
            assert file_hash(index_file) == complete['indices_sha256']
            assert complete['recipe']['dataset'] == source.lower()
            assert complete['recipe']['percentage'] == percentage
            with index_file.open('rb') as stream:
                indices = pickle.load(stream)
            validate_indices(indices, complete['actual'])
            partition = read(folder / 'partition.json')
            assert file_hash(folder / 'partition.json') == complete['partition_sha256']
            groups = partition['groups']
            assert sorted(i for group in groups.values() for i in group) == list(range(50000))
            fraction = float(percentage) / 100
            picked = set(indices)
            for group in groups.values():
                assert len(picked.intersection(group)) == int(fraction * len(group))
            signature = dict(experiment='sas_r18_300_simclr2k_1ssl5probes', source=source,
                source_fingerprint=dataset.fingerprint, model='ResNet18', selection_method='sas',
                subset=float(percentage), ssl='simclr', protocol=protocol, code_hashes=hashes,
                worker_sha256=file_hash(__file__), probe_seeds=list(range(5)),
                selection_origin=dict(indices=str(index_file), indices_sha256=complete['indices_sha256'],
                    complete=str(folder / 'complete.json'), complete_sha256=file_hash(folder/'complete.json'),
                    recipe=complete['recipe'], nominal=complete['nominal'], actual=len(indices)))
            identity = dict(signature=digest_json(signature), seed=0, selection_method='sas',
                selection=dict(indices=indices, actual=len(indices), nominal=complete['nominal']))
            config = dict(signature=signature, identity=identity)
            output = HERE / 'results' / source / f'pt{percentage}'
            output.mkdir(parents=True, exist_ok=True)
            config_path = output / 'config.json'
            if config_path.exists():
                assert read(config_path) == config
            atomic_json(config_path, config)
            print('VALIDATED', source, percentage, len(indices), flush=True)
        del dataset
        gc.collect()
    print('PASS: SAS indices, dataset ordering, unchanged 2k recipe, input validation', flush=True)


def run(stage, source, percentage):
    assert os.environ.get('SLURM_JOB_ID') and torch.cuda.is_available()
    assert os.environ.get('SLURM_JOB_ACCOUNT') == 'aip-yiweilu'
    if stage == 'ssl':
        prepare(source, percentage)
    folder = HERE / 'results' / source / f'pt{percentage}'
    config = read(folder / 'config.json')
    signature, identity = config['signature'], config['identity']
    assert signature['worker_sha256'] == file_hash(__file__)
    assert signature['code_hashes'] == code_hashes()
    assert digest_json(signature) == identity['signature']
    origin = signature['selection_origin']
    assert file_hash(origin['indices']) == origin['indices_sha256']
    assert file_hash(origin['complete']) == origin['complete_sha256']
    indices = identity['selection']['indices']
    validate_indices(indices, origin['actual'])
    protocol = signature['protocol']
    train_set = RawDataset(DATA, source, True)
    assert train_set.fingerprint == signature['source_fingerprint']
    checkpoint = folder / 'encoder.pt'
    with exclusive_run(folder):
        if stage == 'ssl':
            selected = torch.utils.data.Subset(train_set, indices)
            checkpoint, saved = train(selected, protocol, source, 'ResNet18', 'simclr', 0,
                folder, identity, workers=0, device='cuda')
            assert saved['epochs'] == 2000
            assert saved['steps'] == 2000 * (len(indices) // min(len(indices), protocol['ssl']['batch_size']))
            atomic_json(folder/'ssl_complete.json', dict(complete=True, identity=identity,
                epochs=2000, encoder_sha256=saved['encoder_sha256'], checkpoint_file_sha256=file_hash(checkpoint)))
            return
        marker = read(folder/'ssl_complete.json')
        assert marker['complete'] and marker['identity'] == identity and marker['epochs'] == 2000
        assert file_hash(checkpoint) == marker['checkpoint_file_sha256']
        test_set = RawDataset(DATA, source, False)
        for labels in (1, 5):
            samples = []
            for seed in range(5):
                path = folder/'evaluations'/f'{source}_lbl{labels}_probe{seed}.json'
                result = linear_probe(checkpoint, identity, protocol, source, source, train_set,
                    test_set, labels, seed, path, workers=0, device='cuda')
                assert result['complete'] and result['encoder_unchanged']
                assert result['encoder_sha256'] == marker['encoder_sha256']
                assert math.isfinite(result['accuracy_percent']) and 0 <= result['accuracy_percent'] <= 100
                samples.append(dict(ssl_seed=0, probe_seed=seed, accuracy_percent=result['accuracy_percent'],
                    result_file=str(path), encoder_sha256=result['encoder_sha256']))
            scores = [sample['accuracy_percent'] for sample in samples]
            atomic_json(folder/f'summary_lbl{labels}.json', dict(complete=True, source=source,
                model='ResNet18', kind='sas', subset=float(percentage), actual_selected=len(indices),
                labels=labels, ssl='simclr', ssl_seed=0, ssl_epochs=2000, linear_epochs=100,
                completed_probes=5, expected_probes=5, std_ddof=1, samples=samples,
                mean_percent=statistics.mean(scores), std_percent=statistics.stdev(scores)))
        print('EVALUATION_COMPLETE', source, percentage, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['ssl', 'probe'])
    parser.add_argument('source', nargs='?', choices=['CIFAR10', 'CIFAR100'])
    parser.add_argument('percentage', nargs='?', choices=['0.2', '1', '2'])
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert args.source and args.percentage
    run(args.stage, args.source, args.percentage)
