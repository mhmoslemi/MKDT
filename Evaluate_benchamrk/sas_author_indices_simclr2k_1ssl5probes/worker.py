"""Feed prefixes of the authors supplied indices to the unchanged random-baseline trainer and probes."""
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
AUTHOR_ROOT = Path("/scratch/mmoslem3/sas_official_standalone/upstream")
AUTHOR_SHA256 = {
    "CIFAR10": "d6a6101c138ee04fa6a270b3b4a6cc5abf330d0c654da864df28822d04a1f7a2",
    "CIFAR100": "f468c11d142a40218df9c61d633043954792fbc2dddcb5f3cb642cdce2b14de4",
}
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
            index_file = AUTHOR_ROOT / 'final_subsets' / f'{source.lower()}-cl-core-idx.pkl'
            digest = file_hash(index_file)
            assert digest == AUTHOR_SHA256[source], 'Author index file changed'
            with index_file.open('rb') as stream:
                ordered = pickle.load(stream)['indices']
            validate_indices(ordered, 50000)
            examples = sorted((AUTHOR_ROOT / 'examples' / source.lower()).glob('*.pkl'))
            assert len(examples) == 4
            example_hashes = {}
            for example in examples:
                with example.open('rb') as stream:
                    supplied = pickle.load(stream)
                validate_indices(supplied, len(supplied))
                assert supplied == ordered[:len(supplied)], 'Published subset is not a prefix'
                example_hashes[example.name] = file_hash(example)
            count = {'0.2': 100, '1': 500, '2': 1000}[percentage]
            indices = ordered[:count]
            validate_indices(indices, count)
            recipe = dict(source='authors_ordered_indices', operation='prefix', count=count,
                dataset=source.lower(), percentage=percentage,
                upstream_commit='5e5741884c2e5dcdea65467989b1ab0720430eca',
                published_examples_sha256=example_hashes,
                note='Small budgets extend the supplied ordering; no proxy or new SAS selection.')
            signature = dict(experiment='sas_author_indices_simclr2k_1ssl5probes', source=source,
                source_fingerprint=dataset.fingerprint, model='ResNet18', selection_method='sas',
                subset=float(percentage), ssl='simclr', protocol=protocol, code_hashes=hashes,
                worker_sha256=file_hash(__file__), probe_seeds=list(range(5)),
                selection_origin=dict(indices=str(index_file), indices_sha256=digest,
                    recipe=recipe, nominal=count, actual=len(indices)))
            identity = dict(signature=digest_json(signature), seed=0, selection_method='sas',
                selection=dict(indices=indices, actual=len(indices), nominal=count))
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
