"""Author-ranked CIFAR subsets and the existing 2k SimCLR/five-probe evaluation."""
import argparse
import json
import math
import os
from pathlib import Path
import statistics
import sys

HERE = Path(__file__).resolve().parent
BENCH, ROOT = HERE.parent, HERE.parent.parent
sys.path.insert(0, str(BENCH))
sys.path.insert(0, str(BENCH / 'selection_baselines'))
from selection_baselines import worker as reference
from sas_mkdt_indices.worker import IndexUnpickler


def cifar_indices(manifest, source):
    spec = manifest['rankings'][source]
    path = HERE / spec['path']
    assert reference.file_hash(path) == spec['sha256']
    with path.open('rb') as handle:
        ordering = IndexUnpickler(handle).load()['indices']
    assert len(ordering) == len(set(ordering)) == 50000
    assert all(type(i) is int and 0 <= i < 50000 for i in ordering)
    for count in (500, 1000):
        prefix = ROOT / spec['verified_prefixes'][str(count)]['path']
        assert reference.file_hash(prefix) == spec['verified_prefixes'][str(count)]['sha256']
        with prefix.open('rb') as handle:
            assert IndexUnpickler(handle).load() == ordering[:count]
    return dict(complete=True, source=source, percentage=0.2,
                requested_count=100, actual_count=100, indices=ordering[:100],
                origin='prefix_of_authors_released_full_ordering', upstream=spec)


def evaluate(manifest, source, percentage, selection):
    cfg = reference.protocol()
    train_set = reference.RawDataset(manifest['data_root'], source, True)
    test_set = reference.RawDataset(manifest['data_root'], source, False)
    indices = selection['indices']
    assert selection['complete'] and selection['source'] == source and selection['percentage'] == percentage
    assert 1 < len(indices) == len(set(indices)) == selection['actual_count'] <= selection['requested_count']
    assert all(type(i) is int and 0 <= i < len(train_set) for i in indices)
    if 'source_fingerprint' in selection:
        assert selection['source_fingerprint'] == train_set.fingerprint
    signature = dict(experiment='sas_authors_recipe', source=source, subset=percentage,
                     model='ResNet18', ssl='simclr', selection=selection,
                     source_fingerprint=train_set.fingerprint, protocol=cfg,
                     code_hashes=manifest['code_hashes'], ssl_seed=0,
                     probe_seeds=list(range(5)), labels=[1, 5])
    identity = dict(signature=reference.digest_json(signature), seed=0,
                    selection_method='sas_authors', selection=selection)
    config = dict(signature=signature, identity=identity)
    folder = HERE / 'results' / source / 'sas_authors' / f'pt{percentage:g}/simclr/ssl_seed_000'
    with reference.exclusive_run(folder):
        if (folder / 'config.json').is_file():
            assert reference.read(folder / 'config.json') == config
        reference.atomic_json(folder / 'config.json', config)
        reference.train_ssl.print = reference.progress
        checkpoint, trained = reference.train_ssl.train(
            reference.torch.utils.data.Subset(train_set, indices), cfg, source,
            'ResNet18', 'simclr', 0, folder, identity, workers=0, device='cuda')
        assert trained['epochs'] == 2000
        reference.atomic_json(folder / 'ssl_complete.json', dict(complete=True, identity=identity,
                              epochs=2000, encoder_sha256=trained['encoder_sha256']))
        for labels in (1, 5):
            scores = []
            for seed in range(5):
                path = folder / 'evaluations' / f'{source}_lbl{labels}_probe{seed}.json'
                result = reference.linear_probe(checkpoint, identity, cfg, source, source,
                                               train_set, test_set, labels, seed, path,
                                               workers=0, device='cuda')
                assert result['complete'] and result['encoder_unchanged']
                assert result['encoder_sha256'] == trained['encoder_sha256']
                assert math.isfinite(result['accuracy_percent'])
                scores.append(result['accuracy_percent'])
                reference.atomic_json(folder / f'summary_{source}_lbl{labels}.json',
                    dict(complete=len(scores) == 5, source=source, method='sas_authors',
                         subset=percentage, actual_count=len(indices), labels=labels,
                         ssl_epochs=2000, linear_epochs=100, completed_probes=len(scores),
                         expected_probes=5, mean_percent=statistics.mean(scores), std_ddof=1,
                         std_percent=statistics.stdev(scores) if len(scores) > 1 else None,
                         samples=scores))
        reference.atomic_json(folder / 'complete.json', dict(complete=True, identity=identity))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('source', choices=('CIFAR10', 'CIFAR100', 'TinyImageNet'))
    parser.add_argument('percentage', type=float, choices=(0.2, 1, 2))
    args = parser.parse_args()
    assert os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_JOB_ACCOUNT') == 'aip-yiweilu'
    assert reference.torch.cuda.is_available() and reference.torch.cuda.device_count() == 1
    reference.torch.set_num_threads(1)
    reference.torch.set_num_interop_threads(1)
    reference.seed_all(0)
    manifest = reference.read(HERE / 'manifest.json')
    for relative, checksum in manifest['code_hashes'].items():
        assert reference.file_hash(ROOT / relative) == checksum, 'Code changed: ' + relative
    if args.source == 'TinyImageNet':
        selection = reference.read(HERE / 'selections/TinyImageNet' / f'pt{args.percentage:g}.json')
    else:
        assert args.percentage == 0.2
        selection = cifar_indices(manifest, args.source)
        reference.atomic_json(HERE / 'selections' / args.source / 'pt0.2.json', selection)
    print(f'{args.source}: {selection["actual_count"]} author-selected images; 2k SimCLR then 5 probes per label budget', flush=True)
    evaluate(manifest, args.source, args.percentage, selection)


if __name__ == '__main__':
    main()
