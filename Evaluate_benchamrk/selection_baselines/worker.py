"""Select real subsets, then reuse the main-table SSL and frozen-probe code."""
import argparse
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
BENCH, ROOT = HERE.parent, HERE.parent.parent
sys.path.insert(0, str(BENCH))
from common import atomic_json, atomic_save, code_hashes, digest_json, exclusive_run, file_hash, load_encoder, seed_all, state_hash
from data import RawDataset, ImageView, loader, probe_transform
from evaluate import linear_probe
import train_ssl
from selection import cluster_features, sas_subsets, estimate_radius, radius_graph, probcover_order


def read(path):
    return json.loads(Path(path).read_text())


def protocol():
    value = read(BENCH / 'protocol.json')
    value['ssl'].update(subset_epochs=2000, checkpoint_every_epochs=100)
    value['subset_seeds'] = [0]
    assert value['evaluation']['epochs'] == 100
    return value


def progress(*values, **kwargs):
    if len(values) == 1 and isinstance(values[0], str) and values[0].startswith('SSL ') and kwargs.get('file') is None:
        prefix, text = values[0].split(': ', 1)
        stats = json.loads(text)
        if stats['epoch'] != 1 and stats['epoch'] % 100:
            return
        values = (prefix + ': ' + text,)
    print(*values, **kwargs)


def select(manifest, source):
    spec, cfg = manifest['datasets'][source], protocol()
    base = RawDataset(manifest['data_root'], source, True)
    folder = HERE / 'selections' / source
    expected = dict(source=source, source_fingerprint=base.fingerprint, teacher=spec['teacher'],
                    teacher_sha256=spec['teacher_sha256'], teacher_seed=spec['teacher_seed'],
                    count=len(base), classes=base.classes, seed=0, ground_truth_labels_used=False,
                    feature_transform='reference_test_transform', feature_normalization='l2',
                    clustering='euclidean_Lloyd_1000_iterations_random_init_seed0',
                    probcover_purity_alpha=0.95, code_hashes=manifest['code_hashes'])
    with exclusive_run(folder):
        if (folder / 'complete.json').is_file():
            marker = read(folder / 'complete.json')
            assert marker['complete'] and marker['expected'] == expected
            for relative, checksum in marker['files'].items():
                assert file_hash(folder / relative) == checksum
            print('SKIP completed selections:', source, flush=True)
            return
        started = time.monotonic()
        cache = folder / 'features.pt'
        if cache.is_file():
            saved = torch.load(cache, map_location='cpu', weights_only=True)
            assert saved['expected'] == expected
            features, groups = saved['features'].cuda(), saved['groups'].numpy()
            del saved
        else:
            teacher_path = ROOT / spec['teacher']
            assert file_hash(teacher_path) == spec['teacher_sha256']
            teacher, dimension, saved = load_encoder(teacher_path)
            marker = read(teacher_path.parent / 'ssl_complete.json')
            teacher_config = read(teacher_path.parent / 'config.json')
            assert saved['source'] == source and saved['seed'] == spec['teacher_seed'] and saved['epochs'] == 1000
            assert saved['method'] == 'simclr' and saved['identity'] == marker['identity'] == teacher_config['identity']
            assert teacher_config['signature']['source_fingerprint'] == base.fingerprint
            stream = loader(ImageView(base, probe_transform(cfg, source, False)), 256, 0, 0)
            chunks = []
            with torch.no_grad():
                for index, (images, _unused_labels) in enumerate(stream):
                    chunks.append(F.normalize(teacher(images.cuda()), dim=1).cpu())
                    if index % 50 == 0:
                        print(f'{source} feature batches: {index + 1}/{len(stream)}', flush=True)
            assert state_hash(teacher) == saved['encoder_sha256']
            features = torch.cat(chunks).cuda()
            assert features.shape == (len(base), dimension) and torch.isfinite(features).all()
            del teacher, chunks, saved
            torch.cuda.empty_cache()
            groups, iterations = cluster_features(features, base.classes)
            print(f'{source}: clustering finished after {iterations} iterations', flush=True)
            atomic_save(cache, dict(expected=expected, features=features.cpu(), groups=torch.as_tensor(groups)))
        budgets = [int(len(base) * percentage / 100) for percentage in (0.2, 1, 2)]
        sas = sas_subsets(features, groups, budgets)
        radius, purity = estimate_radius(features, groups)
        print(f'{source}: ProbCover delta={radius:.8g}, cluster-ball purity={purity:.6f}', flush=True)
        graph = radius_graph(features, radius)
        order = probcover_order(graph, max(budgets))
        files = {}
        for method in ('sas', 'probcover'):
            for index, (percentage, budget) in enumerate(zip((0.2, 1, 2), budgets)):
                indices = sas[index] if method == 'sas' else order[:budget]
                assert len(indices) == len(set(indices)) == budget and min(indices) >= 0 and max(indices) < len(base)
                result = dict(complete=True, expected=expected, method=method, percentage=percentage,
                              count=budget, indices=indices, radius=radius if method == 'probcover' else None,
                              estimated_purity=purity if method == 'probcover' else None,
                              seconds=time.monotonic() - started)
                relative = f'{method}_pt{percentage:g}.json'
                atomic_json(folder / relative, result)
                files[relative] = file_hash(folder / relative)
        atomic_json(folder / 'complete.json', dict(complete=True, expected=expected, files=files,
                                                   seconds=time.monotonic() - started))


def evaluate(manifest, source, method, percentage):
    cfg = protocol()
    train_set = RawDataset(manifest['data_root'], source, True)
    test_set = RawDataset(manifest['data_root'], source, False)
    selections = HERE / 'selections' / source
    marker = read(selections / 'complete.json')
    selection_path = selections / f'{method}_pt{percentage:g}.json'
    assert marker['complete'] and marker['files'][selection_path.name] == file_hash(selection_path)
    selection = read(selection_path)
    assert selection['complete'] and selection['expected'] == marker['expected']
    assert selection['expected']['source_fingerprint'] == train_set.fingerprint
    assert selection['expected']['code_hashes'] == manifest['code_hashes']
    indices = selection['indices']
    assert len(indices) == len(set(indices)) == int(len(train_set) * percentage / 100)
    assert all(isinstance(i, int) and 0 <= i < len(train_set) for i in indices)
    signature = dict(experiment='sas_probcover_main_table', source=source, subset=percentage,
                     model='ResNet18', ssl='simclr', method=method, selection_sha256=file_hash(selection_path),
                     source_fingerprint=train_set.fingerprint, protocol=cfg, code_hashes=manifest['code_hashes'],
                     ssl_seed=0, probe_seeds=list(range(5)), labels=[1, 5])
    identity = dict(signature=digest_json(signature), seed=0, selection_method=method,
                    selection=dict(indices=indices, metadata=selection['expected']))
    config = dict(signature=signature, identity=identity)
    folder = HERE / 'results' / source / method / f'pt{percentage:g}' / 'simclr/ssl_seed_000'
    with exclusive_run(folder):
        if (folder / 'config.json').is_file():
            assert read(folder / 'config.json') == config, 'Refusing to mix evaluation configurations'
        atomic_json(folder / 'config.json', config)
        selected = torch.utils.data.Subset(train_set, indices)
        train_ssl.print = progress
        checkpoint, trained = train_ssl.train(selected, cfg, source, 'ResNet18', 'simclr', 0,
                                               folder, identity, workers=0, device='cuda')
        assert trained['epochs'] == 2000
        atomic_json(folder / 'ssl_complete.json', dict(complete=True, identity=identity, epochs=2000,
                    encoder_sha256=trained['encoder_sha256'], checkpoint_file_sha256=file_hash(checkpoint)))
        for labels in (1, 5):
            samples = []
            for seed in range(5):
                path = folder / 'evaluations' / f'{source}_lbl{labels}_probe{seed}.json'
                result = linear_probe(checkpoint, identity, cfg, source, source, train_set, test_set,
                                      labels, seed, path, workers=0, device='cuda')
                assert result['complete'] and result['encoder_unchanged'] and result['encoder_sha256'] == trained['encoder_sha256']
                assert math.isfinite(result['accuracy_percent'])
                samples.append(dict(ssl_seed=0, probe_seed=seed, accuracy_percent=result['accuracy_percent'], result_file=str(path)))
                scores = [sample['accuracy_percent'] for sample in samples]
                summary = dict(complete=len(samples) == 5, source=source, target=source, method=method,
                               subset=percentage, ssl_epochs=2000, linear_epochs=100, labels=labels,
                               completed_probes=len(samples), expected_probes=5, std_ddof=1, samples=samples,
                               mean_percent=statistics.mean(scores), std_percent=statistics.stdev(scores) if len(scores) > 1 else None)
                atomic_json(folder / f'summary_{source}_lbl{labels}.json', summary)
            print(f'COMPLETE {source} {method} pt{percentage:g} lbl{labels}: {summary["mean_percent"]:.2f} +/- {summary["std_percent"]:.2f}', flush=True)
        atomic_json(folder / 'complete.json', dict(complete=True, identity=identity, labels=[1, 5], probe_seeds=list(range(5))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('manifest', type=Path)
    parser.add_argument('stage', choices=('select', 'evaluate'))
    parser.add_argument('source', choices=('CIFAR10', 'CIFAR100', 'TinyImageNet'))
    parser.add_argument('method', nargs='?', choices=('sas', 'probcover'))
    parser.add_argument('percentage', nargs='?', type=float, choices=(0.2, 1, 2))
    args = parser.parse_args()
    assert os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_JOB_ACCOUNT') == 'aip-yiweilu'
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    seed_all(0)
    manifest = read(args.manifest)
    for relative, checksum in manifest['code_hashes'].items():
        assert file_hash(ROOT / relative) == checksum, 'Code changed after submission: ' + relative
    if args.stage == 'select':
        select(manifest, args.source)
    else:
        assert args.method is not None and args.percentage is not None
        evaluate(manifest, args.source, args.method, args.percentage)


if __name__ == '__main__':
    main()
