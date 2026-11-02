"""Change only CIFAR10 table initialization; reuse its frozen training/evaluation code."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import types

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
BASE = ROOT / 'Evaluate_benchamrk/mipa_cvpr_color_crop'
IIC = ROOT / 'OT-SSL-DD/ours_syntehtic_iic_main_aux_k4_10k/CIFAR10/ResNet18BN'
sys.path.insert(0, str(ROOT / 'OT-SSL-DD'))
import eval as reference_eval


def read(path):
    return json.loads(Path(path).read_text())


def frozen_module(path, name, replacements):
    source = Path(path).read_text()
    for old, new in replacements:
        assert source.count(old) == 1, 'Frozen source no longer has the expected initialization hook'
        source = source.replace(old, new)
    module = types.ModuleType(name)
    module.__file__ = str(path)
    exec(compile(source, str(path), 'exec'), module.__dict__)
    return module


def tensor_hash(images):
    return hashlib.sha256(images.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def self_check(data):
    features = torch.tensor([[0., 0.], [0., 1.], [0., 2.], [20., 0.], [20., 1.], [20., 2.]], device='cuda')
    indices = data.kmeans_indices(features, 2, 0, 100, 2)
    assert len(indices) == len(set(indices)) == 2 and {index // 3 for index in indices} == {0, 1}
    assert indices == data.kmeans_indices(features, 2, 0, 100, 2)
    duplicates = data.kmeans_indices(torch.zeros(6, 2, device='cuda'), 4, 0, 100, 2)
    assert len(duplicates) == len(set(duplicates)) == 4
    print('PASS: existing K-means selects deterministic distinct exemplars, including tied features', flush=True)


@torch.no_grad()
def choose_images(method, data, common, dataset, count, device, seed, variant, run, job):
    path = run / 'initialization.json'
    fingerprint = dataset.base.fingerprint
    rng, numpy_rng = common.rng_state(), np.random.get_state()
    try:
        if path.exists():
            selection = read(path)
        elif variant == 'iic':
            source = IIC / ('pt%g' % job['generation_args']['percentage']) / 'initialization.json'
            previous = read(source)
            assert previous['method'] == 'iic_main_aux' and previous['epoch'] == 150
            assert previous['source_fingerprint'] == fingerprint
            assert common.file_hash(previous['clusters_file']) == previous['clusters_file_sha256']
            selection = dict(previous, source_selection=str(source), source_selection_sha256=common.file_hash(source))
        else:
            import utils
            network = utils.get_network('ResNet18BN', 3, 10, (32, 32), seed=seed).to(device)
            embed = method.frozen_embedder(network)
            stream = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False, num_workers=0,
                                                 generator=torch.Generator().manual_seed(seed))
            features = torch.cat([embed(method._images_from_batch(batch).to(device)) for batch in stream])
            assert features.shape[0] == len(dataset) and torch.isfinite(features).all()
            indices = data.kmeans_indices(features, count, seed, 100, 1024)
            selection = dict(method='kmeans100', seed=seed, feature_model='random_ResNet18BN', feature_seed=seed,
                             feature_normalization='none', feature_batch=128, kmeans_iterations=100,
                             kmeans_stopping='100 Lloyd updates maximum, or unchanged centers',
                             centers_seed=seed + 1000, exemplar_rule='unique nearest real image per center',
                             source_fingerprint=fingerprint, indices=indices)
            del network, embed, features
        assert selection['source_fingerprint'] == fingerprint
        assert selection['method'] == ('iic_main_aux' if variant == 'iic' else 'kmeans100')
        indices = selection['indices']
        assert len(indices) == len(set(indices)) == count
        assert all(isinstance(i, int) and 0 <= i < len(dataset) for i in indices)
        images = torch.stack([method._images_from_batch(dataset[i]) for i in indices])
        digest = tensor_hash(images)
        if path.exists():
            assert selection['initial_images_sha256'] == digest
        selection.update(initial_images_sha256=digest, count=count, labels_used=False)
        common.atomic_json(path, selection)
        print('INITIALIZATION: %s, %d distinct real images, sha256=%s' % (variant, count, digest), flush=True)
        return images.to(device)
    finally:
        common.restore_rng(rng)
        np.random.set_state(numpy_rng)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('variant', choices=('kmeans100', 'iic'))
    parser.add_argument('percentage', type=float, choices=(.2, 1., 2.))
    parser.add_argument('stage', choices=('generation', 'evaluation'))
    args = parser.parse_args()
    assert os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_JOB_ACCOUNT') == 'aip-yiweilu'
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    reference = reference_eval.reference_modules()
    common, data = reference['common'], reference['data']
    baseline_manifest = BASE / 'snapshot' / ('CIFAR10_pt%s.json' % ('0p2' if args.percentage == .2 else '%g' % args.percentage))
    job = read(baseline_manifest)
    for path, expected in job['code_hashes'].items():
        assert common.file_hash(path) == expected, 'Baseline code changed: ' + path
    assert common.file_hash(job['teacher']) == job['teacher_sha256']
    run = HERE / 'runs' / args.variant / ('pt%g' % args.percentage)
    job.update(name='CIFAR10_%s_pt%g' % (args.variant, args.percentage), account='aip-yiweilu',
               initialization=args.variant, baseline_manifest=str(baseline_manifest),
               synthetic_dir=str(run / 'synthetic'), evaluation_dir=str(run / 'evaluation'))
    job['code_hashes'][str(Path(__file__).resolve())] = common.file_hash(__file__)
    if args.variant == 'iic':
        source = IIC / ('pt%g' % args.percentage) / 'initialization.json'
        selection = read(source)
        assert selection['epoch'] == 150 and selection['method'] == 'iic_main_aux'
        job['code_hashes'][str(source)] = common.file_hash(source)
        job['code_hashes'][selection['clusters_file']] = selection['clusters_file_sha256']
    manifest = run / 'job.json'
    checkpoint = run / 'synthetic' / ('res_IIC_CIFAR10_ResNet18BN_%gpercent_final.pt' % args.percentage)
    if manifest.exists():
        assert read(manifest) == job, 'Ablation inputs changed; preserve this run and use a new directory'

    if args.stage == 'evaluation':
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        assert saved['initialization'] == read(run / 'initialization.json') and saved['init'] == args.variant
        assert tensor_hash(saved['initial_data']) == saved['initialization']['initial_images_sha256']
        del saved
        worker = frozen_module(BASE / 'snapshot/worker.py', 'matched_initialization_evaluation', [
            ("saved['init'] == 'random'", "saved['init'] == %r" % args.variant)])
        sys.argv = [worker.__file__, str(manifest), 'evaluation']
        worker.main()
        return

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    self_check(data)
    method = frozen_module(BASE / 'snapshot/method1_cvpr.py', 'matched_initialization_generation', [
        ("saved.get('init') == 'random'", "saved.get('init') == %r" % args.variant),
        ("'init': 'random'", "'init': %r" % args.variant)])
    with common.exclusive_run(run):
        common.atomic_json(manifest, job)
        if checkpoint.exists():
            previous = torch.load(checkpoint, map_location='cpu', weights_only=True)
            assert previous['initialization'] == read(run / 'initialization.json')
            assert tensor_hash(previous['initial_data']) == previous['initialization']['initial_images_sha256']
            del previous
        original_save = common.atomic_save

        def initialize(dataset, count, device, seed):
            return choose_images(method, data, common, dataset, count, device, seed, args.variant, run, job)

        def save(path, payload):
            if Path(path) == checkpoint:
                payload.update(initialization=read(run / 'initialization.json'),
                               initialization_driver_sha256=job['code_hashes'][str(Path(__file__).resolve())])
            original_save(path, payload)

        def progress(*values, **kwargs):
            if values and isinstance(values[0], str):
                match = re.search(r'iter = (\d+)', values[0])
                if match and int(match[1]) % 200:
                    return
            print(*values, **kwargs)

        method.random_initialize = initialize
        method.print = progress
        common.atomic_save = save
        sys.argv = [method.__file__]
        for key, value in job['generation_args'].items():
            if isinstance(value, bool):
                if value:
                    sys.argv.append('--' + key)
            else:
                sys.argv += ['--' + key, str(value)]
        sys.argv += ['--device', 'cuda', '--data_path', job['data_root'], '--save_path', job['synthetic_dir'], '--resume', '--save_every', '500']
        print('MATCHED TABLE CONFIG: ' + json.dumps(job['generation_args']), flush=True)
        try:
            method.main()
        finally:
            common.atomic_save = original_save
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        assert saved['complete'] and saved['iteration'] == job['generation_args']['Iteration']
        assert saved['run_args'] == job['generation_args'] and saved['init'] == args.variant
        assert tensor_hash(saved['initial_data']) == saved['initialization']['initial_images_sha256']
        assert common.file_hash(job['teacher']) == job['teacher_sha256']
        common.atomic_json(run / 'generation_complete.json', dict(complete=True, initialization=args.variant,
                           iteration=saved['iteration'], checkpoint=str(checkpoint), checkpoint_sha256=common.file_hash(checkpoint)))


if __name__ == '__main__':
    main()
