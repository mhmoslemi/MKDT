"""MIPA initialized by balanced main clusters and diverse auxiliary assignments."""
import argparse
from collections import deque
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import sys

import torch

HERE = Path(__file__).resolve().parent
BENCH, ROOT = HERE.parent, HERE.parent.parent
OT = ROOT / 'OT-SSL-DD'
sys.path.insert(0, str(BENCH))
sys.path.insert(0, str(OT))
import eval as reference_eval
from mipa_multilayer.worker import progress

ITERATIONS = {0.2: 10000, 1: 8000, 2: 6000}


def balanced_main_aux(main, auxiliary, confidence, count):
    """Cycle over main groups, then auxiliary intersections, without replacement."""
    size = len(main)
    if not 0 < count <= size or len(auxiliary) != size or len(confidence) != size:
        raise ValueError('Invalid selection size or clustering vectors')
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in confidence):
        raise ValueError('Cluster confidences must be finite probabilities')
    groups = {}
    # Rank representatives by main confidence; stable index ties make selection deterministic.
    for index in sorted(range(size), key=lambda i: (-confidence[i], i)):
        groups.setdefault(main[index], {}).setdefault(auxiliary[index], deque()).append(index)
    active = deque(deque(groups[cluster].values()) for cluster in sorted(groups))
    selected = []
    while len(selected) < count:
        fine_groups = active.popleft()
        members = fine_groups.popleft()
        selected.append(members.popleft())
        if members:
            fine_groups.append(members)
        if fine_groups:
            active.append(fine_groups)
    assert len(selected) == count == len(set(selected))
    return selected


def check_selection():
    main = [0] * 6 + [1] * 6
    auxiliary = [0, 0, 0, 1, 1, 2] + [0, 0, 1, 1, 2, 2]
    confidence = [.99, .98, .97, .8, .7, .6, .95, .94, .85, .84, .75, .74]
    small = balanced_main_aux(main, auxiliary, confidence, 6)
    assert small == [0, 6, 3, 8, 5, 10]
    assert balanced_main_aux(main, auxiliary, confidence, 10)[:6] == small
    assert sorted(balanced_main_aux(main, auxiliary, confidence, 12)) == list(range(12))
    # Empty main IDs and exhausted small groups must not cause duplicates or missing images.
    assert balanced_main_aux([0, 2, 2], [4, 4, 5], [.9, .8, .7], 3) == [0, 1, 2]


def read(path):
    return json.loads(Path(path).read_text())


def load_clustering(run, common):
    config, completed = read(run / 'config.json'), read(run / 'complete.json')
    assert config['dataset'] in ('CIFAR10', 'CIFAR100')
    assert completed['complete'] and completed['epochs'] == config['epochs'] == 150
    assert completed['config_sha256'] == common.digest_json(config)
    assert completed['teacher_unchanged']
    path = run / 'clusters.pt'
    sha = common.file_hash(path)
    assert sha == completed['outputs']['clusters.pt'], 'Saved cluster assignments changed'
    saved = torch.load(path, map_location='cpu', weights_only=True)
    assert saved['kind'] == 'frozen_teacher_iic_clusters' and saved['complete'] and saved['teacher_unchanged']
    assert saved['config'] == config
    assert torch.equal(saved['image_indices'], torch.arange(50000)), 'Unexpected training-image order'
    for name in ('main', 'overcluster'):
        group = saved['clusters'][name]
        assert group['subhead'] == completed['selected_subheads'][name]
        assignment, confidence = group['assignments'], group['confidence']
        assert assignment.shape == confidence.shape == (50000,)
        assert assignment.dtype == torch.int64
        assert assignment.min() >= 0 and assignment.max() < config['cluster_counts'][name]
        assert torch.isfinite(confidence).all() and ((confidence >= 0) & (confidence <= 1)).all()
        assert torch.equal(torch.bincount(assignment, minlength=config['cluster_counts'][name]), group['counts'])
    assert common.file_hash(path) == sha, 'Cluster file changed while reading'
    return config, saved['clusters'], sha


def distill(args, common, raw, config, clusters, cluster_sha, indices, percentage):
    source = config['dataset']
    count = int(len(raw) * percentage / 100)
    chosen = indices[:count]
    main = clusters['main']['assignments']
    auxiliary = clusters['overcluster']['assignments']
    # Group the saved image grid by main cluster while retaining within-cluster selection order.
    chosen = sorted(chosen, key=lambda index: int(main[index]))
    selection = dict(method='iic_main_aux', dataset=source, percentage=percentage,
        count=count, epoch=150, indices=chosen, source_fingerprint=raw.fingerprint,
        clusters_file=str(args.iic_run / 'clusters.pt'), clusters_file_sha256=cluster_sha,
        selected_subheads={name: value['subhead'] for name, value in clusters.items()},
        ranking='main-head confidence descending, image index as tie-breaker',
        sampling='round-robin main clusters, then occupied (main, auxiliary) intersections',
        main_counts=torch.bincount(main[chosen], minlength=config['cluster_counts']['main']).tolist(),
        auxiliary_groups_per_main={str(cluster): len(set(auxiliary[index].item() for index in chosen
            if main[index].item() == cluster)) for cluster in main.unique().tolist()})
    output = args.save_path / source / 'ResNet18BN' / ('pt%g' % percentage)
    result_path = output / ('res_IIC_%s_ResNet18BN_%gpercent_final.pt' % (source, percentage))
    with common.exclusive_run(output):
        provenance = output / 'pipeline_config.json'
        if result_path.exists() and not provenance.exists():
            raise ValueError('Existing synthetic data has no matching IIC provenance; use a new save_path')
        snapshot = output / 'method1.py'
        if not snapshot.exists():
            code = (OT / 'method1.py').read_text()
            original = "choices=['random', 'kmeans']"
            assert code.count(original) == 1, 'Cannot add the IIC initialization choice to this method1 version'
            snapshot.write_text(code.replace(original, "choices=['random', 'kmeans', 'iic']"))
        teacher = Path(config['teacher']['path']).resolve()
        assert teacher == (BENCH / 'full_ssl_checkpoints_1000ep_3seeds' / source /
                           'ResNet18/full/pt100/simclr/seed_000/encoder.pt').resolve()
        assert common.file_hash(teacher) == config['teacher']['file_sha256'], 'Teacher weights changed'
        assert common.file_hash(teacher.parent / 'ssl_complete.json') == config['teacher']['marker_sha256']
        generation = dict(dataset=source, percentage=percentage, model='ResNet18BN', seed=0,
            Iteration=args.Iteration or ITERATIONS[percentage], lr_img=0.001, batch_real=128, temperature=0.2,
            disp_weight=0.0, pair_weight=0.0, num_random_networks=1, random_models='ResNet18BN',
            dsa=True, distill_aug_strategy='color_crop_cutout_flip_scale_rotate', distill_aug_mode='M',
            teacher_source='pretrained', teacher_seed=[0], teacher_ssl='simclr',
            teacher_path=str(teacher.parents[6]), feature_batch=128)
        if args.cosine_views > 1:
            generation['cosine_views'] = args.cosine_views
        files = [Path(__file__), snapshot, OT / 'eval.py', OT / 'utils.py', OT / 'networks.py',
                 OT / 'dm_dsa.py', OT / 'dd_teachers.py', BENCH / 'mipa_multilayer/worker.py']
        hashes = dict(common.code_hashes(), **{str(path.relative_to(ROOT)): common.file_hash(path) for path in files})
        contract = dict(selection=selection, generation_args=generation, code_hashes=hashes,
                        teacher=config['teacher'])
        if provenance.exists() and read(provenance) != contract:
            raise ValueError('Initialization or code changed; saved data preserved. Use a new save_path.')
        common.atomic_json(provenance, contract)
        common.atomic_json(output / 'initialization.json', selection)
        spec = importlib.util.spec_from_file_location('main_aux_method', snapshot)
        method = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(method)
        method.print = progress

        def initialize(dataset, requested, device, *unused, **keywords):
            assert requested == count and dataset.base.fingerprint == raw.fingerprint
            print('Initializing %d images from saved IIC main + auxiliary assignments: %s' %
                  (count, selection['main_counts']), flush=True)
            return torch.stack([method._images_from_batch(dataset[index]) for index in chosen]).to(device)

        method.deepcluster_initialize = initialize
        original_save, previous_argv = common.atomic_save, sys.argv

        def save(path, value):
            if Path(path).resolve() == result_path:
                value = dict(value, initialization=selection)
            original_save(path, value)

        common.atomic_save = save
        argv = [str(snapshot)]
        for key, value in generation.items():
            if isinstance(value, bool):
                argv.append('--' + (key if value else 'no-' + key))
            else:
                argv += ['--' + key] + [str(item) for item in (value if isinstance(value, list) else [value])]
        sys.argv = argv + ['--device', 'cuda', '--data_path', str(args.data_path), '--save_path', str(output),
            '--init', 'iic', '--similarity', 'cosine', '--feature_layers', 'last',
            '--distill_aug_backend', 'dsa', '--resume', '--save_every', '500', '--num_eval', '0']
        try:
            method.main()
        finally:
            common.atomic_save, sys.argv = original_save, previous_argv
        saved = torch.load(result_path, map_location='cpu', weights_only=True, mmap=True)
        assert saved['complete'] and saved['iteration'] == generation['Iteration']
        assert saved['initialization'] == selection and saved['init'] == 'iic' and saved['run_args'] == generation
        assert saved['iic_params']['cosine_views'] == args.cosine_views
        assert saved['data'].shape == (count, 3, 32, 32) and torch.isfinite(saved['data']).all()
        normalized = reference_eval.reference_modules()['data'].ImageView(raw, method.transforms.Compose([
            method.transforms.ToTensor(), method.transforms.Normalize(*saved['normalization'])]))
        assert torch.equal(saved['initial_data'], torch.stack([normalized[index][0] for index in chosen]))
        assert common.file_hash(teacher) == config['teacher']['file_sha256'], 'Teacher weights changed'
        common.atomic_json(output / 'generation_complete.json', dict(complete=True,
            iteration=saved['iteration'], cosine_views=args.cosine_views,
            checkpoint_file_sha256=common.file_hash(result_path),
            job_file_sha256=common.file_hash(args.job) if args.job else None))
        print('DONE: %s' % result_path, flush=True)


def evaluate(args, source, percentage, reference):
    common, data = reference['common'], reference['data']
    output = args.save_path / source / 'ResNet18BN' / ('pt%g' % percentage)
    contract = read(output / 'pipeline_config.json')
    for name, sha in contract['code_hashes'].items():
        assert common.file_hash(ROOT / name) == sha, 'Code changed: ' + name
    path = output / ('res_IIC_%s_ResNet18BN_%gpercent_final.pt' % (source, percentage))
    saved = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    expected_iterations = args.Iteration or ITERATIONS[percentage]
    assert saved['complete'] and saved['iteration'] == expected_iterations
    assert saved['run_args'] == contract['generation_args'] and saved['initialization'] == contract['selection']
    assert saved['iic_params']['cosine_views'] == args.cosine_views
    assert saved['iic_params']['similarity'] == 'cosine' and saved['iic_params']['feature_layers'] == 'last'
    assert not saved['iic_params']['assignment_filter']
    assert saved['normalization'] == reference_eval.SYNTHETIC_NORMALIZATION[source]
    images = saved['data'].detach().cpu().contiguous()
    assert images.shape == (int(50000 * percentage / 100), 3, 32, 32) and torch.isfinite(images).all()
    mean, std = (torch.tensor(value).view(1, 3, 1, 1) for value in saved['normalization'])
    synthetic = reference_eval.SyntheticImages((images * std + mean).clamp(0, 1).contiguous())
    protocol = read(BENCH / 'protocol.json')
    protocol['ssl'].update(subset_epochs=2000, checkpoint_every_epochs=500)
    protocol['subset_seeds'] = [0]
    assert protocol['evaluation']['epochs'] == 100 and protocol['evaluation']['lr'] == 0.1
    assert protocol['ssl']['batch_size'] == protocol['evaluation']['batch_size'] == 256
    train_set, test_set = (data.RawDataset(args.data_path, source, training) for training in (True, False))
    assert train_set.fingerprint == contract['selection']['source_fingerprint']
    synthetic_sha = common.file_hash(path)
    signature = dict(experiment='iic_main_aux_mipa', source=source, model='ResNet18', subset=percentage,
        ssl='simclr', protocol=protocol, generation=contract, source_fingerprint=train_set.fingerprint,
        synthetic=dict(path=str(path), file_sha256=synthetic_sha, normalization=saved['normalization'],
                       pixel_policy='denormalize_clamp_then_PIL_uint8'),
        ssl_seeds=[0], probe_seeds=list(range(5)), code_hashes=common.code_hashes())
    identity = dict(signature=common.digest_json(signature), seed=0, selection_method='ours')
    config = dict(signature=signature, identity=identity)
    folder = output / 'evaluation/simclr/ssl_seed_000'
    del saved, images
    with common.exclusive_run(folder):
        config_file = folder / 'config.json'
        if config_file.exists() and read(config_file) != config:
            raise ValueError('Evaluation inputs changed; saved results preserved')
        common.atomic_json(config_file, config)
        reference['train_ssl'].print = progress
        checkpoint, trained = reference['train_ssl'].train(synthetic, protocol, source, 'ResNet18', 'simclr',
            0, folder, identity, workers=0, device='cuda')
        assert trained['epochs'] == 2000
        common.atomic_json(folder / 'ssl_complete.json', dict(complete=True, identity=identity,
            epochs=2000, encoder_sha256=trained['encoder_sha256'], checkpoint_file_sha256=common.file_hash(checkpoint)))
        for labels in (1, 5):
            samples = []
            for seed in range(5):
                result_file = folder / 'evaluations' / ('%s_lbl%d_probe%d.json' % (source, labels, seed))
                result = reference['evaluate'].linear_probe(checkpoint, identity, protocol, source, source,
                    train_set, test_set, labels, seed, result_file, workers=0, device='cuda')
                assert result['complete'] and result['encoder_unchanged']
                assert result['encoder_sha256'] == trained['encoder_sha256']
                score = result['accuracy_percent']
                assert math.isfinite(score) and 0 <= score <= 100
                samples.append(dict(ssl_seed=0, probe_seed=seed, accuracy_percent=score, result_file=str(result_file)))
            scores = [sample['accuracy_percent'] for sample in samples]
            common.atomic_json(folder / ('summary_%s_lbl%d.json' % (source, labels)), dict(complete=True,
                source=source, target=source, subset=percentage, model='ResNet18', ssl='simclr', labels=labels,
                initialization='iic_main_aux_epoch150', cosine_views=args.cosine_views,
                distillation_iterations=expected_iterations, ssl_epochs=2000, linear_epochs=100,
                ssl_seeds=[0], probe_seeds=list(range(5)), completed_probes=5, expected_probes=5,
                std_ddof=1, samples=samples, mean_percent=statistics.mean(scores), std_percent=statistics.stdev(scores)))
            print('COMPLETE CELL: %s pt%g lbl%d: %.2f +/- %.2f' %
                  (source, percentage, labels, statistics.mean(scores), statistics.stdev(scores)), flush=True)
        assert common.file_hash(path) == synthetic_sha
        common.atomic_json(folder / 'complete.json', dict(complete=True, identity=identity,
            synthetic_file_sha256=synthetic_sha, encoder_file_sha256=common.file_hash(checkpoint),
            ssl_epochs=2000, linear_epochs=100, labels=[1, 5], probe_seeds=list(range(5)),
            job_file_sha256=common.file_hash(args.job) if args.job else None))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--iic_run', type=Path, default=HERE / 'runs/CIFAR10_ResNet18_e150_lu_killarney')
    parser.add_argument('--percentages', type=float, nargs='+', choices=list(ITERATIONS), default=list(ITERATIONS))
    parser.add_argument('--data_path', type=Path, default=ROOT.parent / 'data')
    parser.add_argument('--save_path', type=Path, default=OT / 'ours_syntehtic_iic_main_aux')
    parser.add_argument('--Iteration', type=int, default=None, help='Override the distillation budget for every selected size')
    parser.add_argument('--cosine_views', type=int, default=1)
    parser.add_argument('--stage', choices=['generate', 'evaluate'], default='generate')
    parser.add_argument('--job', type=Path, default=None, help='Pinned submission manifest')
    args = parser.parse_args()
    if args.job:
        args.job = args.job.resolve()
        job = read(args.job)
        args.iic_run, args.data_path, args.save_path = (Path(job[name]) for name in ('iic_run', 'data_path', 'save_path'))
        args.percentages, args.Iteration, args.cosine_views = [job['percentage']], job['Iteration'], job['cosine_views']
        assert os.environ.get('SLURM_JOB_ACCOUNT') == job['account']
        assert os.environ.get('SLURM_CLUSTER_NAME', os.environ.get('CC_CLUSTER')) == 'vulcan'
    if (args.Iteration is not None and args.Iteration < 1) or args.cosine_views < 1:
        parser.error('Iteration and cosine_views must be positive')
    assert os.environ.get('SLURM_JOB_ID'), 'Run inside a Slurm GPU allocation'
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    args.iic_run, args.data_path, args.save_path = (path.resolve() for path in
        (args.iic_run, args.data_path, args.save_path))
    assert args.iic_run.is_relative_to(HERE / 'runs') and args.save_path.is_relative_to(ROOT)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    check_selection()
    reference = reference_eval.reference_modules()
    common = reference['common']
    if args.job:
        for name, sha in job['code_hashes'].items():
            assert common.file_hash(ROOT / name) == sha, 'Code changed since submission: ' + name
        for name, sha in job['input_hashes'].items():
            assert common.file_hash(Path(name)) == sha, 'Clustering/teacher input changed: ' + name
    if args.stage == 'evaluate':
        source = read(args.iic_run / 'config.json')['dataset']
        for percentage in sorted(set(args.percentages)):
            evaluate(args, source, percentage, reference)
        return
    config, clusters, sha = load_clustering(args.iic_run, common)
    raw = reference['data'].RawDataset(args.data_path, config['dataset'], True)
    assert len(raw) == 50000 and raw.fingerprint == config['source_fingerprint']
    maximum = int(len(raw) * max(args.percentages) / 100)
    indices = balanced_main_aux(clusters['main']['assignments'].tolist(),
        clusters['overcluster']['assignments'].tolist(), clusters['main']['confidence'].tolist(), maximum)
    for percentage in sorted(set(args.percentages)):
        distill(args, common, raw, config, clusters, sha, indices, percentage)


if __name__ == '__main__':
    main()
