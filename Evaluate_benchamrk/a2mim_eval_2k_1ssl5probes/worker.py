"""Use saved A²MIM full-data encoders or train one subset encoder, then five probes."""
import contextlib
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent
ROOT = BENCH.parent
sys.path.insert(0, str(BENCH))
import torch
from torchvision.transforms.functional import to_pil_image
from common import atomic_json, code_hashes, digest_json, exclusive_run, file_hash, load_encoder
from data import ImageView, RawDataset, SOURCE_COUNTS, SyntheticDataset
from evaluate import linear_probe
from a2mim_eval_2k_1ssl5probes.train_subset import train


def read(path):
    return json.loads(Path(path).read_text())


def inside(value):
    folder = (ROOT / value).resolve()
    if not folder.is_relative_to(ROOT) or folder == ROOT:
        raise ValueError('Input and output paths must stay in the repository')
    return folder


def context(unit, data_root):
    source, kind, subset = (unit[k] for k in ('source', 'kind', 'subset'))
    if (source not in ('CIFAR10', 'CIFAR100') or kind not in ('full', 'random', 'mkdt', 'ours')
            or subset not in ((100,) if kind == 'full' else (0.2, 1))):
        raise ValueError('Unrequested SSL-method table unit')
    origin = inside(unit['origin_config'])
    if file_hash(origin) != unit['origin_config_sha256']:
        raise ValueError('Original input configuration changed')
    old = read(origin)
    folder = HERE / 'results' / source / 'ResNet18' / kind / ('pt%g' % subset) / 'a2mim/ssl_seed_000'
    protocol = read(BENCH / 'protocol.json')
    if protocol['evaluation']['epochs'] != 100:
        raise ValueError('Frozen probes must retain the existing 100-epoch recipe')
    if kind == 'full':
        checkpoint = origin.parent / 'encoder.pt'
        marker = read(origin.parent / 'ssl_complete.json')
        if (not marker['complete'] or marker['epochs'] != 300 or marker['method'] != 'a2mim'
                or marker['identity'] != old['identity'] or old['signature']['source'] != source
                or old['signature']['epochs'] != 300
                or digest_json(old['signature']) != old['identity']['signature']):
            raise ValueError('Not the completed full-data A²MIM teacher')
        for relative, expected in old['signature']['code_hashes'].items():
            if file_hash(ROOT / relative) != expected:
                raise ValueError('Full teacher training code changed: ' + relative)
        return folder, old, protocol, checkpoint, marker, None
    count = int(SOURCE_COUNTS[source] * subset / 100)
    raw = RawDataset(data_root, source, True)
    if old['signature']['source_fingerprint'] != raw.fingerprint:
        raise ValueError('Source dataset changed since the saved subset evaluation')
    if kind == 'random':
        selection = old['identity']['selection']
        indices = selection['indices']
        if (old['identity']['seed'] != 0 or old['signature']['selection_method'] != 'random'
                or len(indices) != count or len(set(indices)) != count
                or any(not 0 <= i < len(raw) for i in indices)):
            raise ValueError('Wrong original seed-0 random subset')
        dataset = torch.utils.data.Subset(raw, indices)
        input_metadata = dict(selection=selection, origin_config=unit['origin_config'],
                              origin_config_sha256=unit['origin_config_sha256'])
    elif kind == 'mkdt':
        previous = old['signature']['synthetic']
        path = inside(unit['synthetic'])
        if file_hash(path) != previous['file_sha256']:
            raise ValueError('MKDT synthetic file changed')
        dataset = SyntheticDataset(dict(path=str(path), space=previous['space'],
            pixel_policy=previous['pixel_policy'], generator=previous['generator']), source, subset, 32)
        input_metadata = dataset.metadata
    else:
        path = inside(unit['synthetic'])
        saved = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        if (not saved.get('complete') or saved['iteration'] != {0.2: 10000, 1: 8000}[subset]
                or saved['dataset'] != source or saved['model'] != 'ResNet18BN' or saved['percentage'] != subset):
            raise ValueError('Not the completed original MIPA synthetic set')
        images = saved['data'].detach().cpu().contiguous()
        previous = old['signature']['synthetic']
        if (images.shape != (count, 3, 32, 32) or not images.is_floating_point()
                or not torch.isfinite(images).all() or saved['normalization'] != previous['normalization']
                or hashlib.sha256(images.numpy().tobytes()).hexdigest() != previous['images_sha256']):
            raise ValueError('MIPA images or normalization changed')
        mean, std = (torch.tensor(v).view(1, 3, 1, 1) for v in saved['normalization'])
        pixels = images * std + mean
        dataset = ImageView(torch.utils.data.TensorDataset(pixels.clamp(0, 1).contiguous(),
                            torch.full((count,), -1)), to_pil_image)
        input_metadata = {**previous, 'file_sha256': file_hash(path)}
    if len(dataset) != count:
        raise ValueError('Subset size differs from its table row')
    full_recipe = read(BENCH / 'a2mim_teacher/checkpoints' / source /
                       'ResNet18/full/pt100/a2mim/seed_000/config.json')['signature']
    recipe = json.loads(json.dumps(full_recipe))
    batch = min(256, count)
    recipe.update(epochs=2000, count=count, batch_size=batch, peak_lr=1.2e-3 * batch / 2048,
                  checkpoint_every_epochs=500, selection_method=kind, subset=subset)
    recipe['reference_full_teacher_code_hashes'] = recipe['code_hashes']
    recipe['code_hashes'] = unit['code_hashes']
    recipe['training_split_fingerprint'] = digest_json(input_metadata)
    protocol['ssl'].update(subset_epochs=2000, checkpoint_every_epochs=500)
    protocol['subset_seeds'] = [0]
    signature = dict(experiment=HERE.name, source=source, model='ResNet18', selection_method=kind,
        subset=subset, ssl='a2mim', seed=0, protocol=protocol, recipe=recipe,
        source_fingerprint=raw.fingerprint, input=input_metadata, ssl_seeds=[0], probe_seeds=list(range(5)),
        code_hashes=code_hashes(), driver_code_hashes=unit['code_hashes'])
    identity = dict(signature=digest_json(signature), seed=0, selection_method=kind)
    return folder, dict(signature=signature, identity=identity), protocol, folder / 'encoder.pt', None, dataset


def probes(unit, cells, data_root):
    folder, config, protocol, checkpoint, marker, dataset = context(unit, data_root)
    del dataset
    folder.mkdir(parents=True, exist_ok=True)
    with contextlib.ExitStack() as locks:
        lock = locks.enter_context((checkpoint.parent / '.lock').open('a'))
        fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        if unit['kind'] != 'full':
            if read(folder / 'config.json') != config:
                raise ValueError('Subset encoder configuration differs from its requested probe')
            marker = read(folder / 'ssl_complete.json')
        if (not marker['complete'] or marker['identity'] != config['identity']
                or marker['epochs'] != (300 if unit['kind'] == 'full' else 2000)
                or file_hash(checkpoint) != marker['checkpoint_file_sha256']):
            raise ValueError('Completed encoder file or identity changed')
        encoder, _, saved = load_encoder(checkpoint, config['identity'], 'cpu')
        if saved['method'] != 'a2mim' or saved['encoder_sha256'] != marker['encoder_sha256']:
            raise ValueError('Not the requested frozen A²MIM encoder')
        del encoder, saved
        with exclusive_run(folder / '.metadata'):
            metadata = dict(config=config, checkpoint=str(checkpoint), identity=config['identity'],
                checkpoint_file_sha256=marker['checkpoint_file_sha256'], encoder_sha256=marker['encoder_sha256'],
                epochs=marker['epochs'], evaluation=protocol['evaluation'], evaluation_code_hashes=code_hashes(),
                driver_code_hashes=unit['code_hashes'])
            file = folder / 'probe_contract.json'
            if file.exists() and read(file) != metadata:
                raise ValueError('Probe contract changed')
            atomic_json(file, metadata)
        for cell in cells:
            target, labels = cell['target'], cell['labels']
            if target not in ('CIFAR10', 'CIFAR100', 'TinyImageNet') or labels not in (1, 5):
                raise ValueError('Target or label budget is outside this table')
            with exclusive_run(folder / ('probe_locks/%s_lbl%d' % (target, labels))):
                train_set, test_set = RawDataset(data_root, target, True), RawDataset(data_root, target, False)
                samples = []
                for seed in range(5):
                    output = folder / 'evaluations' / ('%s_lbl%d_probe%d.json' % (target, labels, seed))
                    print('SKIP saved probe:' if output.exists() else 'RUN probe:', unit['source'], unit['kind'], unit['subset'], target, labels, seed, flush=True)
                    result = linear_probe(checkpoint, config['identity'], protocol, unit['source'], target,
                        train_set, test_set, labels, seed, output, workers=0, device='cuda')
                    score = result['accuracy_percent']
                    if (not result['complete'] or not result['encoder_unchanged']
                            or result['encoder_sha256'] != marker['encoder_sha256']
                            or not math.isfinite(score) or not 0 <= score <= 100):
                        raise ValueError('Invalid frozen probe')
                    samples.append(dict(ssl_seed=0, probe_seed=seed, accuracy_percent=score,
                                        encoder_sha256=marker['encoder_sha256'], result_file=str(output)))
                scores = [s['accuracy_percent'] for s in samples]
                summary = {key: unit[key] for key in ('source', 'kind', 'subset')}
                summary.update(complete=True, target=target, labels=labels, ssl='a2mim', model='ResNet18',
                    ssl_epochs=marker['epochs'], ssl_seed=0, linear_epochs=100, std_ddof=1,
                    expected_probes=5, completed_probes=5, samples=samples,
                    mean_percent=statistics.mean(scores), std_percent=statistics.stdev(scores))
                summary['tex_value'] = '\\dsacc{%.2f}{%.2f}' % (summary['mean_percent'], summary['std_percent'])
                summary['cell_key'] = '|'.join(map(str, (unit['source'], 'ResNet18', unit['kind'],
                                                      unit['subset'], 'a2mim', target, labels)))
                file = folder / ('summary_%s_lbl%d.json' % (target, labels))
                atomic_json(file, summary)
                with (ROOT / 'CVPR_tables/.probe_completion.lock').open('a') as table_lock:
                    fcntl.flock(table_lock, fcntl.LOCK_EX)
                    subprocess.run(['node', str(HERE / 'jobs.js'), '--update', str(file)], check=True)


def main():
    file = Path(sys.argv[1]).resolve()
    if not file.is_relative_to(HERE / 'batches'):
        raise ValueError('Unexpected manifest')
    job = read(file)
    if (not os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_JOB_ACCOUNT') != job['account']
            or job['account'] not in ('aip-boyuwang', 'aip-yiweilu')
            or os.environ.get('SLURM_CLUSTER_NAME', os.environ.get('CC_CLUSTER')) != 'vulcan'
            or not torch.cuda.is_available()):
        raise RuntimeError('Use the Wang/Lu Vulcan GPU launcher')
    for relative, expected in job['unit']['code_hashes'].items():
        if file_hash(ROOT / relative) != expected:
            raise ValueError('Code changed since submission: ' + relative)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    data_root = Path(sys.argv[2]).resolve()
    if job['stage'] == 'ssl':
        unit = job['unit']
        if unit['kind'] == 'full':
            raise ValueError('Do not retrain full-data teachers')
        folder, config, protocol, checkpoint, marker, dataset = context(unit, data_root)
        with exclusive_run(folder):
            file = folder / 'config.json'
            if file.exists() and read(file) != config:
                raise ValueError('A²MIM subset training configuration changed')
            atomic_json(file, config)
            train(dataset, config, folder)
    elif job['stage'] == 'probe_bundle' and 1 <= len(job['cells']) <= 4:
        probes(job['unit'], job['cells'], data_root)
    else:
        raise ValueError('Unknown job stage')


if __name__ == '__main__':
    main()
