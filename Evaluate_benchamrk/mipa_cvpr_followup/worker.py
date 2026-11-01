"""Evaluate the CIFAR MIPA sets selected for the main table using existing benchmark recipes."""
import fcntl
import json
import math
import os
from pathlib import Path
import statistics
import sys

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent
ROOT = BENCH.parent
sys.path.insert(0, str(BENCH))

import torch
from torchvision.transforms.functional import to_pil_image
import common
import data
import train_ssl
from evaluate import linear_probe


def read(path):
    return json.loads(Path(path).read_text())


def progress(*values, **kwargs):
    if kwargs.get('file') is None and len(values) == 1 and isinstance(values[0], str) and values[0].startswith('SSL '):
        prefix, line = values[0].split(': ', 1)
        stats = json.loads(line)
        if stats['epoch'] != 1 and stats['epoch'] % 200:
            return
        stats.pop('projector_std', None)
        stats.pop('seconds', None)
        values = (prefix + ': ' + json.dumps(stats),)
    print(*values, **kwargs)


def probes(job, checkpoint, config, marker, folder):
    source, protocol, identity = job['source'], config['signature']['protocol'], config['identity']
    for cell in job['cells']:
        target, labels = cell['target'], cell['labels']
        train, test = data.RawDataset(job['data_root'], target, True), data.RawDataset(job['data_root'], target, False)
        samples = []
        for seed in job['probe_seeds']:
            output = folder / 'evaluations' / ('%s_lbl%d_probe%d.json' % (target, labels, seed))
            result = linear_probe(checkpoint, identity, protocol, source, target, train, test, labels, seed, output, workers=0, device='cuda')
            assert result['complete'] and result['encoder_unchanged'] and result['encoder_sha256'] == marker['encoder_sha256']
            assert math.isfinite(result['accuracy_percent'])
            samples.append(dict(ssl_seed=job['seed'], probe_seed=seed, accuracy_percent=result['accuracy_percent'], result_file=str(output)))
        scores = [sample['accuracy_percent'] for sample in samples]
        summary = dict(complete=True, identity=identity, source=source, target=target, subset=job['subset'],
            model=job['model'], ssl=job['method'], ssl_epochs=job['ssl_epochs'], linear_epochs=protocol['evaluation']['epochs'],
            completed_probes=len(scores), expected_probes=len(job['probe_seeds']), std_ddof=1,
            samples=samples, mean_percent=statistics.mean(scores), std_percent=statistics.stdev(scores),
            labels=labels, synthetic_file=job['synthetic_file'], synthetic_sha256=job['synthetic_sha256'], table=job['table'])
        common.atomic_json(folder / ('summary_%s_lbl%d.json' % (target, labels)), summary)
        print('COMPLETE CELL: %s %g%% %s %s lbl=%d: %.2f +/- %.2f (%d probes, SSL seed %d)' %
            (source, job['subset'], job['model'], target, labels, summary['mean_percent'], summary['std_percent'], len(scores), job['seed']), flush=True)


def main():
    plan = read(HERE / 'manifest.json')
    task = next(task for task in plan['jobs'] if task['name'] == sys.argv[1])
    recipe = plan['recipes'].get(task.get('recipe'), {})
    job = {**plan['shared'], **plan['sets'][task['set']], **recipe, **task}
    if recipe:
        job['protocol'] = plan['protocols'][recipe['protocol_id']]
    assert os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_JOB_ACCOUNT') == job['account']
    assert os.environ.get('SLURM_CLUSTER_NAME', os.environ.get('CC_CLUSTER')) == 'vulcan'
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    assert job['source'] in ('CIFAR10', 'CIFAR100') and job['subset'] in (0.2, 1)
    for relative, expected in job['code_hashes'].items():
        assert common.file_hash(ROOT / relative) == expected, 'Code changed: ' + relative
    assert common.code_hashes() == job['benchmark_code_hashes']
    assert common.file_hash(job['synthetic_file']) == job['synthetic_sha256']
    assert common.file_hash(job['main_config']) == job['main_config_sha256']
    main_config = read(job['main_config'])
    assert main_config['signature']['generation_args'] == job['generation_args']
    assert main_config['signature']['synthetic_sha256'] == job['synthetic_sha256']
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if job['stage'] == 'cross_arch':
        sys.path.insert(0, str(BENCH / 'cross_arch_ablation'))
        from models import install
        install()
    folder = Path(job['output_dir'])
    if job['stage'] == 'downstream':
        encoder_dir = Path(job['encoder_dir'])
        with (encoder_dir / '.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
            config = read(encoder_dir / 'config.json')
            marker = read(encoder_dir / 'ssl_complete.json')
            checkpoint = encoder_dir / 'encoder.pt'
            assert config == main_config and common.digest_json(config['signature']) == config['identity']['signature']
            assert common.file_hash(checkpoint) == job['encoder_sha256'] == marker['checkpoint_file_sha256']
            assert marker['complete'] and marker['epochs'] == 2000 and marker['identity'] == config['identity']
            assert config['signature']['source'] == job['source'] and config['signature']['subset'] == job['subset']
            assert config['signature']['ssl'] == 'simclr' and config['signature']['model'] == 'ResNet18'
            with common.exclusive_run(folder / 'locks' / job['name']):
                probes(job, checkpoint, config, marker, folder)
                common.atomic_json(folder / ('complete_' + job['name'] + '.json'), dict(complete=True, job=job['name'], identity=config['identity']))
        return

    assert job['stage'] in ('barlow', 'cross_arch')
    assert common.file_hash(job['recipe_file']) == job['recipe_sha256']
    protocol = job['protocol']
    original_protocol = read(job['recipe_file'])['signature']['protocol']
    original_protocol['ssl']['checkpoint_every_epochs'] = 100
    assert original_protocol == protocol
    assert protocol['ssl']['subset_epochs'] == job['ssl_epochs']
    assert protocol['evaluation'] == read(BENCH / 'protocol.json')['evaluation']
    saved = torch.load(job['synthetic_file'], map_location='cpu', weights_only=True, mmap=True)
    assert saved['complete'] and saved['run_args'] == job['generation_args']
    images = saved['data'].contiguous()
    assert images.shape == (int(50000 * job['subset'] / 100), 3, 32, 32) and torch.isfinite(images).all()
    assert saved['normalization'] == main_config['signature']['normalization']
    mean, std = (torch.tensor(v).view(1, 3, 1, 1) for v in saved['normalization'])
    pixels = (images * std + mean).clamp(0, 1).contiguous()
    synthetic = data.ImageView(torch.utils.data.TensorDataset(pixels, torch.full((len(pixels),), -1)), to_pil_image)
    source = data.RawDataset(job['data_root'], job['source'], True)
    assert source.fingerprint == main_config['signature']['source_fingerprint']
    signature = dict(experiment=HERE.name, source=job['source'], subset=job['subset'], model=job['model'], ssl=job['method'],
        source_fingerprint=source.fingerprint, protocol=protocol, synthetic_path=job['synthetic_file'], synthetic_sha256=job['synthetic_sha256'],
        generation_args=job['generation_args'], normalization=saved['normalization'], pixel_policy='denormalize_clamp_then_PIL_uint8',
        ssl_seeds=[0, 1], probe_seeds=job['probe_seeds'], code_hashes=common.code_hashes(), driver_code_hashes=job['code_hashes'])
    identity = dict(signature=common.digest_json(signature), seed=job['seed'], selection_method='ours')
    config = dict(signature=signature, identity=identity)
    del saved, images, pixels, source
    train_ssl.print = progress
    with common.exclusive_run(folder):
        config_file = folder / 'config.json'
        if config_file.exists():
            assert read(config_file) == config, 'Evaluation inputs changed'
        common.atomic_json(config_file, config)
        checkpoint, trained = train_ssl.train(synthetic, protocol, job['source'], job['model'], job['method'], job['seed'], folder, identity, workers=0, device='cuda')
        assert trained['complete'] and trained['epochs'] == job['ssl_epochs']
        marker = dict(complete=True, identity=identity, epochs=trained['epochs'], encoder_sha256=trained['encoder_sha256'], checkpoint_file_sha256=common.file_hash(checkpoint))
        common.atomic_json(folder / 'ssl_complete.json', marker)
        del synthetic
        probes(job, checkpoint, config, marker, folder)
        common.atomic_json(folder / 'complete.json', dict(complete=True, job=job['name'], identity=identity, cells=job['cells']))


if __name__ == '__main__':
    main()
