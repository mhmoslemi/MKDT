"""Saved similarity-ablation images: fresh 1k SimCLR and three probes per label budget."""
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys

import torch

HERE = Path(__file__).resolve().parent
BENCH, ROOT = HERE.parent, HERE.parent.parent
sys.path.insert(0, str(ROOT / 'OT-SSL-DD'))
import eval as reference_eval


def read(path):
    return json.loads(Path(path).read_text())


def inputs(job, data_root, reference):
    source, subset = job['source'], job['subset']
    path = ROOT / job['synthetic_path']
    assert path.is_relative_to(ROOT / 'OT-SSL-DD/ours_syntehtic_similarity_lr0p002_b128_i10k')
    assert read(path.parent / 'pipeline_config.json') == job['generation_contract']
    state = read(path.parent / 'generation_state.json')
    assert state['complete'] and state['iteration'] == 10000 and state['contract'] == job['generation_contract']
    saved = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    assert saved['complete'] and saved['iteration'] == 10000
    assert saved['dataset'] == source and saved['percentage'] == subset and saved['model'] == 'ResNet18BN'
    assert saved['run_args'] == job['generation_contract']['generation_args']
    assert saved['initialization'] == job['generation_contract']['selection'] and saved['init'] == 'random'
    assert saved['iic_params']['similarity'] == job['similarity']
    assert all(saved['iic_params'][key] == value for key, value in job['generation_contract']['metric_parameters'].items())
    if source in reference_eval.DIGIT_PREPARED:
        prepared = read(reference_eval.DIGIT_PREPARED[source])
        protocol = prepared['protocol']
        module = reference['_fashion_dataset' if source == 'FashionMNIST' else 'datasets']
        train_set = module.DigitDataset(data_root, source, True)
        test_set = module.DigitDataset(data_root, source, False)
        assert train_set.fingerprint == prepared['datasets'][source]['train_fingerprint']
        assert test_set.fingerprint == prepared['datasets'][source]['test_fingerprint']
    else:
        protocol = read(BENCH / 'protocol.json')
        train_set = reference['data'].RawDataset(data_root, source, True)
        test_set = reference['data'].RawDataset(data_root, source, False)
    assert train_set.fingerprint == job['generation_contract']['selection']['source_fingerprint']
    protocol['ssl'].update(subset_epochs=1000, checkpoint_every_epochs=500)
    protocol['subset_seeds'] = [0]
    protocol['evaluation'].update(epochs=100, lr=0.1, batch_size=256)
    assert protocol['ssl']['batch_size'] == 256
    images = saved['data'].detach().cpu().contiguous()
    assert images.shape == (int(len(train_set) * subset / 100), 3, 32, 32)
    assert images.is_floating_point() and torch.isfinite(images).all()
    assert saved['normalization'] == reference_eval.SYNTHETIC_NORMALIZATION[source]
    mean, std = (torch.tensor(v).view(1, 3, 1, 1) for v in saved['normalization'])
    pixels = images * std + mean
    common = reference['common']
    signature = dict(experiment=HERE.name, source=source, subset=subset, model='ResNet18', ssl='simclr',
        selection_method='ours', similarity=job['similarity'], protocol=protocol,
        generation_contract=job['generation_contract'], source_fingerprint=train_set.fingerprint,
        synthetic=dict(path=str(path), file_sha256=common.file_hash(path), iteration=10000,
            normalization=saved['normalization'], images_sha256=hashlib.sha256(images.numpy().tobytes()).hexdigest(),
            pixel_policy='denormalize_clamp_then_PIL_uint8', clipped_fraction=float(((pixels < 0) | (pixels > 1)).float().mean())),
        ssl_seeds=[0], probe_seeds=[0, 1, 2], labels=[1, 5], code_hashes=common.code_hashes(),
        driver_code_hashes=job['code_hashes'])
    identity = dict(signature=common.digest_json(signature), seed=0, selection_method='ours')
    return reference_eval.SyntheticImages(pixels.clamp(0, 1).contiguous()), train_set, test_set, dict(signature=signature, identity=identity)


def evaluate(job, reference, synthetic, train_set, test_set, config):
    common = reference['common']
    folder = HERE / 'results' / job['source'] / 'ResNet18BN' / ('pt%g' % job['subset']) / job['similarity'] / 'simclr/ssl_seed_000'
    protocol, identity = config['signature']['protocol'], config['identity']
    with common.exclusive_run(folder):
        cfg = folder / 'config.json'
        if cfg.exists() and read(cfg) != config:
            raise ValueError('Evaluation inputs changed; refusing to mix results')
        common.atomic_json(cfg, config)

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

        reference['train_ssl'].print = progress
        checkpoint, trained = reference['train_ssl'].train(synthetic, protocol, job['source'], 'ResNet18', 'simclr',
            0, folder, identity, workers=0, device='cuda')
        assert trained['complete'] and trained['epochs'] == 1000
        common.atomic_json(folder / 'ssl_complete.json', dict(complete=True, identity=identity, epochs=1000,
            encoder_sha256=trained['encoder_sha256'], checkpoint_file_sha256=common.file_hash(checkpoint)))
        for labels in (1, 5):
            samples = []
            for seed in range(3):
                output = folder / 'evaluations' / ('%s_lbl%d_probe%d.json' % (job['source'], labels, seed))
                result = reference['evaluate'].linear_probe(checkpoint, identity, protocol, job['source'], job['source'],
                    train_set, test_set, labels, seed, output, workers=0, device='cuda')
                score = result['accuracy_percent']
                assert result['complete'] and result['encoder_unchanged'] and result['encoder_sha256'] == trained['encoder_sha256']
                assert math.isfinite(score) and 0 <= score <= 100
                samples.append(dict(ssl_seed=0, probe_seed=seed, accuracy_percent=score, result_file=str(output)))
            scores = [r['accuracy_percent'] for r in samples]
            summary = dict(complete=True, experiment=HERE.name, source=job['source'], target=job['source'],
                subset=job['subset'], similarity=job['similarity'], model='ResNet18', ssl='simclr', ssl_epochs=1000,
                linear_epochs=100, labels=labels, ssl_seeds=[0], probe_seeds=[0, 1, 2], completed_probes=3,
                expected_probes=3, std_ddof=1, samples=samples, mean_percent=statistics.mean(scores),
                std_percent=statistics.stdev(scores), identity=identity, checkpoint=str(checkpoint))
            common.atomic_json(folder / ('summary_%s_lbl%d.json' % (job['source'], labels)), summary)
            print('COMPLETE CELL:', job['source'], job['subset'], job['similarity'], labels,
                  '%.2f +/- %.2f' % (summary['mean_percent'], summary['std_percent']), flush=True)
        common.atomic_json(folder / 'complete.json', dict(complete=True, identity=identity, ssl_epochs=1000,
            probe_seeds=[0, 1, 2], labels=[1, 5]))


def main():
    manifest, data_root = (Path(value).resolve() for value in sys.argv[1:3])
    check = sys.argv[3:] == ['--check']
    assert check or not sys.argv[3:]
    assert manifest.is_relative_to(HERE / 'batches') and os.environ.get('SLURM_JOB_ID')
    job = read(manifest)
    assert job['source'] in ('CIFAR10', 'CIFAR100', 'SVHN', 'FashionMNIST') and job['subset'] in (0.2, 1, 2)
    assert job['similarity'] in ('l1', 'l2', 'mmd', 'mmd-dist', 'dm', 'energy', 'whitened-cosine', 'sinkhorn')
    assert job['evaluation'] == dict(ssl='simclr', ssl_epochs=1000, ssl_seed=0, linear_epochs=100, probe_seeds=[0, 1, 2], labels=[1, 5])
    if not check:
        assert os.environ.get('SLURM_JOB_ACCOUNT') == job['account']
        assert os.environ.get('SLURM_CLUSTER_NAME', os.environ.get('CC_CLUSTER')) == 'vulcan'
        assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    reference = reference_eval.reference_modules()
    for relative, expected in job['code_hashes'].items():
        assert reference['common'].file_hash(ROOT / relative) == expected, 'Code changed: ' + relative
    if job.get('resume_generation'):
        if check:
            print('CHECK: generation resume queued separately inside this unit; no generation run during preflight')
            return
        sys.path.insert(0, str(ROOT))
        from Evaluate_benchamrk.mipa_similarity_ablation.worker import configure
        original = ROOT / job['resume_generation']
        old_job = read(original)
        base, original_reference = configure(old_job, original.parent / 'method1.py', data_root)
        base.generation(old_job, data_root, original_reference['common'])
    prepared = inputs(job, data_root, reference)
    if check:
        assert len(prepared[0]) == int(len(prepared[1]) * job['subset'] / 100)
        print('CHECK PASSED:', job['source'], job['subset'], job['similarity'], 'saved images, normalization, data fingerprints, 1k SSL / 3 probes per label')
        return
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    evaluate(job, reference, *prepared)


if __name__ == '__main__':
    main()
