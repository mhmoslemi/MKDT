"""Frozen-teacher MIPA generation, then one 2k SimCLR encoder and five probes per label budget."""
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import sys

import torch

ROOT = Path('/scratch/mmoslem3/MKDT')
sys.path.insert(0, str(ROOT / 'OT-SSL-DD'))
import eval as reference_eval


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


def main():
    manifest, stage = Path(sys.argv[1]).resolve(), sys.argv[2]
    job = read(manifest)
    assert stage in ('generation', 'evaluation')
    assert os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_JOB_ACCOUNT') == job['account']
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    reference = reference_eval.reference_modules()
    common, data = reference['common'], reference['data']
    for path, expected in job['code_hashes'].items():
        assert common.file_hash(path) == expected, 'Code changed: ' + path
    assert common.file_hash(job['teacher']) == job['teacher_sha256'], 'Teacher changed'
    args = job['generation_args']
    source, subset = args['dataset'], args['percentage']
    output = Path(job['synthetic_dir'])
    checkpoint = output / ('res_IIC_%s_ResNet18BN_%gpercent_final.pt' % (source, subset))
    with common.exclusive_run(output if stage == 'generation' else job['evaluation_dir']):
        if stage == 'generation':
            spec = importlib.util.spec_from_file_location('cvpr_mipa', manifest.parent / 'method1_cvpr.py')
            method = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(method)
            sys.argv = [str(manifest.parent / 'method1_cvpr.py')]
            for key, value in args.items():
                if isinstance(value, bool):
                    if value:
                        sys.argv.append('--' + key)
                else:
                    sys.argv += ['--' + key, str(value)]
            sys.argv += ['--device', 'cuda', '--data_path', job['data_root'], '--save_path', str(output), '--resume', '--save_every', '500']
            method.main()
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
        assert saved['complete'] and saved['iteration'] == args['Iteration'] and saved['run_args'] == args
        assert saved['teacher_source'] == 'pretrained' and saved['teacher_seed'] == job['teacher_seed']
        assert saved['init'] == 'random' and saved['iic_params']['strategy'] == 'color_crop_cutout_scale'
        if stage == 'generation':
            common.atomic_json(output / 'complete.json', dict(complete=True, iteration=saved['iteration'], checkpoint=str(checkpoint), generation_args=args))
            return

        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if source == 'SVHN':
            prepared = read(ROOT / 'Evaluate_benchamrk/digits_simclr/prepared.json')
            protocol = prepared['protocol']
            dataset_class = reference['datasets'].DigitDataset
        else:
            protocol = read(ROOT / 'Evaluate_benchamrk/protocol.json')
            dataset_class = data.RawDataset
        train_set, test_set = (dataset_class(job['data_root'], source, training) for training in (True, False))
        if source == 'SVHN':
            assert train_set.fingerprint == prepared['datasets'][source]['train_fingerprint']
            assert test_set.fingerprint == prepared['datasets'][source]['test_fingerprint']
        protocol['ssl'].update(subset_epochs=2000, checkpoint_every_epochs=100)
        protocol['evaluation'].update(epochs=100, lr=0.1, batch_size=256)
        protocol['subset_seeds'] = [0]
        images = saved['data'].contiguous()
        size = protocol['input_size'][source]
        assert images.shape == (int(len(train_set) * subset / 100), 3, size, size) and torch.isfinite(images).all()
        assert saved['normalization'] == reference_eval.SYNTHETIC_NORMALIZATION[source]
        mean, std = (torch.tensor(v).view(1, 3, 1, 1) for v in saved['normalization'])
        synthetic = reference_eval.SyntheticImages((images * std + mean).clamp(0, 1).contiguous())
        signature = dict(experiment='mipa_cvpr_color_crop', source=source, subset=subset, model='ResNet18', ssl='simclr', protocol=protocol,
                         generation_args=args, teacher=job['teacher'], teacher_sha256=job['teacher_sha256'], code_hashes=job['code_hashes'],
                         source_fingerprint=train_set.fingerprint, synthetic_path=str(checkpoint), synthetic_sha256=common.file_hash(checkpoint),
                         normalization=saved['normalization'], ssl_seeds=[0], probe_seeds=list(range(5)), labels=[1, 5])
        identity = dict(signature=common.digest_json(signature), seed=0, selection_method='ours')
        config = dict(signature=signature, identity=identity)
        folder = Path(job['evaluation_dir'])
        if (folder / 'config.json').exists():
            assert read(folder / 'config.json') == config, 'Evaluation inputs changed'
        common.atomic_json(folder / 'config.json', config)
        del images, saved
        reference['train_ssl'].print = progress
        encoder, trained = reference['train_ssl'].train(synthetic, protocol, source, 'ResNet18', 'simclr', 0, folder, identity, workers=0, device='cuda')
        assert trained['epochs'] == 2000
        common.atomic_json(folder / 'ssl_complete.json', dict(complete=True, identity=identity, epochs=2000, encoder_sha256=trained['encoder_sha256'], checkpoint_file_sha256=common.file_hash(encoder)))
        del synthetic
        for labels in (1, 5):
            samples = []
            for seed in range(5):
                result_file = folder / 'evaluations' / ('%s_lbl%d_probe%d.json' % (source, labels, seed))
                result = reference['evaluate'].linear_probe(encoder, identity, protocol, source, source, train_set, test_set, labels, seed, result_file, workers=0, device='cuda')
                assert result['complete'] and result['encoder_unchanged'] and result['encoder_sha256'] == trained['encoder_sha256']
                assert math.isfinite(result['accuracy_percent'])
                samples.append(dict(ssl_seed=0, probe_seed=seed, accuracy_percent=result['accuracy_percent'], result_file=str(result_file)))
            scores = [sample['accuracy_percent'] for sample in samples]
            summary = dict(complete=True, identity=identity, source=source, target=source, subset=subset, model='ResNet18', ssl='simclr',
                           ssl_epochs=2000, linear_epochs=100, labels=labels, completed_probes=5, expected_probes=5, std_ddof=1,
                           samples=samples, mean_percent=statistics.mean(scores), std_percent=statistics.stdev(scores), generation_args=args)
            common.atomic_json(folder / ('summary_%s_lbl%d.json' % (source, labels)), summary)
            print('COMPLETE CELL: %s pt=%g lbl=%d: %.2f +/- %.2f' % (source, subset, labels, summary['mean_percent'], summary['std_percent']), flush=True)
        common.atomic_json(folder / 'complete.json', dict(complete=True, identity=identity, ssl_epochs=2000, labels=[1, 5], probe_seeds=list(range(5))))


if __name__ == '__main__':
    main()
