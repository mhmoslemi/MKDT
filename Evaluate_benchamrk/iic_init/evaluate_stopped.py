"""Evaluate an explicitly accepted stopped checkpoint without changing its data."""
import argparse
import math
import os
from pathlib import Path
import statistics

import torch

from distill_main_aux import BENCH, ROOT, progress, read, reference_eval


def accepted_stop(saved, actual, planned):
    return saved['complete'] is False and saved['iteration'] == actual and 0 < actual < planned


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', type=Path, required=True)
    args = parser.parse_args()
    assert accepted_stop(dict(complete=False, iteration=8000), 8000, 10000)
    assert not accepted_stop(dict(complete=True, iteration=8000), 8000, 10000)
    assert not accepted_stop(dict(complete=False, iteration=7500), 8000, 10000)
    assert not accepted_stop(dict(complete=False, iteration=10000), 10000, 10000)
    assert os.environ.get('SLURM_JOB_ID'), 'Submit through Slurm'
    assert os.environ.get('SLURM_CLUSTER_NAME', os.environ.get('CC_CLUSTER')) == 'vulcan'
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    reference = reference_eval.reference_modules()
    common, data = reference['common'], reference['data']
    request = read(args.job)
    for name, sha in request['code_hashes'].items():
        assert common.file_hash(ROOT / name) == sha, 'Evaluation code changed: ' + name
    original_path = Path(request['original_job'])
    assert common.file_hash(original_path) == request['original_job_sha256']
    job = read(original_path)
    assert os.environ.get('SLURM_JOB_ACCOUNT') == job['account']
    source, percentage = job['source'], job['percentage']
    assert source in ('CIFAR10', 'CIFAR100') and percentage == 2
    assert job['ssl_epochs'] == 2000 and job['ssl_seed'] == 0
    assert job['probe_seeds'] == list(range(5)) and job['labels'] == [1, 5]
    assert job['linear_epochs'] == 100 and job['cosine_views'] == 4
    for name, sha in dict(job['code_hashes'], **job['input_hashes']).items():
        assert common.file_hash(ROOT / name) == sha, 'Original input/code changed: ' + name
    output = Path(job['save_path']) / source / 'ResNet18BN' / 'pt2'
    assert output.resolve().is_relative_to(ROOT / 'OT-SSL-DD')
    contract = read(output / 'pipeline_config.json')
    for name, sha in contract['code_hashes'].items():
        assert common.file_hash(ROOT / name) == sha, 'Generation code changed: ' + name
    path = output / ('res_IIC_%s_ResNet18BN_2percent_final.pt' % source)
    synthetic_sha = common.file_hash(path)
    assert synthetic_sha == request['checkpoint_sha256'], 'Saved synthetic data changed'
    saved = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    iteration = request['accepted_iteration']
    assert accepted_stop(saved, iteration, job['Iteration']), 'Not the accepted stopped iteration'
    assert saved['run_args'] == contract['generation_args']
    assert saved['run_args']['Iteration'] == job['Iteration']
    assert saved['initialization'] == contract['selection'] and saved['init'] == 'iic'
    assert saved['dataset'] == source and saved['percentage'] == percentage
    assert saved['iic_params']['cosine_views'] == job['cosine_views']
    assert saved['iic_params']['similarity'] == 'cosine' and saved['iic_params']['feature_layers'] == 'last'
    assert not saved['iic_params']['assignment_filter']
    assert saved['normalization'] == reference_eval.SYNTHETIC_NORMALIZATION[source]
    images = saved['data'].detach().cpu().contiguous()
    assert images.shape == (1000, 3, 32, 32) and torch.isfinite(images).all()
    mean, std = (torch.tensor(value).view(1, 3, 1, 1) for value in saved['normalization'])
    synthetic = reference_eval.SyntheticImages((images * std + mean).clamp(0, 1).contiguous())
    protocol = read(BENCH / 'protocol.json')
    protocol['ssl'].update(subset_epochs=2000, checkpoint_every_epochs=500)
    protocol['subset_seeds'] = [0]
    assert protocol['evaluation']['epochs'] == 100 and protocol['evaluation']['lr'] == 0.1
    assert protocol['ssl']['batch_size'] == protocol['evaluation']['batch_size'] == 256
    train_set, test_set = (data.RawDataset(job['data_path'], source, training) for training in (True, False))
    assert train_set.fingerprint == contract['selection']['source_fingerprint']
    signature = dict(experiment='iic_main_aux_mipa_stopped', source=source, model='ResNet18', subset=percentage,
        ssl='simclr', protocol=protocol, generation=contract, source_fingerprint=train_set.fingerprint,
        synthetic=dict(path=str(path), file_sha256=synthetic_sha, normalization=saved['normalization'],
                       pixel_policy='denormalize_clamp_then_PIL_uint8'),
        distillation_iterations=iteration, planned_distillation_iterations=job['Iteration'],
        accepted_stopped_checkpoint=True, ssl_seeds=[0], probe_seeds=list(range(5)),
        code_hashes=common.code_hashes(), evaluation_request=request)
    identity = dict(signature=common.digest_json(signature), seed=0, selection_method='ours')
    config = dict(signature=signature, identity=identity)
    # Keep these results separate from any later completed 10k distillation.
    folder = output / ('evaluation_iter%d/simclr/ssl_seed_000' % iteration)
    del saved, images
    print('Accepted saved iteration %d/%d, unchanged synthetic SHA256 %s' %
          (iteration, job['Iteration'], synthetic_sha), flush=True)
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
                initialization='iic_main_aux_epoch150', cosine_views=job['cosine_views'],
                distillation_iterations=iteration, planned_distillation_iterations=job['Iteration'],
                accepted_stopped_checkpoint=True, ssl_epochs=2000, linear_epochs=100,
                ssl_seeds=[0], probe_seeds=list(range(5)), completed_probes=5, expected_probes=5,
                std_ddof=1, samples=samples, mean_percent=statistics.mean(scores), std_percent=statistics.stdev(scores)))
            print('COMPLETE CELL: %s pt2 lbl%d: %.2f +/- %.2f' %
                  (source, labels, statistics.mean(scores), statistics.stdev(scores)), flush=True)
        assert common.file_hash(path) == synthetic_sha, 'Synthetic data changed during evaluation'
        common.atomic_json(folder / 'complete.json', dict(complete=True, identity=identity,
            synthetic_file_sha256=synthetic_sha, encoder_file_sha256=common.file_hash(checkpoint),
            distillation_iterations=iteration, planned_distillation_iterations=job['Iteration'],
            accepted_stopped_checkpoint=True, ssl_epochs=2000, linear_epochs=100,
            labels=[1, 5], probe_seeds=list(range(5)), job_file_sha256=common.file_hash(args.job)))


if __name__ == '__main__':
    main()
