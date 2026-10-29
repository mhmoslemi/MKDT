"""Existing MKDT trajectory matching and benchmark SimCLR, with digit adapters."""
import argparse
import ast
import importlib.util
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys

HERE = Path(__file__).resolve().parent
BENCH, ROOT = HERE.parent, HERE.parent.parent
sys.path[:0] = [str(BENCH / 'digits_simclr'), str(BENCH), str(ROOT)]

import numpy as np
import torch
from torchvision.utils import save_image
from common import (atomic_json, code_hashes, digest_json, exclusive_run, file_hash,
                    load_encoder)
import data
import train_ssl
from evaluate import linear_probe


def module_at(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read(path):
    return json.loads(Path(path).read_text())


def adapter(source):
    if source == 'FashionMNIST':
        fashion = module_at('finish_fashion', BENCH / 'fashionmnist/worker.py')
        return fashion.mkdt_worker, fashion.fashion.DigitDataset, BENCH / 'fashionmnist/full/prepared.json'
    previous = module_at('finish_digits', BENCH / 'digits_mkdt/worker.py')
    return previous, previous.digit_data.DigitDataset, BENCH / 'digits_simclr/prepared.json'


def configure(source, data_root):
    previous, dataset, prepared_file = adapter(source)
    prepared = read(prepared_file)
    for file, sha in prepared['code_hashes'].items():
        assert file_hash(ROOT / file) == sha, 'Prepared implementation changed: ' + file
    recipe = previous.recipe()
    assert str(data_root) == recipe['data_root']
    assert recipe['normalization'][source] == prepared['protocol']['normalization'][source]
    expected = prepared['datasets'][source]
    data.SOURCE_COUNTS[source] = expected['train_count']
    data.SPECS[source] = (10, *recipe['normalization'][source])
    return previous, dataset, prepared


def generation_folder(source, subset):
    return HERE / 'synthetic' / source / 'ConvNet' / ('pt%g' % subset)


def generation(job, data_root):
    source, subset, stop = job['source'], job['subset'], job['stop']
    previous, _, prepared = configure(source, data_root)
    count = int(prepared['datasets'][source]['train_count'] * subset / 100)
    work = previous.HERE / 'work' / source
    init = work / ('%s_barlow_twins_%g_high_loss_indices.pkl' % (source, subset))
    marker = read(work / 'init_complete.json')
    assert marker['complete'] and marker['identity'] == previous.source_identity(source)
    assert marker['expert_count'] == 100 and marker['expert_epochs'] == 20
    assert marker['files'][init.name] == file_hash(init)
    features = previous.validate_target(source)
    banks = {}
    for start in range(0, 100, 5):
        previous.previous_mkdt.validate_shard(work, start)
        shard = read(work / ('shard_%03d.json' % start))
        assert shard['identity'] == marker['identity']
        assert shard['target_file_sha256'] == features['file_sha256']
        banks.update(shard['files'])
    with init.open('rb') as stream:
        indices = np.asarray(pickle.load(stream))
    assert len(indices) == len(np.unique(indices)) == count
    assert indices.min() >= 0 and indices.max() < prepared['datasets'][source]['train_count']
    output = generation_folder(source, subset)
    contract = dict(source=source, subset=subset, count=count, iterations=5000,
        recipe_identity=marker['identity'], normalization=previous.recipe()['normalization'][source],
        initialization_sha256=file_hash(init), target_sha256=features['file_sha256'],
        expert_files=banks, code_hashes=job['code_hashes'],
        settings=dict(model='ConvNet', seed=0, expert_epochs=2, syn_steps=40,
            lr_img=1000, lr_labels=1, lr_lr=1e-4, lr_teacher=0.1,
            batch_syn=256, min_start_epoch=0, max_start_epoch=2, criterion='mse'))
    with exclusive_run(output):
        config = output / 'generation_config.json'
        if config.exists():
            assert read(config) == contract, 'Saved generation recipe changed'
        else:
            assert not (output / 'training.pt').exists(), 'Unrecognized generation checkpoint'
            atomic_json(config, contract)
        state_path = output / 'training.pt'
        iteration = 0
        if state_path.exists():
            state = torch.load(state_path, map_location='cpu', weights_only=False)
            iteration = state['iteration']
            del state
        if iteration < stop:
            raw, normalization = previous.raw_data(source, data_root)
            normalized = previous.NormalizedDigits(raw, normalization)
            original = module_at('finish_original_distill', ROOT / 'distill.py')

            def get_dataset(dataset, data_path):
                assert dataset == source and Path(data_path).resolve() == data_root
                return 3, (32, 32), 10, normalized, None

            original.get_dataset = get_dataset
            original.main(argparse.Namespace(**contract['settings'], dataset=source,
                iters=5000, stop_at=stop, checkpoint_every=100, eval_it=5000,
                data_path=str(data_root), train_labels_path=str(previous.target(source)),
                image_init_idx_path=str(init), expert_dir=str(work / 'buffers' / source / 'ConvNet'),
                run_id='%s_ConvNet_%gpct' % (source, subset), save_dir=str(output),
                resume=state_path.exists()))
        state = torch.load(state_path, map_location='cpu', weights_only=False)
        iteration = state['iteration']
        assert stop <= iteration <= 5000
        assert state['config']['dataset'] == source and state['config']['iters'] == 5000
        assert state['images'].shape == (count, 3, 32, 32)
        assert all(torch.isfinite(state[k]).all() for k in ('images', 'labels', 'synlr'))
        mean, std = (torch.tensor(v)[None, :, None, None] for v in contract['normalization'])
        for step in (0, 5000):
            image_file, png = output / ('images_%d.pt' % step), output / ('images_%d.png' % step)
            if image_file.exists() and not png.exists():
                pixels = torch.load(image_file, map_location='cpu', weights_only=True) * std + mean
                save_image(pixels.clamp(0, 1), png, nrow=10)
        progress = dict(complete=iteration == 5000, iteration=iteration,
                       contract_sha256=digest_json(contract))
        atomic_json(output / 'progress.json', progress)
        if progress['complete']:
            progress['files'] = {name: file_hash(output / name)
                for name in ('images_5000.pt', 'labels_5000.pt', 'synlr_5000.pt')}
            atomic_json(output / 'complete.json', progress)
        print('RETAINED MKDT:', source, subset, iteration, '/ 5000', flush=True)


def evaluate(job, data_root):
    source, subset = job['source'], job['subset']
    previous, dataset, prepared = configure(source, data_root)
    generated = generation_folder(source, subset)
    marker, contract = read(generated / 'complete.json'), read(generated / 'generation_config.json')
    assert marker['complete'] and marker['iteration'] == 5000
    assert marker['contract_sha256'] == digest_json(contract)
    assert contract['code_hashes'] == job['code_hashes']
    for file, sha in marker['files'].items():
        assert file_hash(generated / file) == sha, 'Synthetic data changed'
    protocol = prepared['protocol']
    protocol['ssl'].update(subset_epochs=2000, checkpoint_every_epochs=500)
    protocol['subset_seeds'] = [0]
    assert protocol['ssl']['batch_size'] == 256 and protocol['evaluation']['epochs'] == 100
    train_set, test_set = dataset(data_root, source, True), dataset(data_root, source, False)
    expected = prepared['datasets'][source]
    assert train_set.fingerprint == expected['train_fingerprint']
    assert test_set.fingerprint == expected['test_fingerprint']
    synthetic = data.SyntheticDataset(dict(path=str(generated / 'images_5000.pt'),
        space='repository_normalized', pixel_policy='clamp_to_unit_rgb', generator='MKDT_ConvNet'),
        source, subset, 32)
    output = HERE / 'results' / source / 'ResNet18' / ('pt%g' % subset) / 'simclr/ssl_seed_000'
    subset_worker = module_at('finish_subset', BENCH / 'digits_subset_simclr2k/worker.py')
    signature = dict(experiment=HERE.name, source=source, model='ResNet18', ssl='simclr',
        subset=subset, selection_method='mkdt', protocol=protocol, code_hashes=code_hashes(),
        driver_code_hashes=job['code_hashes'], generation=contract, synthetic=synthetic.metadata,
        source_fingerprint=train_set.fingerprint, test_fingerprint=test_set.fingerprint,
        probe_seeds=list(range(5)), label_percentages=[1, 5], table_scope='CVPR_tables/table.tex')
    identity = dict(signature=digest_json(signature), seed=0, selection_method='mkdt')
    config = dict(signature=signature, identity=identity)
    with exclusive_run(output):
        if (output / 'config.json').exists():
            assert read(output / 'config.json') == config, 'Saved evaluation recipe changed'
        atomic_json(output / 'config.json', config)
        atomic_json(output / 'environment.json', dict(job=os.environ['SLURM_JOB_ID'],
            account=os.environ.get('SLURM_JOB_ACCOUNT'), gpu=torch.cuda.get_device_name(),
            torch=str(torch.__version__), python=sys.version))
        train_ssl.print = subset_worker.progress
        checkpoint, metadata = train_ssl.train(synthetic, protocol, source, 'ResNet18', 'simclr',
            0, output, identity, workers=0, device='cuda')
        assert metadata['epochs'] == 2000
        encoder, _, checked = load_encoder(checkpoint, identity, device='cpu')
        assert checked['encoder_sha256'] == metadata['encoder_sha256']
        del encoder, checked
        atomic_json(output / 'ssl_complete.json', dict(complete=True, epochs=2000, identity=identity,
            encoder_sha256=metadata['encoder_sha256'], checkpoint_file_sha256=file_hash(checkpoint)))
        for labels in (1, 5):
            for seed in range(5):
                result = linear_probe(checkpoint, identity, protocol, source, source, train_set, test_set,
                    labels, seed, output / 'evaluations' / ('%s_lbl%d_probe%d.json' % (source, labels, seed)),
                    workers=0, device='cuda')
                assert result['complete'] and result['encoder_unchanged']
                summary = subset_worker.summarize(output, source, 'mkdt', subset, labels,
                    identity, metadata['encoder_sha256'])
            assert summary['complete']
        atomic_json(output / 'complete.json', dict(complete=True, identity=identity, ssl_epochs=2000,
            linear_epochs=100, probes_per_cell=5, completed_probes=10))


def check(data_root):
    for file in (HERE / 'worker.py',):
        ast.parse(file.read_text(), filename=str(file))
    subprocess.run([sys.executable, str(BENCH / 'mkdt_02_jobs/check_resume.py')], check=True,
                   cwd=ROOT, env=dict(os.environ, PYTHONPATH=str(ROOT)))
    for source in ('MNIST', 'FashionMNIST', 'SVHN'):
        previous, dataset, prepared = configure(source, data_root)
        raw, normalization = previous.raw_data(source, data_root)
        images = torch.stack([previous.NormalizedDigits(raw, normalization)[i][0] for i in range(2)])
        mean, std = (torch.tensor(v)[None, :, None, None] for v in normalization)
        pixels = images * std + mean
        torch.testing.assert_close(pixels, raw.images[:2].float().div(255).expand(-1, 3, -1, -1))
        assert raw.fingerprint == prepared['datasets'][source]['train_fingerprint']
        assert images.shape == (2, 3, 32, 32) and torch.isfinite(images).all()
        print('CHECK PASSED:', source, 'existing dataset/order/normalization', flush=True)
    atomic_json(HERE / 'checked.json', dict(complete=True, worker_sha256=file_hash(__file__)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('data_root', type=Path)
    args = parser.parse_args()
    job = read(args.manifest)
    assert os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_JOB_ACCOUNT') == job['account']
    assert os.environ.get('SLURM_CLUSTER_NAME', os.environ.get('CC_CLUSTER')) == 'vulcan'
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    for file, sha in job['code_hashes'].items():
        assert file_hash(ROOT / file) == sha, 'Implementation changed: ' + file
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    data_root = args.data_root.resolve()
    if job['stage'] == 'check':
        check(data_root)
    elif job['stage'] == 'generation':
        generation(job, data_root)
    else:
        assert job['stage'] == 'evaluation'
        evaluate(job, data_root)


if __name__ == '__main__':
    main()
