"""Rebuild MKDT experts, high-loss initialization, and 0.2% distilled images."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys

import numpy as np
import torch
from torchvision.utils import save_image


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    temporary = path.with_name(f'.mkdt-{path.name}.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(path)


def run(script, *arguments):
    subprocess.run([sys.executable, '-u', str(Path(__file__).parent / script),
                    *map(str, arguments)], check=True)


def validate_buffer(path):
    bank = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(bank, list) or len(bank) != 1 or len(bank[0]) != 21:
        raise ValueError(f'Expected one expert with initial + 20 trained snapshots: {path}')
    shapes = [tuple(p.shape) for p in bank[0][0]]
    for snapshot in bank[0]:
        if [tuple(p.shape) for p in snapshot] != shapes or not all(
                torch.isfinite(p).all().item() for p in snapshot):
            raise ValueError(f'Invalid expert snapshot: {path}')


def experts(args, folder, target, model):
    start = args.stop
    if start not in range(0, 100, 5):
        raise ValueError('Expert shards start at 0, 5, ..., 95')
    marker = folder / f'shard_{start:03d}.json'
    bank = folder / 'buffers' / args.dataset / model
    losses = folder / f'losses_{start:03d}.pt'
    if marker.is_file():
        validate_shard(folder, start)
        print('SKIP completed expert/ranking shard:', start, flush=True)
        return
    if start == 0:
        run('check_resume.py')
    for index in range(start, start + 5):
        path = bank / f'replay_buffer_{index}.pt'
        if path.exists():
            validate_buffer(path)
    # Completed experts survive timeouts; only an unfinished expert is repeated.
    run('buffer.py', '--dataset', args.dataset, '--model', model, '--num_experts', 5,
        '--start_index', start, '--seed', 0, '--buffer_path', folder / 'buffers',
        '--train_labels_path', target, '--train_epochs', 20, '--batch_train', 256,
        '--lr_teacher', 0.1, '--mom', 0.9, '--l2', 1e-4, '--criterion', 'mse',
        '--data_path', args.data, '--device', 0, '--stream_data', '--workers', 0)
    hashes = {}
    for index in range(start, start + 5):
        path = bank / f'replay_buffer_{index}.pt'
        validate_buffer(path)
        hashes[str(path.relative_to(folder))] = file_hash(path)
    run('sort_dset.py', '--dataset', args.dataset, '--model', model,
        '--num_buffers', 5, '--buffer_start', start, '--loss_output', losses,
        '--ssl_algo', 'barlow_twins', '--batch_train', 256, '--data_path', args.data,
        '--device', 0, '--train_labels_path', target, '--buffer_dir', bank,
        '--stream_data', '--workers', 0)
    hashes[losses.name] = file_hash(losses)
    write(marker, {'complete': True, 'files': hashes})


def validate_shard(folder, start):
    saved = read(folder / f'shard_{start:03d}.json')
    if not saved.get('complete') or len(saved['files']) != 6:
        raise ValueError('Incomplete expert/ranking shard')
    for relative, checksum in saved['files'].items():
        if file_hash(folder / relative) != checksum:
            raise ValueError(f'Expert/ranking file changed: {folder / relative}')


def initialize(folder, dataset, count):
    # Average five-expert shard means: all 100 experts have equal weight.
    losses = torch.zeros(count, dtype=torch.float64)
    for start in range(0, 100, 5):
        validate_shard(folder, start)
        values = torch.load(folder / f'losses_{start:03d}.pt', map_location='cpu', weights_only=True)
        if tuple(values.shape) != (count,) or not torch.isfinite(values).all().item():
            raise ValueError('Invalid per-image high-loss ranking')
        losses += values / 20
    indices = np.argsort(losses.numpy())[::-1][:count // 500].copy()
    if len(indices) != count // 500 or len(set(indices.tolist())) != len(indices):
        raise ValueError('Invalid 0.2% initialization indices')
    path = folder / f'{dataset}_barlow_twins_0.2_high_loss_indices.pkl'
    temporary = path.with_suffix('.tmp.pkl')
    with temporary.open('wb') as stream:
        pickle.dump(indices, stream)
    temporary.replace(path)
    write(folder / 'init_complete.json', {'complete': True, 'count': len(indices),
                                        'indices_sha256': file_hash(path)})
    print(f'Prepared {len(indices)} high-loss real-image initialization indices: {path}', flush=True)


def distill(args, folder, target, model, count):
    tiny = args.dataset == 'Tiny'
    total = 10000 if tiny else 5000
    if args.stop not in range(1000, total + 1, 1000):
        raise ValueError('Distillation chunks must end on a 1000-update boundary')
    init = folder / f'{args.dataset}_barlow_twins_0.2_high_loss_indices.pkl'
    if read(folder / 'init_complete.json')['indices_sha256'] != file_hash(init):
        raise ValueError('Initialization indices changed')
    bank = folder / 'buffers' / args.dataset / model
    # Validate all banks, including changed files after a failed/restarted job.
    for start in range(0, 100, 5):
        validate_shard(folder, start)
    source = 'TinyImageNet' if tiny else args.dataset
    output = args.synthetic / 'generated' / 'ConvNet' / source / f'{args.dataset}_0.2pct'
    output.mkdir(parents=True, exist_ok=True)
    provenance = output / 'generation_config.json'
    expected = {'recipe': read(folder / 'config.json'),
                'initialization_sha256': file_hash(init), 'iterations': total}
    if provenance.exists():
        if read(provenance) != expected:
            raise ValueError('Synthetic output belongs to another generation recipe')
    else:
        if any(output.iterdir()):
            raise FileExistsError(f'Refusing to overwrite unrecognized synthetic outputs: {output}')
        write(provenance, expected)
    state_path = output / 'training.pt'
    iteration = 0
    if state_path.exists():
        state = torch.load(state_path, map_location='cpu', weights_only=False)
        iteration = state['iteration']
        del state
    if iteration < args.stop:
        command = ['--dataset', args.dataset, '--model', model, '--iters', total,
            '--stop_at', args.stop, '--checkpoint_every', 100, '--eval_it', 1000,
            '--seed', 0, '--train_labels_path', target, '--expert_epochs', 2,
            '--lr_img', 100000 if tiny else 1000, '--lr_labels', 1, '--lr_lr', 1e-4,
            '--lr_teacher', 0.1, '--syn_steps', 10 if tiny else 40,
            '--batch_syn', 256, '--image_init_idx_path', init,
            '--min_start_epoch', 0, '--max_start_epoch', 2,
            '--expert_dir', bank, '--data_path', args.data, '--criterion', 'mse',
            '--run_id', f'{args.dataset}_ConvNet_0.2pct', '--save_dir', output]
        if state_path.exists():
            command.append('--resume')
        run('distill.py', *command)
    state = torch.load(state_path, map_location='cpu', weights_only=False)
    iteration = state['iteration']
    if (state['config']['dataset'] != args.dataset or state['config']['model'] != model
            or state['config']['iters'] != total or state['config']['seed'] != 0):
        raise ValueError('Saved state does not match this MKDT generation')
    if not args.stop <= iteration <= total or tuple(state['images'].shape) != (
            count // 500, 3, 64 if tiny else 32, 64 if tiny else 32):
        raise ValueError('Wrong saved distillation iteration/image budget')
    if not all(torch.isfinite(state[key]).all().item() for key in ('images', 'labels', 'synlr')):
        raise ValueError('Non-finite distilled data')
    mean, std = (([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) if tiny else
        (([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]) if args.dataset == 'CIFAR10'
         else ([0.5071, 0.4865, 0.4409], [0.2009, 0.1984, 0.2023])))
    for step in (0, total):
        image_path = output / f'images_{step}.pt'
        png_path = output / f'images_{step}.png'
        if image_path.exists() and not png_path.exists():
            images = torch.load(image_path, map_location='cpu', weights_only=True)
            images = images * torch.tensor(std)[None, :, None, None] + torch.tensor(mean)[None, :, None, None]
            save_image(images.clamp(0, 1), png_path, nrow=10)
    write(folder / f'distill_{args.stop:05d}.json', {'complete': True, 'iteration': iteration})
    if iteration == total:
        files = {f'{name}_{total}.pt': file_hash(output / f'{name}_{total}.pt')
                 for name in ('images', 'labels', 'synlr')}
        write(output / 'complete_0.2.json', {'complete': True, 'dataset': source,
            'generator': 'ConvNet', 'expert_model': model, 'percentage': 0.2,
            'count': count // 500, 'iterations': total, 'expert_count': 100,
            'expert_epochs': 20, 'files': files, 'recipe': read(folder / 'config.json')})
    print(f'RETAINED state after {iteration}/{total} updates: {state_path}', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('experts', 'init', 'distill'))
    parser.add_argument('dataset', choices=('CIFAR10', 'CIFAR100', 'Tiny'))
    parser.add_argument('stop', type=int)
    parser.add_argument('work', type=Path)
    parser.add_argument('synthetic', type=Path)
    parser.add_argument('data', type=Path)
    args = parser.parse_args()
    if 'SLURM_JOB_ID' not in os.environ:
        raise RuntimeError('Run through the Killarney Slurm launcher')
    if os.environ.get('SLURM_CLUSTER_NAME', os.environ.get('CC_CLUSTER')) != 'killarney':
        raise RuntimeError('Killarney jobs only')
    if args.stage != 'init' and (not torch.cuda.is_available() or torch.cuda.device_count() != 1):
        raise RuntimeError('Exactly one visible CUDA GPU required')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    tiny = args.dataset == 'Tiny'
    model, count = ('ConvNetD4', 100000) if tiny else ('ConvNet', 50000)
    target = args.synthetic / 'work' / 'target_rep' / 'barlow_twins' / f'{args.dataset}_target_rep_train.pt'
    folder = args.work / args.dataset
    folder.mkdir(parents=True, exist_ok=True)
    recipe = {'dataset': args.dataset, 'model': model, 'percentage': 0.2,
              'target_rep_sha256': file_hash(target), 'data_path': str(args.data),
              'code_sha256': (args.work / 'recipe.sha256').read_text().strip(),
              'torch_version': str(torch.__version__)}
    with (folder / '.config.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        config = folder / 'config.json'
        if config.exists() and read(config) != recipe:
            raise ValueError('MKDT input/code/runtime changed; refusing to mix runs')
        write(config, recipe)
    features = torch.load(target, map_location='cpu', weights_only=True)
    if tuple(features.shape) != (count, 512) or not torch.isfinite(features).all().item():
        raise ValueError('Invalid cached full-dataset teacher representations')
    del features
    scope = f'experts_{args.stop:03d}' if args.stage == 'experts' else args.stage
    with (folder / f'.{scope}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.stage == 'experts':
            experts(args, folder, target, model)
        elif args.stage == 'init':
            initialize(folder, args.dataset, count)
        else:
            distill(args, folder, target, model, count)


if __name__ == '__main__':
    main()
