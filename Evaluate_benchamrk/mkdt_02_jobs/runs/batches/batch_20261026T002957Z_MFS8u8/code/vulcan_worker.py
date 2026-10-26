"""Vulcan dispatch for the unchanged, frozen MKDT generation functions."""
import argparse
import fcntl
import importlib.util
import os
from pathlib import Path
import sys

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('experts', 'init', 'distill'))
    parser.add_argument('dataset', choices=('CIFAR10', 'CIFAR100', 'Tiny'))
    parser.add_argument('stop', type=int)
    parser.add_argument('work', type=Path)
    parser.add_argument('synthetic', type=Path)
    parser.add_argument('data', type=Path)
    args = parser.parse_args()
    account = 'aip-yiweilu' if args.dataset == 'CIFAR100' else 'aip-boyuwang'
    if (not os.environ.get('SLURM_JOB_ID')
            or os.environ.get('SLURM_CLUSTER_NAME', os.environ.get('CC_CLUSTER')) != 'vulcan'
            or os.environ.get('SLURM_JOB_ACCOUNT') != account):
        parser.error('Use the saved MKDT Vulcan submitter and its selected account')
    if args.stage != 'init' and (not torch.cuda.is_available() or torch.cuda.device_count() != 1):
        parser.error('Generation/expert stages require one allocated GPU')
    code = Path(__file__).resolve().parent
    args.work, args.synthetic, args.data = (p.resolve() for p in (args.work, args.synthetic, args.data))
    if not code.is_relative_to(args.work / 'batches') or code.name != 'code':
        raise ValueError('Use the frozen generation snapshot, not the live source directory')
    sys.path[:0] = [str(code), str(args.work.parents[2])]
    spec = importlib.util.spec_from_file_location('frozen_mkdt_generation', code / 'worker.py')
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    tiny = args.dataset == 'Tiny'
    model, count = ('ConvNetD4', 100000) if tiny else ('ConvNet', 50000)
    folder = args.work / args.dataset
    folder.mkdir(parents=True, exist_ok=True)
    target = args.synthetic / 'work/target_rep/barlow_twins' / f'{args.dataset}_target_rep_train.pt'
    expected = {'dataset': args.dataset, 'model': model, 'percentage': 0.2,
        'target_rep_sha256': worker.file_hash(target), 'data_path': str(args.data),
        'code_sha256': (args.work / 'recipe.sha256').read_text().strip(),
        'torch_version': str(torch.__version__)}
    with (folder / '.config.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        config = folder / 'config.json'
        if config.exists() and worker.read(config) != expected:
            raise ValueError('Saved MKDT recipe, teacher features or PyTorch runtime changed')
        worker.write(config, expected)
    features = torch.load(target, map_location='cpu', weights_only=True)
    if tuple(features.shape) != (count, 512) or not torch.isfinite(features).all().item():
        raise ValueError('Invalid original full-data teacher representations')
    del features
    scope = f'experts_{args.stop:03d}' if args.stage == 'experts' else args.stage
    with (folder / f'.{scope}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.stage == 'experts':
            worker.experts(args, folder, target, model)
        elif args.stage == 'init':
            worker.initialize(folder, args.dataset, count)
        else:
            worker.distill(args, folder, target, model, count)


if __name__ == '__main__':
    main()
