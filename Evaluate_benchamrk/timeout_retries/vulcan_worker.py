"""Vulcan entry point for the unchanged expert and SSL recovery routines."""
import argparse
import fcntl
import functools
import importlib.util
import os
from pathlib import Path
import sys

import torch

BENCH = Path(__file__).resolve().parent.parent


def load_worker(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    return worker


def main():
    if (not os.environ.get('SLURM_JOB_ID')
            or os.environ.get('SLURM_CLUSTER_NAME', os.environ.get('CC_CLUSTER')) != 'vulcan'
            or os.environ.get('SLURM_JOB_ACCOUNT') != 'aip-boyuwang'
            or not torch.cuda.is_available() or torch.cuda.device_count() != 1):
        raise RuntimeError('Use one Wang-account GPU job on Vulcan')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    stage, *arguments = sys.argv[1:]
    if stage == 'experts':
        code, dataset, start, work, synthetic, data = arguments
        code, work, synthetic, data = (Path(p).resolve() for p in (code, work, synthetic, data))
        start = int(start)
        if (dataset not in ('CIFAR10', 'CIFAR100') or start not in range(0, 100, 5)
                or work != BENCH / 'mkdt_02_jobs/runs'
                or not code.is_relative_to(BENCH / 'timeout_retries/vulcan')):
            raise ValueError('Unexpected expert retry scope or frozen-code path')
        sys.path[:0] = [str(code), str(BENCH.parent)]
        worker = load_worker('vulcan_experts', code / 'worker.py')
        folder = work / dataset
        target = synthetic / 'work/target_rep/barlow_twins' / f'{dataset}_target_rep_train.pt'
        expected = {'dataset': dataset, 'model': 'ConvNet', 'percentage': 0.2,
            'target_rep_sha256': worker.file_hash(target), 'data_path': str(data),
            'code_sha256': (work / 'recipe.sha256').read_text().strip(), 'torch_version': str(torch.__version__)}
        if worker.read(folder / 'config.json') != expected:
            raise ValueError('Saved expert recipe, target features or PyTorch version changed')
        features = torch.load(target, map_location='cpu', weights_only=True)
        if tuple(features.shape) != (50000, 512) or not torch.isfinite(features).all().item():
            raise ValueError('Invalid cached teacher representations')
        del features
        with (folder / f'.experts_{start:03d}.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            worker.experts(argparse.Namespace(dataset=dataset, stop=start, data=data), folder, target, 'ConvNet')
    elif stage == 'ssl':
        manifest, data, synthetic = (Path(p).resolve() for p in arguments)
        if not manifest.is_relative_to(BENCH / 'downstream_finish_jobs/batches'):
            raise ValueError('Unexpected original SSL manifest')
        worker = load_worker('vulcan_ssl', BENCH / 'downstream_finish_jobs/worker.py')
        job = worker.base.read(manifest)
        allowed = {('mkdt', 1, 0), ('mkdt', 1, 1), ('ours', 1, 1), ('mkdt', 2, 0), ('mkdt', 2, 1)}
        unit = job['unit']
        if (job['stage'] != 'ssl' or unit['source'] != 'TinyImageNet' or unit['ssl'] != 'simclr'
                or unit['epochs'] != 2000 or (unit['kind'], unit['subset'], unit['seed']) not in allowed):
            raise ValueError('Unexpected SSL retry scope')

        @functools.lru_cache(maxsize=1)
        def datasets(name):
            return worker.base.RawDataset(data, name, True), worker.base.RawDataset(data, name, False)

        worker.train(unit, argparse.Namespace(data_root=data, synthetic_root=synthetic), datasets)
    else:
        raise ValueError('Only expert and SSL retries are supported')
    print('COMPLETE Vulcan retry:', stage, flush=True)


if __name__ == '__main__':
    main()
