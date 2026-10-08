"""Exercise actual CIFAR, every architecture/loss, full-data and K-means paths."""

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'OT-SSL-DD'))
import benchmark_cifar10 as bench


def main():
    bench.torch.set_num_threads(1)
    root = ROOT / 'benchmarks/cifar10/validation' / f'smoke_{os.environ["SLURM_JOB_ID"]}'
    root.mkdir()
    args = argparse.Namespace(output=str(root / 'selections'),
        data_path=os.path.join(os.environ['SCRATCH'], 'data'), device='cuda',
        runs=2, seed_start=0, selection_dir=str(root / 'selections'),
        method='random', model='ConvNet', ssl_method='simclr',
        subset_percentage=1, label_percentage=1, ssl_epochs=1, probe_epochs=1)
    bench.prepare(args)
    for model in ('ConvNet', 'VGG11', 'ResNet18'):
        for ssl in ('simclr', 'barlowtwins'):
            args.model, args.ssl_method, args.method = model, ssl, 'random'
            args.output = str(root / f'{model}_{ssl}')
            bench.benchmark(args)
    for method, size, labels in (('kmeans', 5, 5), ('full', 100, 5), ('no_pretrain', 0, 1)):
        args.method, args.subset_percentage, args.label_percentage = method, size, labels
        args.ssl_method = 'none' if method == 'no_pretrain' else 'barlowtwins'
        args.output = str(root / method)
        bench.benchmark(args)
    print('SMOKE PASSED: all architectures/losses, all methods, two seeds per case.', flush=True)


if __name__ == '__main__':
    main()
