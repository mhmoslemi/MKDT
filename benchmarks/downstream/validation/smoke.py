"""Short GPU checkpoint/transfer tests with synthetic target caches, no target downloads."""

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'OT-SSL-DD'))
import benchmark_downstream as downstream
import downstream_data as data
import torch


def main():
    torch.set_num_threads(1)
    root = ROOT / 'benchmarks/downstream/validation' / f'smoke_{os.environ["SLURM_JOB_ID"]}'
    root.mkdir()
    args = argparse.Namespace(output='', data_path=os.path.join(os.environ['SCRATCH'], 'data'),
        device='cuda', runs=2, model='ConvNet', method='random', ssl_method='simclr',
        subset_percentage=1, ssl_epochs=1, selection_dir=None, encoder_dir=None,
        target='CIFAR100', target_cache='', label_percentage=1, label_policy='exact', probe_updates=2)
    for target, (classes, mean, std) in data.SPECS.items():
        # Several images per class make 1% and 5% distinct, with many unlabeled classes.
        n_train = classes * 10
        cache = dict(protocol=data.CACHE_PROTOCOL, dataset=target, num_classes=classes,
            mean=mean, std=std, train_images=torch.randint(0, 256, (n_train, 3, 32, 32), dtype=torch.uint8),
            train_labels=torch.arange(n_train) % classes,
            test_images=torch.randint(0, 256, (classes, 3, 32, 32), dtype=torch.uint8),
            test_labels=torch.arange(classes))
        torch.save(cache, root / f'{target}.pt')
    parameter_sizes = {}
    for model in ('ConvNet', 'VGG11', 'ResNet18'):
        for ssl in ('simclr', 'barlowtwins'):
            args.model, args.ssl_method, args.method = model, ssl, 'random'
            args.subset_percentage = 1
            args.output = str(root / f'encoder_{model}_{ssl}')
            downstream.pretrain(args)
            args.encoder_dir = args.output
            for target in data.SPECS:
                args.target = target
                args.target_cache = str(root / f'{target}.pt')
                args.output = str(root / f'eval_{model}_{ssl}_{target}')
                downstream.evaluate(args)
        args.method, args.ssl_method, args.subset_percentage = 'no_pretrain', 'none', 0
        args.label_percentage = 5
        args.output = str(root / f'supervised_{model}')
        downstream.evaluate(args)
        args.label_percentage = 1
        net = downstream.source.get_network(model, 3, 10, (32, 32), seed=0)
        parameter_sizes[model] = sum(value.numel() * value.element_size() for value in net.state_dict().values())
        del net
    gb = sum(parameter_sizes.values()) * 14 * 15 / 1e9
    print(f'Estimated complete encoder checkpoint storage: {gb:.2f} GB', flush=True)
    print('GPU SMOKE PASSED: every architecture/loss, all five class counts, frozen transfer, and scratch training.', flush=True)


if __name__ == '__main__':
    main()
