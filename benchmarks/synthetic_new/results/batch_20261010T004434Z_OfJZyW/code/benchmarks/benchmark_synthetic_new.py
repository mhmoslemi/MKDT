"""Benchmark pre-generated CIFAR synthetic sets without modifying older runs."""

import argparse
import csv
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import time


MODELS = ('ConvNet', 'VGG11', 'ResNet18')
LOSSES = ('simclr', 'barlowtwins')
SOURCES = ('CIFAR10', 'CIFAR100')
SIZES = (1, 2, 5)
RUNS = 10
PROTOCOL = 'synthetic_new_v1_random_labels_barlow2x'
SOURCE_SLUGS = {'CIFAR10': 'cifar10', 'CIFAR100': 'cifar100'}
DOWNSTREAM_TARGETS = {
    'CIFAR10': ('CIFAR100', 'Aircraft', 'CUB2011', 'Dogs', 'Flowers', 'TinyImageNet'),
    'CIFAR100': ('TinyImageNet', 'CIFAR10', 'Aircraft', 'CUB2011', 'Dogs', 'Flowers'),
}


def synthetic_path(root, source, size):
    return Path(root) / f'res_IIC_{source}_ConvNet_{size}percent.pt'


def source_id(source, model, ssl_method, size):
    return f'{SOURCE_SLUGS[source]}_{model}_{ssl_method}_ours_size{size}'


def load_synthetic(path, source, size):
    import torch
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(saved, dict) or 'data' not in saved:
        raise ValueError(f'{path}: expected a dictionary containing data')
    images = saved['data']
    expected = (500 * size, 3, 32, 32)
    if tuple(images.shape) != expected or not images.is_floating_point():
        raise ValueError(f'{path}: expected floating tensor {expected}, found {images.shape}')
    if not torch.isfinite(images).all():
        raise ValueError(f'{path}: synthetic images contain non-finite values')
    return saved, images.contiguous()


def preflight(args):
    for source in SOURCES:
        for size in SIZES:
            path = synthetic_path(args.synthetic_root, source, size)
            if not path.is_file():
                raise FileNotFoundError(path)
            saved, images = load_synthetic(path, source, size)
            print(f'VALID source={source} size={size} shape={tuple(images.shape)} '
                  f'iteration={saved.get("iteration")} seed={saved.get("seed")}', flush=True)


def atomic_save(path, value):
    import torch
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def pretrain(args):
    import torch
    import benchmark_cifar10 as shared
    import benchmark_dataset as main_dataset
    from downstream_data import SPECS

    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', '1')))
    path = synthetic_path(args.synthetic_root, args.source, args.subset_percentage)
    saved, images = load_synthetic(path, args.source, args.subset_percentage)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    images = images.to(args.device)
    classes = SPECS[args.source][0]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    settings = dict(
        vars(args), protocol=PROTOCOL, source_protocol=main_dataset.PROTOCOL,
        source_classes=classes, synthetic_path=str(path), synthetic_sha256=digest,
        synthesis_seed=saved.get('seed'), synthesis_iteration=saved.get('iteration'),
        image_space=f'{args.source}_normalized', temperature=0.2, batch_size=256,
        ssl_aug_strategy=shared.STRATEGY, ssl_aug_mode='S', runs=RUNS,
    )
    shared.write_json(output / 'config.json', settings)
    for seed in range(RUNS):
        started = time.perf_counter()
        network = shared.get_network(args.model, 3, classes, (32, 32), seed=seed + 3000)
        steps = shared.train_ssl(network, images, args.ssl_method, args.ssl_epochs,
                                 seed, log_every=args.ssl_log_every)
        checkpoint = {
            'protocol': PROTOCOL,
            'source_protocol': main_dataset.PROTOCOL,
            'source_dataset': args.source,
            'source_classes': classes,
            'seed': seed,
            'model': args.model,
            'method': 'ours',
            'ssl_method': args.ssl_method,
            'subset_percentage': args.subset_percentage,
            'ssl_epochs': args.ssl_epochs,
            'temperature': 0.2,
            'ssl_aug_strategy': shared.STRATEGY,
            'ssl_aug_mode': 'S',
            'synthetic_path': str(path),
            'synthetic_sha256': digest,
            'synthesis_seed': saved.get('seed'),
            'synthesis_iteration': saved.get('iteration'),
            'ssl_steps': steps,
            'state_dict': {key: value.detach().cpu()
                           for key, value in network.state_dict().items()},
        }
        atomic_save(output / f'seed_{seed:03d}.pt', checkpoint)
        print(f'PRETRAIN source={args.source} model={args.model} ssl={args.ssl_method} '
              f'size={args.subset_percentage} seed={seed} steps={steps} '
              f'seconds={time.perf_counter() - started:.1f}', flush=True)
        del network, checkpoint
        gc.collect()


def load_encoder(args, seed):
    import torch
    import benchmark_cifar10 as shared
    import benchmark_dataset as main_dataset
    from downstream_data import SPECS

    path = Path(args.encoder_dir) / f'seed_{seed:03d}.pt'
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    expected = {
        'protocol': PROTOCOL,
        'source_protocol': main_dataset.PROTOCOL,
        'source_dataset': args.source,
        'source_classes': SPECS[args.source][0],
        'seed': seed,
        'model': args.model,
        'method': 'ours',
        'ssl_method': args.ssl_method,
        'subset_percentage': args.subset_percentage,
        'ssl_epochs': args.ssl_epochs,
        'temperature': 0.2,
        'ssl_aug_strategy': shared.STRATEGY,
        'ssl_aug_mode': 'S',
    }
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise ValueError(
                f'{path}: unexpected {key}={checkpoint.get(key)!r}, expected {value!r}')
    network = shared.get_network(args.model, 3, checkpoint['source_classes'],
                                 (32, 32), seed=seed + 3000)
    network.load_state_dict(checkpoint['state_dict'], strict=True)
    network.eval().requires_grad_(False)
    return network


def evaluate(args):
    import torch
    import benchmark_cifar10 as shared
    import benchmark_dataset as main_dataset
    from downstream_data import load_target, select_labels

    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', '1')))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    settings = dict(
        vars(args), protocol=PROTOCOL, source_protocol=main_dataset.PROTOCOL,
        method='ours', runs=RUNS, std_ddof=1,
        label_selection='uniform_random_without_replacement',
        normalization='target_dataset_repository_statistics', image_size=32,
        temperature=0.2, ssl_aug_strategy=shared.STRATEGY, ssl_aug_mode='S',
        probe_protocol='frozen_encoder',
        synthetic_path=str(synthetic_path(
            args.synthetic_root, args.source, args.subset_percentage)),
    )
    shared.write_json(output / 'config.json', settings)
    train, labels, test, test_labels, classes = load_target(
        args.target_cache, args.device, args.target)
    labels_cpu = labels.cpu().numpy()
    rows = []
    for seed in range(RUNS):
        started = time.perf_counter()
        indices = select_labels(labels_cpu, args.label_percentage, seed, 'random')
        chosen = torch.tensor(indices, device=args.device)
        network = load_encoder(args, seed)
        accuracy = main_dataset.linear_probe(
            network, train[chosen], labels[chosen], test, test_labels,
            classes, args.probe_epochs, seed)
        row = {
            'seed': seed,
            'accuracy_percent': accuracy,
            'labeled_indices': indices.tolist(),
            'labeled_count': len(indices),
            'requested_label_percentage': args.label_percentage,
            'probe_epochs': args.probe_epochs,
            'seconds': time.perf_counter() - started,
        }
        shared.write_json(output / f'seed_{seed:03d}.json', row)
        rows.append(row)
        scores = [item['accuracy_percent'] for item in rows]
        summary = {
            'config': settings,
            'complete': len(rows) == RUNS,
            'completed_runs': len(rows),
            'expected_runs': RUNS,
            'mean_percent': statistics.mean(scores),
            'std_percent': statistics.stdev(scores) if len(scores) > 1 else None,
            'seeds': [item['seed'] for item in rows],
            'accuracies_percent': scores,
        }
        shared.write_json(output / 'summary.json', summary)
        print(f'RUN source={args.source} target={args.target} model={args.model} '
              f'ssl={args.ssl_method} size={args.subset_percentage} seed={seed} '
              f'labels={len(indices)} accuracy={accuracy:.4f}% '
              f'seconds={row["seconds"]:.1f}', flush=True)
        del network
        gc.collect()
    print(f'COMPLETE source={args.source} target={args.target} model={args.model} '
          f'n={len(rows)} mean={summary["mean_percent"]:.4f} '
          f'std={summary["std_percent"]}', flush=True)


def replace_ours_rows(text, cells_for_context, downstream=False):
    lines = text.splitlines()
    output = []
    index = 0
    labels = size = None
    model = 'ResNet18' if downstream else None
    while index < len(lines):
        line = lines[index]
        section = re.search(r'\\label\{tab:downstream-(convnet|vgg11|resnet18)\}', line)
        if section:
            model = {'convnet': 'ConvNet', 'vgg11': 'VGG11',
                     'resnet18': 'ResNet18'}[section.group(1)]
        label_match = re.search(r'\\multirow\{17\}\{\*\}\{([15])\\%\}', line)
        if label_match:
            labels = int(label_match.group(1))
        size_match = re.search(r'\\multirow\{5\}\{\*\}\{([125])\\%\}', line)
        if size_match:
            size = int(size_match.group(1))
        if not re.match(r'&\s*&\s*Ours(?=\s*(?:&|$))', line):
            output.append(line)
            index += 1
            continue
        end = index
        while end < len(lines) and not lines[end].rstrip().endswith('\\\\'):
            end += 1
        cells = cells_for_context(labels, size, model)
        if cells is None or not any(cells):
            output.extend(lines[index:end + 1])
        else:
            output.append('& & Ours & ' + ' & '.join(cells) + r' \\')
        index = end + 1
    return '\n'.join(output) + '\n'


def report(args):
    batch = Path(args.batch)
    with (batch / 'configs.tsv').open() as handle:
        rows = list(csv.DictReader(handle, delimiter='\t'))
    scores = {}
    for row in rows:
        paths = list((batch / 'evaluations' / row['config_id']).glob('job_*/summary.json'))
        if len(paths) > 1:
            raise ValueError(f'Multiple results for {row["config_id"]}: {paths}')
        summary = json.loads(paths[0].read_text()) if paths else {}
        complete = (
            summary.get('complete')
            and summary.get('completed_runs') == RUNS
            and summary.get('expected_runs') == RUNS
            and sorted(summary.get('seeds', [])) == list(range(RUNS))
        )
        row.update(completed_runs=summary.get('completed_runs', 0),
                   status='complete' if complete else 'incomplete',
                   mean_percent='', std_percent='')
        if complete:
            values = summary['accuracies_percent']
            mean, std = statistics.mean(values), statistics.stdev(values)
            row.update(mean_percent=mean, std_percent=std)
            key = (row['source'], row['target'], row['model'], row['ssl_method'],
                   int(row['subset_percentage']), int(row['label_percentage']))
            scores[key] = f'\\dsacc{{{mean:.2f}}}{{{std:.2f}}}'
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'results.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    templates = Path(args.templates)
    for source in SOURCES:
        slug = SOURCE_SLUGS[source]

        def main_cells(labels, size, unused_model):
            if labels not in (1, 5) or size not in SIZES:
                return None
            return [scores.get((source, source, model, ssl, size, labels), '')
                    for model in MODELS for ssl in LOSSES]

        main_template = templates / f'{slug}_main.tex'
        (output / f'{slug}_main.tex').write_text(
            replace_ours_rows(main_template.read_text(), main_cells))

        def downstream_cells(labels, size, model):
            if model != 'ResNet18' or labels not in (1, 5) or size not in SIZES:
                return None
            return [scores.get((source, target, 'ResNet18', ssl, size, labels), '')
                    for target in DOWNSTREAM_TARGETS[source] for ssl in LOSSES]

        downstream_template = templates / f'{slug}_downstream.tex'
        (output / f'{slug}_downstream.tex').write_text(
            replace_ours_rows(
                downstream_template.read_text(), downstream_cells, downstream=True))
    complete = sum(row['status'] == 'complete' for row in rows)
    print(f'Complete: {complete}/{len(rows)} configurations. Tables: {output}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('preflight', 'pretrain', 'evaluate', 'report'))
    parser.add_argument('--synthetic-root')
    parser.add_argument('--source', choices=SOURCES)
    parser.add_argument('--target', choices=(
        'CIFAR10', 'CIFAR100', 'Aircraft', 'CUB2011', 'Dogs', 'Flowers', 'TinyImageNet'))
    parser.add_argument('--target-cache')
    parser.add_argument('--encoder-dir')
    parser.add_argument('--output')
    parser.add_argument('--batch')
    parser.add_argument('--templates')
    parser.add_argument('--model', choices=MODELS)
    parser.add_argument('--ssl-method', choices=LOSSES)
    parser.add_argument('--subset-percentage', type=int, choices=SIZES)
    parser.add_argument('--ssl-epochs', type=int)
    parser.add_argument('--probe-epochs', type=int)
    parser.add_argument('--label-percentage', type=int, choices=(1, 5))
    parser.add_argument('--ssl-log-every', type=int, default=0)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    args = parser.parse_args()
    if args.command == 'preflight':
        if not args.synthetic_root:
            parser.error('preflight requires --synthetic-root')
    elif args.command == 'pretrain':
        required = (args.synthetic_root, args.source, args.output, args.model,
                    args.ssl_method, args.subset_percentage, args.ssl_epochs)
        if not all(value is not None for value in required) or args.ssl_epochs < 1:
            parser.error('pretrain requires synthetic/source/output/model/SSL/size/epochs')
    elif args.command == 'evaluate':
        required = (args.synthetic_root, args.source, args.target, args.target_cache,
                    args.encoder_dir, args.output, args.model, args.ssl_method,
                    args.subset_percentage, args.ssl_epochs, args.probe_epochs,
                    args.label_percentage)
        if not all(value is not None for value in required):
            parser.error('evaluate is missing required arguments')
    else:
        if not args.batch or not args.templates or not args.output:
            parser.error('report requires batch/templates/output')
    if args.command in ('pretrain', 'evaluate') and args.device == 'cuda':
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('A GPU allocation is required')
    {'preflight': preflight, 'pretrain': pretrain,
     'evaluate': evaluate, 'report': report}[args.command](args)


if __name__ == '__main__':
    main()
