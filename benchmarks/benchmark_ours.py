"""Synthesize CIFAR-10 once per size, then benchmark frozen probes over 15 seeds."""

import argparse
import csv
import gc
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import statistics
import tarfile
import time

MODELS = ('ConvNet', 'VGG11', 'ResNet18')
LOSSES = ('simclr', 'barlowtwins')
TARGETS = ('CIFAR10', 'CIFAR100', 'Aircraft', 'CUB2011', 'Dogs', 'Flowers', 'TinyImageNet')
SIZES = (1, 2, 5)
RUNS = 15


def write_tsv(path, rows):
    with path.open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)


def synthetic_path(args):
    return Path(args.batch) / 'synthetic' / f'size{args.subset_percentage}' / f'res_IIC_CIFAR10_ConvNet_{args.subset_percentage}percent.pt'


def source_id(model, ssl, size):
    return f'{model}_{ssl}_ours_size{size}'


def pretrain(args):
    import torch
    import benchmark_cifar10 as source
    import benchmark_downstream as downstream
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', '1')))
    path = synthetic_path(args)
    saved = torch.load(path, map_location='cpu', weights_only=True)
    images = saved['data']
    if images.shape != (500 * args.subset_percentage, 3, 32, 32) or not torch.isfinite(images).all():
        raise ValueError(f'Invalid synthetic images: {path}')
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    images = images.to(args.device)
    output = Path(args.batch) / 'encoders' / source_id(args.model, args.ssl_method, args.subset_percentage)
    output.mkdir(parents=True, exist_ok=False)
    source.write_json(output / 'config.json', dict(vars(args), synthetic_path=str(path), synthetic_sha256=digest, synthesis_seed=saved['seed'], synthesis_iteration=saved['iteration'], image_space='CIFAR10_normalized'))
    for seed in range(RUNS):
        started = time.perf_counter()
        network = source.get_network(args.model, 3, 10, (32, 32), seed=seed + 3000)
        steps = source.train_ssl(network, images, args.ssl_method, args.ssl_epochs, seed, log_every=args.ssl_log_every)
        checkpoint = dict(protocol=downstream.PROTOCOL, source_protocol=source.PROTOCOL, seed=seed, model=args.model, method='ours', ssl_method=args.ssl_method, subset_percentage=args.subset_percentage, ssl_epochs=args.ssl_epochs, temperature=0.2, ssl_aug_strategy=source.STRATEGY, ssl_aug_mode='S', synthetic_path=str(path), synthetic_sha256=digest, synthesis_seed=saved['seed'], synthesis_iteration=saved['iteration'], ssl_steps=steps, state_dict={key: value.detach().cpu() for key, value in network.state_dict().items()})
        downstream.atomic_save(output / f'seed_{seed:03d}.pt', checkpoint)
        print(f'PRETRAIN seed={seed} seconds={time.perf_counter() - started:.1f}', flush=True)
        del network, checkpoint
        gc.collect()


def evaluate(args):
    import torch
    import benchmark_cifar10 as source
    import benchmark_downstream as downstream
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', '1')))
    args.method, args.runs, args.label_policy = 'ours', RUNS, 'exact'
    args.encoder_dir = str(Path(args.batch) / 'encoders' / source_id(args.model, args.ssl_method, args.subset_percentage))
    args.synthetic_path = str(synthetic_path(args))
    source.seed_all(0)
    if args.target != 'CIFAR10':
        downstream.evaluate(args)
        return
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    settings = dict(vars(args), protocol=source.PROTOCOL, std_ddof=1, ssl_aug_strategy=source.STRATEGY, ssl_aug_mode='S', temperature=0.2, batch_size=256, probe_protocol='frozen_encoder')
    source.write_json(output / 'config.json', settings)
    train, labels, test, test_labels = source.load_data(args.data_path, args.device)
    labels_cpu = labels.cpu().numpy()
    rows = []
    for seed in range(RUNS):
        started = time.perf_counter()
        indices = source.labeled_indices(labels_cpu, args.label_percentage, seed)
        chosen = torch.tensor(indices, device=args.device)
        network = downstream.load_encoder(args, seed)
        accuracy = source.linear_probe(network, train[chosen], labels[chosen], test, test_labels, args.probe_epochs, seed)
        row = dict(seed=seed, accuracy_percent=accuracy, labeled_indices=indices.tolist(), probe_epochs=args.probe_epochs, seconds=time.perf_counter() - started)
        source.write_json(output / f'seed_{seed:03d}.json', row)
        rows.append(row)
        values = [item['accuracy_percent'] for item in rows]
        source.write_json(output / 'summary.json', dict(config=settings, complete=len(rows) == RUNS, completed_runs=len(rows), expected_runs=RUNS, seeds=list(range(len(rows))), accuracies_percent=values, mean_percent=statistics.mean(values), std_percent=statistics.stdev(values) if len(values) > 1 else None))
        print(f'RUN seed={seed} accuracy={accuracy:.4f}% seconds={row["seconds"]:.1f}', flush=True)
        del network
        gc.collect()
    print(f'COMPLETE mean={statistics.mean(values):.4f} std={statistics.stdev(values):.4f} n={len(rows)}', flush=True)


def fill_ours(path, scores, downstream):
    lines = path.read_text().splitlines()
    output, index, filled, labels, size, model = [], 0, 0, None, None, None
    while index < len(lines):
        line = lines[index]
        match = re.search(r'\\label\{tab:downstream-(convnet|vgg11|resnet18)\}', line)
        if match:
            model = dict(convnet='ConvNet', vgg11='VGG11', resnet18='ResNet18')[match[1]]
        match = re.search(r'\\multirow\{17\}\{\*\}\{([15])\\%\}', line)
        if match:
            labels = int(match[1])
        match = re.search(r'\\multirow\{5\}\{\*\}\{([125])\\%\}', line)
        if match:
            size = int(match[1])
        match = re.match(r'(&\s*&\s*Ours)(?=\s*(?:&|$))', line)
        if not match:
            output.append(line)
            index += 1
            continue
        end = index
        while not lines[end].rstrip().endswith('\\\\'):
            end += 1
        pairs = [(target, model) for target in TARGETS[1:]] if downstream else [('CIFAR10', backbone) for backbone in MODELS]
        keys = [(target, backbone, ssl, size, labels) for target, backbone in pairs for ssl in LOSSES]
        if not any(key in scores for key in keys):
            output.extend(lines[index:end + 1])
        else:
            assert labels in (1, 5) and size in SIZES and (not downstream or model in MODELS)
            output.append(match[1])
            for pair_index, (target, backbone) in enumerate(pairs):
                cells = [scores.get((target, backbone, ssl, size, labels), '') for ssl in LOSSES]
                filled += sum(bool(cell) for cell in cells)
                output.append('& ' + ' & '.join(cells) + (' \\\\' if pair_index == len(pairs) - 1 else ''))
        index = end + 1
    path.write_text('\n'.join(output) + '\n')
    print(f'Updated {path}: {filled} Ours cells.', flush=True)


def report(args):
    batch = Path(args.batch)
    with (batch / 'configs.tsv').open() as handle:
        rows = list(csv.DictReader(handle, delimiter='\t'))
    scores = {}
    for row in rows:
        paths = list((batch / 'evaluations' / row['config_id']).glob('job_*/summary.json'))
        if len(paths) > 1:
            raise ValueError(f'Multiple results: {row["config_id"]}')
        summary = json.loads(paths[0].read_text()) if paths else {}
        complete = summary.get('complete') and summary.get('completed_runs') == RUNS and summary.get('expected_runs') == RUNS and sorted(summary.get('seeds', [])) == list(range(RUNS))
        row.update(completed_runs=summary.get('completed_runs', 0), status='complete' if complete else 'incomplete', mean_percent='', std_percent='')
        if complete:
            values = summary['accuracies_percent']
            assert len(values) == RUNS
            mean, std = statistics.mean(values), statistics.stdev(values)
            row.update(mean_percent=mean, std_percent=std)
            scores[(row['target'], row['model'], row['ssl_method'], int(row['subset_percentage']), int(row['label_percentage']))] = f'\\dsacc{{{mean:.2f}}}{{{std:.2f}}}'
    with (batch / 'tables.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    fill_ours(Path(args.project_root) / 'benchmarks/res.tex', scores, False)
    fill_ours(Path(args.project_root) / 'benchmarks/down.tex', scores, True)
    print(f'Complete: {len(scores)}/{len(rows)} configurations.', flush=True)


def make_jobs(args):
    root, batch = Path(args.project_root), Path(args.batch)
    settings = json.loads((batch / 'code/OT-SSL-DD/training_epochs.json').read_text())
    timings = json.loads((root / 'benchmarks/timing.json').read_text())['models']
    epochs = settings['pretraining']['random_subset_epochs_by_data_percentage']
    assert epochs == settings['pretraining']['kmeans_subset_epochs_by_data_percentage']
    assert settings['runs_per_configuration'] == RUNS
    target_root = Path(args.target_cache)
    for target in TARGETS[1:]:
        if not (target_root / f'{target}.pt').is_file():
            raise FileNotFoundError(target_root / f'{target}.pt')
    jobs, configs, scripts = [], [], {}

    def add(stage, name, dependency, limit, command, gpu=True):
        header = ['#!/bin/bash', '#SBATCH --account=aip-yiweilu', f'#SBATCH --job-name=ours_{name}', f'#SBATCH --time={limit}', '#SBATCH --nodes=1', '#SBATCH --ntasks=1', '#SBATCH --cpus-per-task=1', '#SBATCH --mem=' + ('15G' if gpu else '1G')]
        if gpu:
            header.append('#SBATCH --gres=gpu:l40s:1')
        header.extend(['set -euo pipefail', 'cd "$OURS_CODE_ROOT"', 'module load StdEnv/2023 python/3.11.5', 'source "${VENV_ACTIVATE:-$HOME/ENV/bin/activate}"', 'export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1', 'export HF_HOME="$SCRATCH/hf_cache" XDG_CACHE_HOME="$SCRATCH/cache"', command])
        member = f'{stage}/{name}.sh'
        scripts[member] = '\n'.join(header) + '\n'
        jobs.append(dict(stage=stage, key=name, dependency=dependency or '-', walltime=limit, script=member))

    for size in SIZES:
        add('synthesize', f'syn_size{size}', '', '0-02:00:00', f'[[ ! -e "$OURS_BATCH/synthetic/size{size}" ]]\nsrun --unbuffered python3 OT-SSL-DD/mina_IF.py --percentage {size} --data_path "$DATA_ROOT" --save_path "$OURS_BATCH/synthetic/size{size}"')
    for model in MODELS:
        for ssl in LOSSES:
            for size in SIZES:
                name = source_id(model, ssl, size)
                ssl_epochs = epochs[str(size)]
                seconds = RUNS * (math.ceil(500 * size / 256) * ssl_epochs * timings[model][f'{ssl}_step_seconds'] + 5)
                minutes = math.ceil(seconds / 300) * 5 + 60
                limit = f'0-{minutes // 60:02d}:{minutes % 60:02d}:00'
                common = f'--batch "$OURS_BATCH" --model {model} --ssl-method {ssl} --subset-percentage {size} --ssl-epochs {ssl_epochs}'
                add('pretrain', name, f'syn_size{size}', limit, f'srun --unbuffered python3 benchmarks/benchmark_ours.py pretrain {common}')
                for target in TARGETS:
                    for labels in (1, 5):
                        probe_epochs = settings['evaluation_by_dataset'][target]['linear_probe_epochs_by_label_percentage'][str(labels)]
                        config_id = f'{target}_{name}_lbl{labels}'
                        cache = f' --target-cache "$TARGET_CACHE_ROOT/{target}.pt"' if target != 'CIFAR10' else ''
                        command = f'srun --unbuffered python3 benchmarks/benchmark_ours.py evaluate {common} --target {target} --label-percentage {labels} --probe-epochs {probe_epochs} --data-path "$DATA_ROOT" --output "$OURS_BATCH/evaluations/{config_id}/job_$SLURM_JOB_ID"{cache}'
                        add('evaluate', config_id, name, '0-01:05:00', command)
                        configs.append(dict(config_id=config_id, target=target, model=model, method='ours', ssl_method=ssl, subset_percentage=size, label_percentage=labels, ssl_epochs=ssl_epochs, probe_epochs=probe_epochs, runs=RUNS))
    add('report', 'tables', 'all_evaluations', '0-00:05:00', 'python3 benchmarks/benchmark_ours.py report --batch "$OURS_BATCH" --project-root "$PROJECT_ROOT"', False)
    with tarfile.open(batch / 'jobs.tar', 'w') as archive:
        for name, content in scripts.items():
            data = content.encode()
            member = tarfile.TarInfo(name)
            member.size, member.mode = len(data), 0o755
            archive.addfile(member, io.BytesIO(data))
    write_tsv(batch / 'jobs.tsv', jobs)
    write_tsv(batch / 'configs.tsv', configs)
    (batch / 'protocol.json').write_text(json.dumps(dict(method='ours', synthesis_runs_per_size=1, synthesis_seed=0, synthesis_checkpoint='final_iteration', mina_sha256=hashlib.sha256((batch / 'code/OT-SSL-DD/mina_IF.py').read_bytes()).hexdigest(), synthetic_counts={str(size): size * 500 for size in SIZES}, repetitions=RUNS, uncertainty='sample_std_across_encoder_and_probe_seeds_conditional_on_one_synthetic_set', epochs=settings, account='aip-yiweilu', target_cache_root=str(target_root)), indent=2) + '\n')
    print(f'Created {len(jobs)} jobs: 3 synthesis, 18 shared SSL pretraining, {len(configs)} evaluations, 1 report.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['make-jobs', 'pretrain', 'evaluate', 'report'])
    parser.add_argument('--batch', required=True)
    parser.add_argument('--project-root')
    parser.add_argument('--target-cache')
    parser.add_argument('--model', choices=MODELS)
    parser.add_argument('--ssl-method', choices=LOSSES)
    parser.add_argument('--subset-percentage', type=int, choices=SIZES)
    parser.add_argument('--ssl-epochs', type=int)
    parser.add_argument('--ssl-log-every', type=int, default=0, help='Print SSL loss every N epochs; 0 prints four times per run')
    parser.add_argument('--probe-epochs', type=int)
    parser.add_argument('--target', choices=TARGETS)
    parser.add_argument('--label-percentage', type=int, choices=[1, 5])
    parser.add_argument('--data-path', default=os.path.join(os.environ.get('SCRATCH', '.'), 'data'))
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--output')
    args = parser.parse_args()
    if args.ssl_log_every < 0:
        parser.error('--ssl-log-every must be nonnegative')
    {'make-jobs': make_jobs, 'pretrain': pretrain, 'evaluate': evaluate, 'report': report}[args.command](args)


if __name__ == '__main__':
    main()
