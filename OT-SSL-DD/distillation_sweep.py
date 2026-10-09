"""Generate the requested distillation sweep and evaluate one saved synthetic set."""

import argparse
import csv
import gc
import io
import json
import math
import statistics
import sys
import tarfile
from pathlib import Path


PAIRS = ((0.005, 1000), (0.002, 2000))
NETWORKS = (20, 10, 5, 1)
ENCODER_DATA = ('real', 'synthetic')
ENCODER_STEPS = (1, 2, 5, 10, 20, 50)
PERCENTAGES = (1, 2, 5)


def tag(lr, iterations, networks, encoder_data, steps, percentage):
    lr_tag = str(lr).replace('0.', '')
    return f'lr{lr_tag}_iter{iterations}_net{networks}_{encoder_data}_steps{steps}_pct{percentage}'


def time_limit(iterations, networks, steps, percentage, ssl_epochs):
    measured = {1: 13.2, 2: 16.2, 5: 24.0}[percentage]
    distill_minutes = 3 + (measured - 3) * (iterations / 505) * (networks / 5) * ((steps + 5) / 25)
    eval_steps = math.ceil(500 * percentage / 256) * ssl_epochs
    eval_minutes = 5 * eval_steps * 0.01454434720799327 / 60 + 10
    minutes = math.ceil((1.5 * distill_minutes + eval_minutes) / 60) * 60 + 60
    return f'{minutes // 1440}-{minutes // 60 % 24:02d}:00:00', round(distill_minutes + eval_minutes, 1)


def make_jobs(args):
    root = Path(args.root).resolve()
    project = Path(args.project_root).resolve()
    settings = json.loads((project / 'OT-SSL-DD/training_epochs.json').read_text())
    ssl_epochs_by_size = settings['pretraining']['random_subset_epochs_by_data_percentage']
    probe_epochs = int(settings['evaluation_by_dataset']['CIFAR10']['linear_probe_epochs_by_label_percentage'][str(args.label_percentage)])
    scripts, rows = {}, []
    for lr, iterations in PAIRS:
        for networks in NETWORKS:
            for encoder_data in ENCODER_DATA:
                for steps in ENCODER_STEPS:
                    for percentage in PERCENTAGES:
                        config_id = tag(lr, iterations, networks, encoder_data, steps, percentage)
                        ssl_epochs = int(ssl_epochs_by_size[str(percentage)])
                        limit, estimated = time_limit(iterations, networks, steps, percentage, ssl_epochs)
                        result_name = f'res_IIC_CIFAR10_ConvNet_{percentage}percent.pt'
                        selection = '\n'.join((
                            f'lr_img={lr}', f'Iteration={iterations}', f'num_random_networks={networks}',
                            f'encoder_train_data={encoder_data}', f'encoder_train_steps={steps}',
                            f'percentage={percentage}', 'eval_runs=5', 'eval_model=ConvNet',
                            'eval_ssl_method=simclr', 'eval_label_percentage=1',
                            f'eval_ssl_epochs={ssl_epochs}', f'eval_probe_epochs={probe_epochs}',
                        ))
                        script = f'''#!/bin/bash
#SBATCH --account=aip-yiweilu
#SBATCH --job-name=syn_{config_id}
#SBATCH --time={limit}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=15G
#SBATCH --gres=gpu:l40s:1
set -euo pipefail
cd "$SWEEP_CODE_ROOT"
module load StdEnv/2023 python/3.11.5
source "${{VENV_ACTIVATE:-$HOME/ENV/bin/activate}}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1
export HF_HOME="$SCRATCH/hf_cache" XDG_CACHE_HOME="$SCRATCH/cache"
OUT="$SWEEP_ROOT/{config_id}"
[[ ! -e "$OUT" ]] || {{ echo "Refusing to replace $OUT"; exit 1; }}
mkdir "$OUT"
srun --unbuffered python3 OT-SSL-DD/mina_IF_distill_sweep.py --data_path "$DATA_ROOT" --save_path "$OUT" --device cuda --percentage {percentage} --Iteration {iterations} --lr_img {lr} --num_random_networks {networks} --encoder_train_data {encoder_data} --encoder_train_steps {steps} --num_eval 0
cat > "$OUT/selection.txt" <<'SELECTION'
{selection}
SELECTION
srun --unbuffered python3 OT-SSL-DD/distillation_sweep.py evaluate --checkpoint "$OUT/{result_name}" --percentage {percentage} --data-path "$DATA_ROOT" --epochs-json OT-SSL-DD/training_epochs.json --output "$OUT/eval.txt"
'''
                        member = f'jobs/{config_id}.sh'
                        scripts[member] = script
                        rows.append(dict(config_id=config_id, lr_img=lr, iterations=iterations,
                                         num_random_networks=networks, encoder_train_data=encoder_data,
                                         encoder_train_steps=steps, percentage=percentage, eval_runs=5,
                                         eval_ssl_method='simclr', eval_label_percentage=1,
                                         eval_ssl_epochs=ssl_epochs, eval_probe_epochs=probe_epochs,
                                         estimated_minutes=estimated, walltime=limit, script=member))
    root.mkdir(parents=True, exist_ok=False)
    (root / 'logs').mkdir()
    with (root / 'configs.tsv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)
    with tarfile.open(root / 'jobs.tar', 'w') as archive:
        for name, content in scripts.items():
            data = content.encode()
            member = tarfile.TarInfo(name)
            member.size, member.mode = len(data), 0o755
            archive.addfile(member, io.BytesIO(data))
    print(f'Generated {len(rows)} jobs in {root}', flush=True)


def evaluate(args):
    import torch

    code_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(code_root / 'benchmarks'))
    import benchmark_cifar10 as bench

    settings = json.loads(Path(args.epochs_json).read_text())
    ssl_epochs = int(settings['pretraining']['random_subset_epochs_by_data_percentage'][str(args.percentage)])
    probe_epochs = int(settings['evaluation_by_dataset']['CIFAR10']['linear_probe_epochs_by_label_percentage']['1'])
    checkpoint = Path(args.checkpoint)
    synthetic = torch.load(checkpoint, map_location='cpu', weights_only=True)['data'].to('cuda')
    expected_shape = (500 * args.percentage, 3, 32, 32)
    if tuple(synthetic.shape) != expected_shape:
        raise ValueError(f'Unexpected synthetic shape {tuple(synthetic.shape)}; expected {expected_shape}')
    output = Path(args.output)
    with output.open('x') as report:
        def emit(message):
            print(message, flush=True)
            print(message, file=report, flush=True)

        emit(f'checkpoint={checkpoint}')
        emit(f'percentage={args.percentage}')
        emit(f'runs={args.runs}')
        emit(f'model=ConvNet')
        emit(f'ssl_method=simclr')
        emit(f'label_percentage={args.label_percentage}')
        emit(f'ssl_epochs={ssl_epochs}')
        emit(f'probe_epochs={probe_epochs}')
        torch.set_num_threads(1)
        train, labels, test, test_labels = bench.load_data(args.data_path, 'cuda')
        labels_cpu = labels.cpu().numpy()
        accuracies = []
        for seed in range(args.runs):
            chosen = torch.tensor(bench.labeled_indices(labels_cpu, args.label_percentage, seed), device='cuda')
            network = bench.get_network('ConvNet', 3, 10, (32, 32), seed=seed + 3000).to('cuda')
            bench.train_ssl(network, synthetic, 'simclr', ssl_epochs, seed, log_every=args.ssl_log_every)
            accuracy = bench.linear_probe(network, train[chosen], labels[chosen], test, test_labels, probe_epochs, seed)
            accuracies.append(accuracy)
            emit(f'seed={seed} accuracy={accuracy:.2f}%')
            del network
            gc.collect()
            torch.cuda.empty_cache()
        mean = statistics.mean(accuracies)
        std = statistics.stdev(accuracies)
        emit('accuracies=' + ','.join(f'{value:.4f}' for value in accuracies))
        emit(f'mean={mean:.4f}%')
        emit(f'sample_std={std:.4f}%')
        emit(f'mean_plus_minus_std={mean:.2f} +/- {std:.2f}%')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest='command', required=True)
    make = subparsers.add_parser('make-jobs')
    make.add_argument('--root', required=True)
    make.add_argument('--project-root', required=True)
    run = subparsers.add_parser('evaluate')
    run.add_argument('--checkpoint', required=True)
    run.add_argument('--percentage', type=int, choices=PERCENTAGES, required=True)
    run.add_argument('--data-path', required=True)
    run.add_argument('--epochs-json', required=True)
    run.add_argument('--output', required=True)
    run.add_argument('--runs', type=int, default=5)
    run.add_argument('--label-percentage', type=int, choices=(1, 5), default=1)
    run.add_argument('--ssl-log-every', type=int, default=200)
    args = parser.parse_args()
    {'make-jobs': make_jobs, 'evaluate': evaluate}[args.command](args)


if __name__ == '__main__':
    main()
