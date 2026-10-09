"""Generate CIFAR-100 and Tiny ImageNet main-table Slurm jobs."""

import argparse
import csv
import io
import json
import math
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TIMING = ROOT / 'benchmarks/timing.json'
EPOCHS = ROOT / 'OT-SSL-DD/training_epochs.json'
DATASETS = {
    'CIFAR100': ('cifar100', 'c100', 50000),
    'TinyImageNet': ('tinyimagenet', 'tiny', 100000),
}
WANG = 'aip-boyuwang'
LU = 'aip-yiweilu'


def walltime(seconds):
    minutes = max(10, math.ceil((seconds * 1.15 + 300) / 300) * 5)
    if minutes > 24 * 60:
        raise ValueError(f'Estimated walltime exceeds one day: {minutes} minutes')
    return f'0-{minutes // 60:02d}:{minutes % 60:02d}:00'


def header(name, time_limit, account, gpu=True, memory='4G'):
    return f'''#!/bin/bash
#SBATCH --account={account}
#SBATCH --job-name={name}
#SBATCH --time={time_limit}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem={memory}
''' + ('#SBATCH --gres=gpu:l40s:1\n' if gpu else '') + '''#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the MKDT project directory}}"
source "$PROJECT_ROOT/benchmarks/main/environment.sh"
'''


def generate(dataset):
    slug, short, train_count = DATASETS[dataset]
    destination = ROOT / 'benchmarks' / slug
    destination.mkdir(parents=True, exist_ok=True)
    timing = json.loads(TIMING.read_text())
    protocol = json.loads(EPOCHS.read_text())
    runs = int(protocol['runs_per_configuration'])
    pretraining = protocol['pretraining']
    evaluation = protocol['evaluation_by_dataset'][dataset]
    probe_epochs = evaluation['linear_probe_epochs_by_label_percentage']
    no_pretrain_epochs = evaluation['no_pretraining_linear_probe_epochs_by_label_percentage']
    barlow_multiplier = int(pretraining['barlowtwins_epoch_multiplier'])
    scripts = {}
    rows = []
    job_index = 0
    for model in ('ConvNet', 'VGG11', 'ResNet18'):
        model_timing = timing['models'][model]
        for labels in (1, 5):
            specs = [('no_pretrain', 'none', 0, 0, int(no_pretrain_epochs[str(labels)]))]
            for ssl in ('simclr', 'barlowtwins'):
                multiplier = barlow_multiplier if ssl == 'barlowtwins' else 1
                for method in ('random', 'kmeans'):
                    epoch_map = pretraining[f'{method}_subset_epochs_by_data_percentage']
                    specs.extend((method, ssl, size, int(epoch_map[str(size)]) * multiplier,
                                  int(probe_epochs[str(labels)])) for size in (1, 2, 5))
                full_epochs = int(pretraining['full_data_epochs_by_data_percentage']['100'])
                specs.append(('full', ssl, 100, full_epochs, int(probe_epochs[str(labels)])))
            for method, ssl, size, ssl_epochs, probe in specs:
                account = LU if job_index % 5 == 4 else WANG
                job_index += 1
                name = f'{model}_{ssl}_{method}_size{size}_lbl{labels}'
                labeled_count = max(1, train_count * labels // 100)
                probe_updates = math.ceil(labeled_count / 256) * probe
                probe_seconds = model_timing['probe_seconds'] * max(1, probe_updates / 400)
                if method == 'no_pretrain':
                    runtime = runs * (probe_seconds + 5)
                else:
                    ssl_updates = math.ceil(train_count * size / 100 / 256) * ssl_epochs
                    runtime = runs * (ssl_updates * model_timing[f'{ssl}_step_seconds'] + probe_seconds + 5)
                limit = walltime(runtime)
                relative = f'benchmarks/{slug}/jobs/{name}.sh'
                script = header(f'{short}_{name}', limit, account)
                script += f'''CONFIG_ID={name}
OUTPUT="$MAIN_RUN_ROOT/$CONFIG_ID/job_$SLURM_JOB_ID"
mkdir -p "$MAIN_RUN_ROOT/$CONFIG_ID"
srun --unbuffered python3 benchmarks/benchmark_dataset.py run \\
    --dataset {dataset} --target-cache "$TARGET_CACHE" \\
    --output "$OUTPUT" --selection-dir "$SELECTION_DIR" \\
    --runs {runs} --seed-start 0 --method {method} --model {model} \\
    --ssl-method {ssl} --subset-percentage {size} --label-percentage {labels} \\
    --ssl-epochs {ssl_epochs} --probe-epochs {probe}
'''
                scripts[relative] = script
                rows.append({
                    'config_id': name,
                    'dataset': dataset,
                    'method': method,
                    'model': model,
                    'ssl_method': ssl,
                    'subset_percentage': size,
                    'label_percentage': labels,
                    'ssl_epochs': ssl_epochs,
                    'probe_epochs': probe,
                    'runs': runs,
                    'label_policy': 'random',
                    'estimated_minutes': round(runtime / 60, 2),
                    'walltime': limit,
                    'memory': '4G',
                    'account': account,
                    'job_script': relative,
                })
    with (destination / 'configs.tsv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)
    with tarfile.open(destination / 'jobs.tar', 'w') as archive:
        for name, content in scripts.items():
            data = content.encode()
            member = tarfile.TarInfo(name)
            member.size, member.mode = len(data), 0o755
            archive.addfile(member, io.BytesIO(data))
    prepare_script = header(f'{short}_prepare_kmeans', '0-00:10:00', WANG)
    prepare_script += f'''mkdir -p "$MAIN_RUN_ROOT"
srun --unbuffered python3 benchmarks/benchmark_dataset.py prepare \\
    --dataset {dataset} --target-cache "$TARGET_CACHE" \\
    --output "$SELECTION_DIR" --runs {runs} --seed-start 0
'''
    (destination / 'prepare.sh').write_text(prepare_script)
    (destination / 'prepare.sh').chmod(0o755)
    report_script = header(f'{short}_main_table', '0-00:10:00', WANG, gpu=False, memory='1G')
    report_script += f'''python3 benchmarks/main/report.py \\
    --dataset {dataset} --results "$MAIN_RUN_ROOT" \\
    --output "$MAIN_RUN_ROOT/{slug}_main.tex"
'''
    (destination / 'report.sh').write_text(report_script)
    (destination / 'report.sh').chmod(0o755)
    longest = max(rows, key=lambda row: row['estimated_minutes'])
    protocol_text = (
        f'{dataset} same-dataset main benchmark ({runs} runs).\n'
        'Models: ConvNet, VGG11, ResNet18. SSL: SimCLR, Barlow Twins.\n'
        'Methods: no pretraining, random subset, K-means subset, full data.\n'
        'Label budgets: 1% and 5%, uniform random without replacement.\n'
        'Frozen linear probes; source and evaluation dataset are identical.\n'
        'Epoch budgets come from OT-SSL-DD/training_epochs.json.\n'
        'One L40S, one CPU, and 4 GB host RAM per configuration.\n'
        'Walltimes use measured L40S step timings plus a 15% safety margin.\n'
        f'Accounts: {sum(row["account"] == WANG for row in rows)} Wang, '
        f'{sum(row["account"] == LU for row in rows)} Lu.\n'
        f'Longest estimate: {longest["config_id"]} at {longest["estimated_minutes"]:.1f} minutes '
        f'(requested {longest["walltime"]}).\n'
    )
    (destination / 'protocol.txt').write_text(protocol_text)
    print(f'{dataset}: {len(rows)} jobs; Wang={sum(row["account"] == WANG for row in rows)}, '
          f'Lu={sum(row["account"] == LU for row in rows)}, longest={longest["walltime"]}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=DATASETS)
    args = parser.parse_args()
    for dataset in (args.dataset,) if args.dataset else DATASETS:
        generate(dataset)


if __name__ == '__main__':
    main()
