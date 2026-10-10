"""Generate a fresh CIFAR10/CIFAR100 synthetic benchmark batch."""

import argparse
import csv
import json
import math
from pathlib import Path


WANG = 'aip-boyuwang'
LU = 'aip-yiweilu'
RUNS = 10
SIZES = (1, 2, 5)
LOSSES = ('simclr', 'barlowtwins')
MODELS = ('ConvNet', 'VGG11', 'ResNet18')
SOURCES = {
    'CIFAR10': {
        'slug': 'cifar10',
        'targets': ('CIFAR100', 'Aircraft', 'CUB2011', 'Dogs', 'Flowers', 'TinyImageNet'),
    },
    'CIFAR100': {
        'slug': 'cifar100',
        'targets': ('TinyImageNet', 'CIFAR10', 'Aircraft', 'CUB2011', 'Dogs', 'Flowers'),
    },
}


def walltime(seconds):
    minutes = max(10, math.ceil((seconds * 1.15 + 300) / 300) * 5)
    if minutes > 24 * 60:
        raise ValueError(f'Estimated walltime exceeds one day: {minutes} minutes')
    return f'0-{minutes // 60:02d}:{minutes % 60:02d}:00'


def header(name, time_limit, account, gpu=True, memory='4G'):
    lines = [
        '#!/bin/bash',
        f'#SBATCH --account={account}',
        f'#SBATCH --job-name={name}',
        f'#SBATCH --time={time_limit}',
        '#SBATCH --nodes=1',
        '#SBATCH --ntasks=1',
        '#SBATCH --cpus-per-task=1',
        f'#SBATCH --mem={memory}',
    ]
    if gpu:
        lines.append('#SBATCH --gres=gpu:l40s:1')
    lines.extend([
        '#SBATCH --output=slurm-%x-%j.out',
        '',
        'set -euo pipefail',
        'source "$PROJECT_ROOT/benchmarks/synthetic_new/environment.sh"',
    ])
    return '\n'.join(lines) + '\n'


def write_tsv(path, rows):
    with path.open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)


def cache_expression(target):
    if target == 'CIFAR10':
        return '"$CIFAR10_CACHE"'
    return f'"$TARGET_CACHE_ROOT/{target}.pt"'


def evaluate_command(source, target, model, ssl, size, ssl_epochs,
                     labels, probe_epochs, config_id):
    slug = SOURCES[source]['slug']
    encoder = f'{slug}_{model}_{ssl}_ours_size{size}'
    return f'''srun --unbuffered python3 benchmarks/benchmark_synthetic_new.py evaluate \\
    --synthetic-root "$SYNTHETIC_INPUT_ROOT" --source {source} --target {target} \\
    --target-cache {cache_expression(target)} \\
    --encoder-dir "$SYNTH_BATCH_ROOT/encoders/{encoder}" \\
    --output "$SYNTH_BATCH_ROOT/evaluations/{config_id}/job_$SLURM_JOB_ID" \\
    --model {model} --ssl-method {ssl} --subset-percentage {size} \\
    --ssl-epochs {ssl_epochs} --label-percentage {labels} \\
    --probe-epochs {probe_epochs}
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', required=True)
    parser.add_argument('--project-root', required=True)
    args = parser.parse_args()

    batch = Path(args.batch)
    root = Path(args.project_root)
    jobs_root = batch / 'jobs'
    for stage in ('pretrain', 'main', 'downstream'):
        (jobs_root / stage).mkdir(parents=True, exist_ok=False)

    timing = json.loads((root / 'benchmarks/timing.json').read_text())['models']
    protocol = json.loads((root / 'OT-SSL-DD/training_epochs.json').read_text())
    if int(protocol['runs_per_configuration']) != RUNS:
        raise ValueError('Expected the current 10-run protocol')
    pretraining = protocol['pretraining']
    random_epochs = pretraining['random_subset_epochs_by_data_percentage']
    multiplier = int(pretraining['barlowtwins_epoch_multiplier'])

    jobs = []
    configs = []
    gpu_index = 0

    def next_account():
        nonlocal gpu_index
        account = LU if gpu_index % 5 == 0 else WANG
        gpu_index += 1
        return account

    def add_job(stage, key, dependency, content, account, time_limit):
        relative = f'jobs/{stage}/{key}.sh'
        path = batch / relative
        path.write_text(content)
        path.chmod(0o755)
        jobs.append({
            'stage': stage,
            'key': key,
            'dependency': dependency,
            'account': account,
            'walltime': time_limit,
            'script': relative,
        })

    for source, source_meta in SOURCES.items():
        slug = source_meta['slug']
        for model in MODELS:
            for ssl in LOSSES:
                ssl_multiplier = multiplier if ssl == 'barlowtwins' else 1
                for size in SIZES:
                    ssl_epochs = int(random_epochs[str(size)]) * ssl_multiplier
                    steps_per_epoch = math.ceil(500 * size / 256)
                    seconds = RUNS * (
                        steps_per_epoch * ssl_epochs
                        * timing[model][f'{ssl}_step_seconds'] + 5)
                    limit = walltime(seconds)
                    encoder = f'{slug}_{model}_{ssl}_ours_size{size}'
                    account = next_account()
                    pretrain = header(
                        f'synpre_{slug}_{model}_{ssl}{size}', limit, account)
                    pretrain += f'''srun --unbuffered python3 benchmarks/benchmark_synthetic_new.py pretrain \\
    --synthetic-root "$SYNTHETIC_INPUT_ROOT" --source {source} \\
    --output "$SYNTH_BATCH_ROOT/encoders/{encoder}" --model {model} \\
    --ssl-method {ssl} --subset-percentage {size} --ssl-epochs {ssl_epochs}
'''
                    add_job('pretrain', encoder, '-', pretrain, account, limit)

                    main_key = f'{encoder}__main'
                    account = next_account()
                    main_script = header(
                        f'synmain_{slug}_{model}_{ssl}{size}', '0-00:20:00', account)
                    for labels in (1, 5):
                        probe_epochs = int(protocol['evaluation_by_dataset'][source]
                                           ['linear_probe_epochs_by_label_percentage'][str(labels)])
                        config_id = f'{encoder}__{source}_lbl{labels}'
                        main_script += evaluate_command(
                            source, source, model, ssl, size, ssl_epochs,
                            labels, probe_epochs, config_id)
                        configs.append({
                            'config_id': config_id, 'table': 'main',
                            'source': source, 'target': source, 'model': model,
                            'method': 'ours', 'ssl_method': ssl,
                            'subset_percentage': size, 'label_percentage': labels,
                            'ssl_epochs': ssl_epochs, 'probe_epochs': probe_epochs,
                            'runs': RUNS, 'account': account, 'job_key': main_key,
                        })
                    add_job('main', main_key, encoder, main_script,
                            account, '0-00:20:00')

                    if model != 'ResNet18':
                        continue
                    downstream_key = f'{encoder}__downstream'
                    account = next_account()
                    downstream = header(
                        f'syndown_{slug}_{ssl}{size}', '0-00:20:00', account)
                    for target in source_meta['targets']:
                        for labels in (1, 5):
                            probe_epochs = int(protocol['evaluation_by_dataset'][target]
                                               ['linear_probe_epochs_by_label_percentage'][str(labels)])
                            config_id = f'{encoder}__{target}_lbl{labels}'
                            downstream += evaluate_command(
                                source, target, model, ssl, size, ssl_epochs,
                                labels, probe_epochs, config_id)
                            configs.append({
                                'config_id': config_id, 'table': 'downstream',
                                'source': source, 'target': target,
                                'model': model, 'method': 'ours',
                                'ssl_method': ssl, 'subset_percentage': size,
                                'label_percentage': labels,
                                'ssl_epochs': ssl_epochs,
                                'probe_epochs': probe_epochs, 'runs': RUNS,
                                'account': account, 'job_key': downstream_key,
                            })
                    add_job('downstream', downstream_key, encoder, downstream,
                            account, '0-00:20:00')

    if len(jobs) != 84 or len(configs) != 216:
        raise AssertionError((len(jobs), len(configs)))
    lu_gpu = sum(row['account'] == LU for row in jobs)
    wang_gpu = sum(row['account'] == WANG for row in jobs)
    if (lu_gpu, wang_gpu) != (17, 67):
        raise AssertionError((lu_gpu, wang_gpu))

    report = header('synthetic_new_tables', '0-00:10:00', WANG,
                    gpu=False, memory='1G')
    report += '''python3 benchmarks/benchmark_synthetic_new.py report \\
    --batch "$SYNTH_BATCH_ROOT" \\
    --templates "$SYNTH_BATCH_ROOT/table_templates" \\
    --output "$SYNTH_BATCH_ROOT/tables"
'''
    report_path = jobs_root / 'report.sh'
    report_path.write_text(report)
    report_path.chmod(0o755)

    write_tsv(batch / 'jobs.tsv', jobs)
    write_tsv(batch / 'configs.tsv', configs)
    (batch / 'protocol.json').write_text(json.dumps({
        'protocol': 'synthetic_new_v1_random_labels_barlow2x',
        'sources': list(SOURCES),
        'synthetic_sizes_percent': list(SIZES),
        'main_models': list(MODELS),
        'downstream_models': ['ResNet18'],
        'ssl_methods': list(LOSSES),
        'runs_per_configuration': RUNS,
        'simclr_epochs': {size: int(random_epochs[str(size)]) for size in SIZES},
        'barlowtwins_epochs': {
            size: int(random_epochs[str(size)]) * multiplier for size in SIZES},
        'probe_epochs': protocol['evaluation_by_dataset'],
        'gpu_jobs': len(jobs),
        'gpu_accounts': {WANG: wang_gpu, LU: lu_gpu},
        'report_account': WANG,
        'total_jobs_including_report': len(jobs) + 1,
        'account_split_including_report': {WANG: wang_gpu + 1, LU: lu_gpu},
    }, indent=2) + '\n')
    print(f'Generated {len(jobs)} GPU jobs and one report job; '
          f'GPU accounts: Wang={wang_gpu}, Lu={lu_gpu}; '
          f'all benchmark jobs: Wang={wang_gpu + 1}, Lu={lu_gpu}.')


if __name__ == '__main__':
    main()
