"""Generate ResNet18 CIFAR-100/Tiny-ImageNet cross-dataset Slurm jobs."""

import csv
import io
import json
import math
import tarfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
WANG = 'aip-boyuwang'
LU = 'aip-yiweilu'
SOURCES = {
    'CIFAR100': {
        'slug': 'cifar100', 'short': 'c100', 'train_count': 50000,
        'targets': ('TinyImageNet', 'CIFAR10', 'Aircraft', 'CUB2011', 'Dogs', 'Flowers'),
    },
    'TinyImageNet': {
        'slug': 'tinyimagenet', 'short': 'tiny', 'train_count': 100000,
        'targets': ('CIFAR10', 'CIFAR100', 'Aircraft', 'CUB2011', 'Dogs', 'Flowers'),
    },
}
ALL_TARGETS = ('CIFAR10', 'CIFAR100', 'TinyImageNet', 'Aircraft', 'CUB2011', 'Dogs', 'Flowers')


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
source "$PROJECT_ROOT/benchmarks/cross_downstream/environment.sh"
'''


def save_manifest(path, rows):
    with path.open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)


def evaluate_command(source, target, spec, labels, probe_epochs, result_id):
    encoder = (f' --encoder-dir "$CROSS_RUN_ROOT/encoders/{spec["source_id"]}"'
               if spec['method'] != 'no_pretrain' else '')
    return f'''mkdir -p "$CROSS_RUN_ROOT/evaluations/{result_id}"
srun --unbuffered python3 benchmarks/benchmark_cross_dataset.py evaluate \\
    --output "$CROSS_RUN_ROOT/evaluations/{result_id}/job_$SLURM_JOB_ID" \\
    --source {source} --target {target} \\
    --target-cache "$CROSS_RUN_ROOT/targets/{target}.pt"{encoder} \\
    --method {spec['method']} --ssl-method {spec['ssl_method']} \\
    --subset-percentage {spec['subset_percentage']} --ssl-epochs {spec['ssl_epochs']} \\
    --label-percentage {labels} --probe-epochs {probe_epochs} --runs {spec['runs']}
'''


def main():
    timing = json.loads((ROOT / 'benchmarks/timing.json').read_text())
    protocol = json.loads((ROOT / 'OT-SSL-DD/training_epochs.json').read_text())
    runs = int(protocol['runs_per_configuration'])
    pretraining = protocol['pretraining']
    barlow_multiplier = int(pretraining['barlowtwins_epoch_multiplier'])
    model_timing = timing['models']['ResNet18']
    scripts = {}
    sources = []
    configs = []
    job_index = 0

    def next_account():
        nonlocal job_index
        account = LU if job_index % 10 == 0 else WANG
        job_index += 1
        return account

    for source, source_meta in SOURCES.items():
        specs = []
        for ssl in ('simclr', 'barlowtwins'):
            multiplier = barlow_multiplier if ssl == 'barlowtwins' else 1
            for method in ('random', 'kmeans'):
                epoch_map = pretraining[f'{method}_subset_epochs_by_data_percentage']
                specs.extend((method, ssl, size, int(epoch_map[str(size)]) * multiplier)
                             for size in (1, 2, 5))
            specs.append(('full', ssl, 100,
                          int(pretraining['full_data_epochs_by_data_percentage']['100'])))
        for method, ssl, size, ssl_epochs in specs:
            source_id = f'{source_meta["slug"]}_ResNet18_{ssl}_{method}_size{size}'
            updates = math.ceil(source_meta['train_count'] * size / 100 / 256) * ssl_epochs
            seconds = runs * (updates * model_timing[f'{ssl}_step_seconds'] + 5)
            limit = walltime(seconds)
            pretrain_account = next_account()
            pretrain_script = f'benchmarks/cross_downstream/jobs/pretrain/{source_id}.sh'
            content = header(f'xpre_{source_meta["short"]}_{ssl}_{method}{size}',
                             limit, pretrain_account)
            content += f'''mkdir -p "$CROSS_RUN_ROOT/encoders"
srun --unbuffered python3 benchmarks/benchmark_cross_dataset.py pretrain \\
    --source {source} --source-cache "${{{source.upper()}_SOURCE_CACHE}}" \\
    --selection-dir "${{{source.upper()}_SELECTION_DIR}}" \\
    --output "$CROSS_RUN_ROOT/encoders/{source_id}" --runs {runs} \\
    --method {method} --ssl-method {ssl} --subset-percentage {size} \\
    --ssl-epochs {ssl_epochs}
'''
            scripts[pretrain_script] = content

            evaluate_account = next_account()
            evaluate_script = f'benchmarks/cross_downstream/jobs/evaluate/{source_id}.sh'
            eval_content = header(f'xev_{source_meta["short"]}_{ssl}_{method}{size}',
                                  '0-00:20:00', evaluate_account)
            spec = {
                'source_id': source_id, 'method': method, 'ssl_method': ssl,
                'subset_percentage': size, 'ssl_epochs': ssl_epochs, 'runs': runs,
            }
            for target in source_meta['targets']:
                target_epochs = protocol['evaluation_by_dataset'][target]
                for labels in (1, 5):
                    probe_epochs = int(target_epochs['linear_probe_epochs_by_label_percentage'][str(labels)])
                    result_id = f'{source_id}__{target}_lbl{labels}'
                    eval_content += evaluate_command(source, target, spec, labels,
                                                     probe_epochs, result_id)
                    configs.append({
                        'config_id': result_id, 'result_id': result_id,
                        'source': source, 'target': target, 'model': 'ResNet18',
                        'ssl_method': ssl, 'method': method,
                        'subset_percentage': size, 'label_percentage': labels,
                        'source_id': source_id, 'ssl_epochs': ssl_epochs,
                        'runs': runs, 'label_policy': 'random',
                        'probe_epochs': probe_epochs, 'account': evaluate_account,
                        'job_script': evaluate_script,
                    })
            scripts[evaluate_script] = eval_content
            sources.append({
                'source_id': source_id, 'source': source, 'model': 'ResNet18',
                'ssl_method': ssl, 'method': method, 'subset_percentage': size,
                'ssl_epochs': ssl_epochs, 'runs': runs,
                'estimated_minutes': round(seconds / 60, 2), 'walltime': limit,
                'pretrain_account': pretrain_account,
                'evaluate_account': evaluate_account,
                'pretrain_script': pretrain_script,
                'evaluate_script': evaluate_script,
            })

    baseline_account = next_account()
    baseline_script = 'benchmarks/cross_downstream/jobs/evaluate/baseline.sh'
    baseline_content = header('xev_baseline', '0-00:20:00', baseline_account)
    baseline_spec = {
        'source_id': 'none', 'method': 'no_pretrain', 'ssl_method': 'none',
        'subset_percentage': 0, 'ssl_epochs': 0, 'runs': runs,
    }
    for target in ALL_TARGETS:
        target_epochs = protocol['evaluation_by_dataset'][target]
        for labels in (1, 5):
            probe_epochs = int(target_epochs['no_pretraining_linear_probe_epochs_by_label_percentage'][str(labels)])
            result_id = f'baseline_ResNet18__{target}_lbl{labels}'
            baseline_content += evaluate_command('CIFAR100', target, baseline_spec,
                                                 labels, probe_epochs, result_id)
    scripts[baseline_script] = baseline_content
    for source, source_meta in SOURCES.items():
        for target in source_meta['targets']:
            for labels in (1, 5):
                probe_epochs = int(protocol['evaluation_by_dataset'][target]
                                   ['no_pretraining_linear_probe_epochs_by_label_percentage'][str(labels)])
                configs.append({
                    'config_id': f'{source_meta["slug"]}_baseline__{target}_lbl{labels}',
                    'result_id': f'baseline_ResNet18__{target}_lbl{labels}',
                    'source': source, 'target': target, 'model': 'ResNet18',
                    'ssl_method': 'none', 'method': 'no_pretrain',
                    'subset_percentage': 0, 'label_percentage': labels,
                    'source_id': 'none', 'ssl_epochs': 0, 'runs': runs,
                    'label_policy': 'random', 'probe_epochs': probe_epochs,
                    'account': baseline_account, 'job_script': baseline_script,
                })

    save_manifest(HERE / 'sources.tsv', sources)
    save_manifest(HERE / 'configs.tsv', configs)
    with tarfile.open(HERE / 'jobs.tar', 'w') as archive:
        for name, content in scripts.items():
            data = content.encode()
            member = tarfile.TarInfo(name)
            member.size, member.mode = len(data), 0o755
            archive.addfile(member, io.BytesIO(data))

    prepare = header('xpack_cifar10', '0-00:10:00', WANG, gpu=False)
    prepare += '''mkdir -p "$CROSS_RUN_ROOT/targets"
srun --unbuffered python3 benchmarks/benchmark_cross_dataset.py prepare-target \\
    --device cpu --data-path "$DATA_ROOT" --target CIFAR10 \\
    --output "$CROSS_RUN_ROOT/targets/CIFAR10.pt"
'''
    (HERE / 'prepare_cifar10.sh').write_text(prepare)
    (HERE / 'prepare_cifar10.sh').chmod(0o755)

    report = header('cross_tables', '0-00:10:00', WANG, gpu=False, memory='1G')
    report += '''python3 benchmarks/cross_downstream/report.py \\
    --results "$CROSS_RUN_ROOT" --output-dir "$CROSS_RUN_ROOT"
'''
    (HERE / 'report.sh').write_text(report)
    (HERE / 'report.sh').chmod(0o755)

    wang = sum(account == WANG for row in sources
               for account in (row['pretrain_account'], row['evaluate_account']))
    lu = sum(account == LU for row in sources
             for account in (row['pretrain_account'], row['evaluate_account']))
    wang += baseline_account == WANG
    lu += baseline_account == LU
    longest = max(sources, key=lambda row: row['estimated_minutes'])
    (HERE / 'protocol.txt').write_text(
        f'ResNet18 cross-dataset transfer benchmark ({runs} runs).\n'
        'Sources: CIFAR-100 and Tiny ImageNet.\n'
        'Each source is evaluated on the other source, CIFAR-10, Aircraft, CUB-2011, Dogs, and Flowers.\n'
        'SSL: SimCLR and Barlow Twins. Methods: random subset, K-means subset, and full data.\n'
        'Label budgets: 1% and 5%, uniform random without replacement. Frozen linear probes.\n'
        'Epoch budgets come from OT-SSL-DD/training_epochs.json.\n'
        'One L40S, one CPU, and 4 GB host RAM per GPU job. Evaluation targets are bundled by encoder.\n'
        f'GPU jobs: {wang} Wang, {lu} Lu ({100 * lu / (wang + lu):.2f}% Lu).\n'
        f'Longest estimate: {longest["source_id"]} at {longest["estimated_minutes"]:.1f} minutes '
        f'(requested {longest["walltime"]}).\n')
    print(f'Generated {len(sources)} source jobs, {len(sources)} bundled evaluation jobs, '
          f'and one baseline job; Wang={wang}, Lu={lu}.')


if __name__ == '__main__':
    main()
