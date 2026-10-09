"""Generate downstream jobs and their shared CIFAR-10 pretraining dependencies."""

import csv
import io
import json
import math
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
TARGETS = ('CIFAR100', 'Aircraft', 'CUB2011', 'Dogs', 'Flowers', 'TinyImageNet')
TRAIN_COUNTS = dict(CIFAR100=50000, Aircraft=6667, CUB2011=5994, Dogs=12000, Flowers=1020, TinyImageNet=100000)
EPOCHS = ROOT / 'OT-SSL-DD/training_epochs.json'
SCRIPTS = {}


def walltime(seconds):
    minutes = math.ceil(seconds / 300) * 5 + 60
    return f'{minutes // 1440}-{minutes // 60 % 24:02d}:{minutes % 60:02d}:00'


def header(name, limit, gpu=True):
    return f'''#!/bin/bash
#SBATCH --account=aip-boyuwang
#SBATCH --job-name={name}
#SBATCH --time={limit}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=15G
''' + ('#SBATCH --gres=gpu:l40s:1\n' if gpu else '') + '''#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the MKDT project directory}}"
source "$PROJECT_ROOT/benchmarks/downstream/environment.sh"
'''


def save_script(path, content):
    if HERE / 'jobs' in path.parents:
        SCRIPTS[str(path.relative_to(ROOT))] = content
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    path.chmod(0o755)


def save_manifest(path, rows):
    with path.open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)


def main():
    timing = json.loads((ROOT / 'benchmarks/timing.json').read_text())
    protocol = json.loads(EPOCHS.read_text())
    runs = int(protocol['runs_per_configuration'])
    pretraining = protocol['pretraining']
    barlow_multiplier = int(pretraining['barlowtwins_epoch_multiplier'])
    sources, evaluations = [], []
    for model in ('ConvNet', 'VGG11', 'ResNet18'):
        for ssl in ('simclr', 'barlowtwins'):
            multiplier = barlow_multiplier if ssl == 'barlowtwins' else 1
            specifications = [(method, size, int(pretraining[f'{method}_subset_epochs_by_data_percentage'][str(size)]) * multiplier) for method in ('random', 'kmeans') for size in (1, 2, 5)]
            for method, size, epochs in specifications:
                name = f'{model}_{ssl}_{method}_size{size}'
                seconds = runs * (math.ceil(size * 500 / 256) * epochs * timing['models'][model][f'{ssl}_step_seconds'] + 5)
                limit = walltime(seconds)
                path = HERE / 'jobs/pretrain' / f'{name}.sh'
                content = header(f'ds_pre_{name}', limit)
                content += f'''mkdir -p "$DOWNSTREAM_RUN_ROOT/encoders"
srun --unbuffered python3 benchmarks/benchmark_downstream.py pretrain \\
    --data-path "$DATA_ROOT" --output "$DOWNSTREAM_RUN_ROOT/encoders/{name}" \\
    --selection-dir "$DOWNSTREAM_RUN_ROOT/selections" --runs {runs} \\
    --model {model} --ssl-method {ssl} --method {method} \\
    --subset-percentage {size} --ssl-epochs {epochs}
'''
                save_script(path, content)
                sources.append(dict(source_id=name, model=model, ssl_method=ssl, method=method, subset_percentage=size, ssl_epochs=epochs, runs=runs, estimated_minutes=round(seconds / 60, 2), walltime=limit, job_script=str(path.relative_to(ROOT))))
    for target in TARGETS:
        target_epochs = protocol['evaluation_by_dataset'][target]
        for model in ('ConvNet', 'VGG11', 'ResNet18'):
            applicable = [row for row in sources if row['model'] == model]
            applicable += [dict(source_id='none', model=model, ssl_method='none', method='no_pretrain', subset_percentage=0, ssl_epochs=0)]
            for spec in applicable:
                for labels in (1, 5):
                    epoch_key = 'no_pretraining_linear_probe_epochs_by_label_percentage' if spec['method'] == 'no_pretrain' else 'linear_probe_epochs_by_label_percentage'
                    probe_epochs = int(target_epochs[epoch_key][str(labels)])
                    steps = math.ceil(max(1, TRAIN_COUNTS[target] * labels // 100) / 256) * probe_epochs
                    name = f'{target}_{model}_{spec["ssl_method"]}_{spec["method"]}_size{spec["subset_percentage"]}_lbl{labels}'
                    seconds = runs * (timing['models'][model]['probe_seconds'] * max(1, steps / 400) + 5)
                    limit = walltime(seconds)
                    path = HERE / 'jobs/evaluate' / f'{name}.sh'
                    content = header(f'ds_{name}', limit)
                    content += f'''mkdir -p "$DOWNSTREAM_RUN_ROOT/evaluations/{name}"
srun --unbuffered python3 benchmarks/benchmark_downstream.py evaluate \\
    --output "$DOWNSTREAM_RUN_ROOT/evaluations/{name}/job_$SLURM_JOB_ID" \\
    --target {target} --target-cache "$DOWNSTREAM_RUN_ROOT/targets/{target}.pt" \\
    --encoder-dir "$DOWNSTREAM_RUN_ROOT/encoders/{spec['source_id']}" \\
    --model {model} --method {spec['method']} --ssl-method {spec['ssl_method']} \\
    --subset-percentage {spec['subset_percentage']} --ssl-epochs {spec['ssl_epochs']} \\
    --label-percentage {labels} --label-policy random --probe-epochs {probe_epochs} --runs {runs}
'''
                    save_script(path, content)
                    evaluations.append(dict(config_id=name, target=target, model=model, ssl_method=spec['ssl_method'], method=spec['method'], subset_percentage=spec['subset_percentage'], label_percentage=labels, source_id=spec['source_id'], ssl_epochs=spec['ssl_epochs'], runs=runs, label_policy='random', probe_epochs=probe_epochs, estimated_minutes=round(seconds / 60, 2), walltime=limit, job_script=str(path.relative_to(ROOT))))
        path = HERE / 'jobs/prepare' / f'{target}.sh'
        content = header(f'ds_pack_{target}', '0-01:15:00', gpu=False)
        content += f'''mkdir -p "$DOWNSTREAM_RUN_ROOT/targets"
srun --unbuffered python3 benchmarks/benchmark_downstream.py prepare-target \\
    --device cpu --data-path "$DATA_ROOT" --target {target} \\
    --output "$DOWNSTREAM_RUN_ROOT/targets/{target}.pt"
'''
        save_script(path, content)
    save_manifest(HERE / 'pretraining.tsv', sources)
    save_manifest(HERE / 'configs.tsv', evaluations)
    content = header('ds_selections', '0-01:05:00')
    content += f'''python3 benchmarks/benchmark_downstream.py prepare-selections \\
    --data-path "$DATA_ROOT" --output "$DOWNSTREAM_RUN_ROOT/selections" \\
    --source-batch "$SOURCE_CIFAR10_BATCH" --runs {runs}
'''
    save_script(HERE / 'prepare_selections.sh', content)
    content = header('ds_preflight', '0-00:10:00', gpu=False)
    content += '''python3 benchmarks/benchmark_downstream.py preflight \\
    --device cpu --data-path "$DATA_ROOT" --targets "$TARGET_DATASETS"
'''
    save_script(HERE / 'preflight.sh', content)
    with tarfile.open(HERE / 'jobs.tar', 'w') as archive:
        for name, content in SCRIPTS.items():
            data = content.encode()
            member = tarfile.TarInfo(name)
            member.size, member.mode = len(data), 0o755
            archive.addfile(member, io.BytesIO(data))
    print(f'Generated {len(sources)} shared pretraining jobs and {len(evaluations)} downstream configuration jobs with {runs} runs each.')


if __name__ == '__main__':
    main()
