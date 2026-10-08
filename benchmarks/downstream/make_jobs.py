"""Generate all target jobs and their shared CIFAR-10 pretraining dependencies."""

import csv
import json
import math
import io
import tarfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
TARGETS = ('CIFAR100', 'Aircraft', 'CUB2011', 'Dogs', 'Flowers')
TRAIN_COUNTS = dict(CIFAR100=50000, Aircraft=6667, CUB2011=5994, Dogs=12000, Flowers=1020)
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
    sources, evaluations = [], []
    for model in ('ConvNet', 'VGG11', 'ResNet18'):
        for ssl in ('simclr', 'barlowtwins'):
            for method, size, epochs in [('full', 100, 300)] + [
                (method, size, epochs) for method in ('random', 'kmeans')
                for size, epochs in ((1, 1200), (2, 800), (5, 500))
            ]:
                name = f'{model}_{ssl}_{method}_size{size}'
                seconds = 15 * (math.ceil(size * 500 / 256) * epochs * timing['models'][model][f'{ssl}_step_seconds'] + 5)
                limit = walltime(seconds)
                path = HERE / 'jobs/pretrain' / f'{name}.sh'
                content = header(f'ds_pre_{name}', limit)
                content += f'''mkdir -p "$DOWNSTREAM_RUN_ROOT/encoders"
srun --unbuffered python3 benchmarks/benchmark_downstream.py pretrain \\
    --data-path "$DATA_ROOT" --output "$DOWNSTREAM_RUN_ROOT/encoders/{name}" \\
    --selection-dir "$DOWNSTREAM_RUN_ROOT/selections" --runs 15 \\
    --model {model} --ssl-method {ssl} --method {method} \\
    --subset-percentage {size} --ssl-epochs {epochs}
'''
                save_script(path, content)
                sources.append(dict(source_id=name, model=model, ssl_method=ssl, method=method,
                                    subset_percentage=size, ssl_epochs=epochs, runs=15,
                                    estimated_minutes=round(seconds / 60, 2), walltime=limit,
                                    job_script=str(path.relative_to(ROOT))))
    for target in TARGETS:
        for model in ('ConvNet', 'VGG11', 'ResNet18'):
            applicable = [row for row in sources if row['model'] == model]
            applicable += [dict(source_id='none', model=model, ssl_method='none', method='no_pretrain',
                                subset_percentage=0, ssl_epochs=0)]
            for spec in applicable:
                for labels in (1, 5):
                    probe_epochs = {1: 200, 5: 100}[labels]
                    steps = math.ceil(math.ceil(TRAIN_COUNTS[target] * labels / 100) / 256) * probe_epochs
                    name = f'{target}_{model}_{spec["ssl_method"]}_{spec["method"]}_size{spec["subset_percentage"]}_lbl{labels}'
                    seconds = 15 * (timing['models'][model]['probe_seconds'] * max(1, steps / 400) + 5)
                    if spec['method'] == 'no_pretrain':
                        seconds += 15 * steps * timing['models'][model]['supervised_step_seconds']
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
    --label-percentage {labels} --label-policy exact --probe-epochs {probe_epochs} --runs 15
'''
                    save_script(path, content)
                    evaluations.append(dict(config_id=name, target=target, model=model,
                        ssl_method=spec['ssl_method'], method=spec['method'],
                        subset_percentage=spec['subset_percentage'], label_percentage=labels,
                        source_id=spec['source_id'], ssl_epochs=spec['ssl_epochs'], runs=15,
                        label_policy='exact', probe_epochs=probe_epochs, estimated_minutes=round(seconds / 60, 2),
                        walltime=limit, job_script=str(path.relative_to(ROOT))))
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
    content += '''python3 benchmarks/benchmark_downstream.py prepare-selections \\
    --data-path "$DATA_ROOT" --output "$DOWNSTREAM_RUN_ROOT/selections" \\
    --source-batch "$SOURCE_CIFAR10_BATCH" --runs 15
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
    print(f'Generated {len(sources)} shared pretraining jobs and {len(evaluations)} downstream configuration jobs.')


if __name__ == '__main__':
    main()
