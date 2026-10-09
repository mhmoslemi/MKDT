"""Generate CIFAR-10 Slurm jobs from the recorded epoch protocol."""

import csv
import io
import json
import math
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
TIMING = HERE.parent / 'timing.json'
EPOCHS = ROOT / 'OT-SSL-DD/training_epochs.json'
SCRIPTS = {}


def save_job(path, content):
    SCRIPTS[str(path.relative_to(ROOT))] = content


def save_jobs():
    with tarfile.open(HERE / 'jobs.tar', 'w') as archive:
        for name, content in SCRIPTS.items():
            data = content.encode()
            member = tarfile.TarInfo(name)
            member.size, member.mode = len(data), 0o755
            archive.addfile(member, io.BytesIO(data))


def walltime(seconds):
    minutes = math.ceil(seconds / 300) * 5 + 60
    return f'{minutes // 1440}-{minutes // 60 % 24:02d}:{minutes % 60:02d}:00'


def script_header(name, time_limit, gpu=True):
    return f'''#!/bin/bash
#SBATCH --account=aip-yiweilu
#SBATCH --job-name={name}
#SBATCH --time={time_limit}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=15G
''' + ('#SBATCH --gres=gpu:l40s:1\n' if gpu else '') + '''#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the MKDT project directory}}"
source "$PROJECT_ROOT/benchmarks/cifar10/environment.sh"
'''


def main():
    measurements = json.loads(TIMING.read_text())
    protocol = json.loads(EPOCHS.read_text())
    runs = int(protocol['runs_per_configuration'])
    pretraining = protocol['pretraining']
    probe_epochs = protocol['evaluation_by_dataset']['CIFAR10']['linear_probe_epochs_by_label_percentage']
    no_pretrain_epochs = protocol['evaluation_by_dataset']['CIFAR10']['no_pretraining_linear_probe_epochs_by_label_percentage']
    barlow_multiplier = int(pretraining['barlowtwins_epoch_multiplier'])
    jobs = HERE / 'jobs'
    rows = []
    for model in ('ConvNet', 'VGG11', 'ResNet18'):
        timing = measurements['models'][model]
        for labels in (1, 5):
            specifications = [('no_pretrain', 'none', 0, 0, int(no_pretrain_epochs[str(labels)]))]
            for ssl in ('simclr', 'barlowtwins'):
                multiplier = barlow_multiplier if ssl == 'barlowtwins' else 1
                for method in ('random', 'kmeans'):
                    epoch_map = pretraining[f'{method}_subset_epochs_by_data_percentage']
                    specifications.extend((method, ssl, size, int(epoch_map[str(size)]) * multiplier, int(probe_epochs[str(labels)])) for size in (1, 2, 5))
            for method, ssl, size, epochs, probe in specifications:
                name = f'{model}_{ssl}_{method}_size{size}_lbl{labels}'
                steps = math.ceil(50000 * size / 100 / 256) * epochs
                supervised_steps = math.ceil(500 * labels / 256) * probe
                probe_seconds = timing['probe_seconds'] * max(1, supervised_steps / 400)
                runtime = runs * (probe_seconds + 5) if method == 'no_pretrain' else runs * (steps * timing[f'{ssl}_step_seconds'] + probe_seconds + 5)
                limit = walltime(runtime)
                path = jobs / f'{name}.sh'
                script = script_header(f'bench_{name}', limit)
                script += f'''CONFIG_ID={name}
OUTPUT="$BENCH_RUN_ROOT/$CONFIG_ID/job_$SLURM_JOB_ID"
mkdir -p "$BENCH_RUN_ROOT/$CONFIG_ID"
srun --unbuffered python3 benchmarks/benchmark_cifar10.py run \\
    --data-path "$DATA_ROOT" --output "$OUTPUT" \\
    --selection-dir "$SELECTION_DIR" --runs {runs} --seed-start 0 \\
    --method {method} --model {model} --ssl-method {ssl} \\
    --subset-percentage {size} --label-percentage {labels} \\
    --ssl-epochs {epochs} --probe-epochs {probe}
'''
                save_job(path, script)
                rows.append(dict(config_id=name, method=method, model=model, ssl_method=ssl, subset_percentage=size, label_percentage=labels, ssl_epochs=epochs, probe_epochs=probe, runs=runs, label_policy='random', estimated_minutes=round(runtime / 60, 2), walltime=limit, job_script=str(path.relative_to(ROOT))))
    with (HERE / 'configs.tsv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)
    prepare_seconds = runs * (measurements['selection_features_seconds'] + sum(measurements['kmeans'].values()) + 5)
    prepare = script_header('bench_prepare_kmeans', walltime(prepare_seconds))
    prepare += f'''mkdir -p "$SELECTION_DIR"
srun --unbuffered python3 benchmarks/benchmark_cifar10.py prepare \\
    --data-path "$DATA_ROOT" --output "$SELECTION_DIR" --runs {runs} --seed-start 0
'''
    (HERE / 'prepare.sh').write_text(prepare)
    (HERE / 'prepare.sh').chmod(0o755)
    save_jobs()
    print(f'Created {len(rows)} configuration jobs with {runs} runs each; preparation estimate={prepare_seconds / 60:.1f} minutes.')


if __name__ == '__main__':
    main()
