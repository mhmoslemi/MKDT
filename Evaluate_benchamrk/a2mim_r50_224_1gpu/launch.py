"""Prepare signed inputs and submit the requested, separate ResNet50 A2MIM cohort."""
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from contracts import BENCH, COMMIT, HERE, ROOT, allocated, atomic_json, digest_json, exclusive_run, file_hash, folder, read
from data import RawDataset
from downstream_data import SPECS

ACCOUNTS = ('aip-boyuwang', 'aip-yiweilu')
SCRIPT = BENCH / 'submit_a2mim_r50_224_1gpu_vulcan.sh'


def origin(source, kind, subset):
    if kind == 'full':
        return BENCH / ('a2mim_teacher/checkpoints/%s/ResNet18/full/pt100/a2mim/seed_000/config.json' % source)
    if kind == 'random':
        return BENCH / ('cifar_small_simclr2k_1ssl5probes/results/%s/ResNet18/random/pt%g/simclr/ssl_seed_000/config.json' % (source, subset))
    if kind == 'ours':
        return BENCH / ('ours_eval_results_simclr2k_barlow10k_2ssl3probes_main/%s/ResNet18BN/pt%g/simclr/ssl_seed_000/config.json' % (source, subset))
    base = 'mkdt_eval_results_simclr2k_2ssl3probes' if subset == 0.2 else source.lower() + '_main_results_simclr2k3k_barlow10k_2ssl3probes'
    return BENCH / base / source / ('ResNet18/mkdt/pt%g/barlowtwins/ssl_seed_000/config.json' % subset)


def prepare(data_root):
    files = sorted(BENCH.glob('*.py')) + [ROOT / 'networks.py', ROOT / 'benchmarks/downstream_data.py', BENCH / 'protocol.json']
    files += sorted(HERE.glob('*.py')) + sorted(HERE.glob('*.sh')) + [HERE / 'requirements.txt', SCRIPT]
    hashes = {str(path.relative_to(ROOT)): file_hash(path) for path in files}
    units = []
    for source in ('CIFAR10', 'CIFAR100'):
        raw = RawDataset(data_root, source, True)
        settings = [('full', 100)] + [(kind, subset) for subset in (0.2, 1) for kind in ('random', 'mkdt', 'ours')]
        for kind, subset in settings:
            path = origin(source, kind, subset)
            old = read(path)
            fingerprint = old['signature']['training_split_fingerprint' if kind == 'full' else 'source_fingerprint']
            if raw.fingerprint != fingerprint:
                raise ValueError('Source data differs from original experiment: ' + str(path))
            spec = dict(source_fingerprint=raw.fingerprint, origin_config=str(path.relative_to(ROOT)), origin_config_sha256=file_hash(path))
            count = int(50000 * subset / 100)
            if kind == 'random':
                indices = old['identity']['selection']['indices']
                assert old['identity']['seed'] == 0 and len(indices) == len(set(indices)) == count
                assert all(isinstance(index, int) and 0 <= index < 50000 for index in indices)
                spec['indices'] = indices
            elif kind in ('mkdt', 'ours'):
                previous = old['signature']['synthetic']
                synthetic = Path(previous['path']).resolve()
                if not synthetic.is_relative_to(ROOT):
                    raise ValueError('Synthetic input must stay inside this repository')
                spec.update(path=str(synthetic.relative_to(ROOT)), file_sha256=file_hash(synthetic))
                if kind == 'mkdt':
                    assert spec['file_sha256'] == previous['file_sha256']
                    assert previous['space'] == 'repository_normalized' and previous['pixel_policy'] == 'clamp_to_unit_rgb'
                    spec['normalization'] = list(SPECS[source][1:])
                else:
                    spec.update(normalization=previous['normalization'], images_sha256=previous['images_sha256'])
            unit = dict(source=source, kind=kind, subset=subset, epochs=1000 if kind == 'full' else 10000,
                        input=spec, code_hashes=hashes)
            unit['signature'] = dict(**unit, upstream_commit=COMMIT, architecture='ResNet50ImageNet',
                input_size=224, physical_batch=64, accumulation=8, local_batchnorm=True,
                fp16=True, drop_last=False, communication_backend='gloo', official_optimizer_and_schedule=True,
                ssl_seed=0, linear_epochs=100, probe_seeds=list(range(5)),
                probe_lr={'CIFAR10': 0.1, 'CIFAR100': 1.0, 'TinyImageNet': 0.1})
            unit['identity'] = dict(signature=digest_json(unit['signature']), seed=0, selection_method=kind)
            destination = HERE / 'units' / ('%s_%s_pt%g.json' % (source, kind, subset))
            if destination.exists() and read(destination) != unit:
                raise ValueError('Do not mix a changed recipe into existing units: ' + str(destination))
            atomic_json(destination, unit)
            units.append((destination, unit, count))
        del raw
    assert len(units) == 14 and sum(unit['kind'] == 'full' for _, unit, _ in units) == 2
    return units


def percentile(values):
    assert values and all(math.isfinite(value) and value > 0 for value in values)
    return sorted(values)[math.ceil(0.95 * len(values)) - 1]


def main():
    allocated()
    data_root = Path(sys.argv[1]).resolve()
    venv = Path(sys.argv[2]).resolve()
    ready, timing = read(HERE / 'environment_ready.json'), read(HERE / 'probe_calibration.json')
    assert ready['complete'] and ready['strict_torchvision_export'] and ready['upstream_commit'] == COMMIT
    assert timing['complete']
    with exclusive_run(HERE / '.submission'):
        units = prepare(data_root)
        smoke = next(path for path, unit, _ in units if unit['source'] == 'CIFAR10' and unit['kind'] == 'random' and unit['subset'] == 0.2)
        subprocess.run(['bash', str(SCRIPT), '--worker', 'check', str(smoke), str(data_root), str(venv), '0'], check=True)
        queue = subprocess.check_output(['squeue', '-h', '-u', os.environ['USER'], '-o', '%200j|%i'], text=True)
        active = {name.strip(): job.strip() for name, job in (line.split('|') for line in queue.splitlines())}
        batch = HERE / 'batches' / time.strftime('batch_%Y%m%dT%H%M%SZ')
        batch.mkdir(parents=True, exist_ok=False)
        jobs = []
        step_seconds = percentile(ready['train_step_seconds'])
        for unit_file, unit, count in units:
            output = folder(unit)
            base = 'a2r50_%s_%s_p%g' % (unit['source'], unit['kind'], unit['subset'])
            complete = False
            marker_path = output / 'ssl_complete.json'
            if marker_path.exists():
                marker = read(marker_path)
                assert marker['complete'] and marker['epochs'] == unit['epochs'] and marker['identity'] == unit['identity']
                assert file_hash(output / 'encoder.pt') == marker['checkpoint_file_sha256']
                complete = True
            if not complete and base + '_ssl' not in active:
                # MMCV's epoch runner sleeps two seconds per epoch, including with zero workers.
                seconds = math.ceil((math.ceil(count / 64) * step_seconds + 2) * unit['epochs'] + 120)
                jobs.append(dict(name=base + '_ssl', stage='ssl', unit=str(unit_file), seconds=seconds,
                                 dependencies=[], signal_seconds=min(65000, math.ceil(math.ceil(count / 64) * step_seconds + 60))))
            targets = [unit['source'], 'CIFAR100' if unit['source'] == 'CIFAR10' else 'CIFAR10', 'TinyImageNet']
            cells = [(target, labels) for target in targets for labels in (1, 5)]
            for bundle in (0, 1):
                name = base + '_p%d' % bundle
                if name in active:
                    continue
                seconds = 120
                missing = 0
                for target, labels in cells[bundle * 4:(bundle + 1) * 4]:
                    observation = timing['observations']['%s|%d' % (target, labels)]
                    for seed in range(5):
                        file = output / 'evaluations' / ('%s_lbl%d_probe%d.json' % (target, labels, seed))
                        if file.exists():
                            result = read(file)
                            assert result['complete'] and result['encoder_unchanged'] and result['expected']['identity'] == unit['identity']
                        else:
                            seconds += 100 * percentile(observation['epoch_seconds']) + observation['test_seconds'] + 5
                            missing += 1
                if missing:
                    jobs.append(dict(name=name, stage='probe', unit=str(unit_file), bundle=bundle,
                        seconds=math.ceil(seconds), dependencies=[] if complete else [base + '_ssl'], missing_probes=missing))
        assert all(0 < job['seconds'] < 7 * 86400 for job in jobs), 'Measured runtime exceeds cluster maximum; split before submitting'
        for index, job in enumerate(jobs):
            job['account'] = ACCOUNTS[index % 2]
        atomic_json(batch / 'plan.json', dict(jobs=jobs, expected_encoders=14, expected_cells=84, expected_probes=420,
            training_microbatch_p95_seconds=step_seconds, upstream_commit=COMMIT, one_gpu_adaptation=True))
        submitted = []
        with (batch / 'submitted.tsv').open('w', buffering=1) as receipt:
            print('job\tname\taccount\tstage\tseconds\tdependency', file=receipt)
            for job in jobs:
                dependencies = [active[name] for name in job['dependencies']]
                seconds = job['seconds']
                walltime = '%d-%02d:%02d:%02d' % (seconds // 86400, seconds // 3600 % 24, seconds // 60 % 60, seconds % 60)
                command = ['sbatch', '--parsable', '--hold', '--job-name=' + job['name'], '--account=' + job['account'],
                    '--nodes=1', '--ntasks=1', '--cpus-per-task=1', '--mem=6G', '--gres=gpu:l40s:1', '--time=' + walltime,
                    '--chdir=' + str(BENCH), '--output=' + str(HERE / 'logs/%x-%j.out'), '--error=' + str(HERE / 'logs/%x-%j.err')]
                if dependencies:
                    command += ['--dependency=afterok:' + ':'.join(dependencies), '--kill-on-invalid-dep=yes']
                if job['stage'] == 'ssl':
                    command += ['--signal=B:USR1@%d' % job['signal_seconds']]
                command += [str(SCRIPT), '--worker', job['stage'], job['unit'], str(data_root), str(venv), str(job.get('bundle', 0))]
                jobid = subprocess.check_output(command, text=True).strip().split(';')[0]
                assert jobid.isdigit()
                active[job['name']] = jobid
                submitted.append(jobid)
                print('\t'.join(map(str, (jobid, job['name'], job['account'], job['stage'], seconds, ','.join(dependencies)))), file=receipt)
                print('SUBMITTED:', jobid, job['name'], job['account'], walltime, flush=True)
        if submitted:
            subprocess.run(['scontrol', 'release', ','.join(submitted)], check=True)
        atomic_json(batch / 'complete.json', dict(submission_complete=True, jobs=submitted,
            ssl_jobs=sum(job['stage'] == 'ssl' for job in jobs), probe_bundles=sum(job['stage'] == 'probe' for job in jobs),
            expected_cells=84, expected_probes=420, receipt=str(batch / 'submitted.tsv')))
        print('RECEIPT:', batch / 'submitted.tsv', flush=True)


if __name__ == '__main__':
    main()
