"""Existing cross-architecture recipe, using the new 1% batch-128 MIPA sets."""
import importlib.util
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
RUN = HERE / 'random128_p1'
sys.path.insert(0, str(HERE))
# The old worker imports eval_jobs/worker.py; do not occupy its module name.
spec = importlib.util.spec_from_file_location('cross_arch_existing_worker', HERE / 'worker.py')
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)


def summarize(task):
    group = base.folder_for(task).parent
    targets = (task['source'], 'CIFAR100' if task['source'] == 'CIFAR10' else 'CIFAR10', 'TinyImageNet')
    rows = []
    for target, labels in base.itertools.product(targets, (1, 5)):
        samples, signature, fingerprints = [], None, None
        for seed, probe in base.itertools.product((0, 1), (0, 1)):
            folder = base.folder_for(task, seed)
            path = folder / 'evaluations' / ('%s_lbl%d_probe%d.json' % (target, labels, probe))
            if not path.is_file():
                continue
            result, config, marker = base.read(path), base.read(folder / 'config.json'), base.read(folder / 'ssl_complete.json')
            expected, score = result['expected'], result['accuracy_percent']
            if (not result.get('complete') or not result.get('encoder_unchanged') or not marker['complete']
                    or expected['identity'] != config['identity'] or marker['identity'] != config['identity']
                    or result['encoder_sha256'] != marker['encoder_sha256']
                    or [expected['source'], expected['target'], expected['label_percentage'], expected['seed']] != [task['source'], target, labels, probe]
                    or expected['evaluation'] != config['signature']['protocol']['evaluation']
                    or expected['evaluation_code_hashes'] != base.common.code_hashes()
                    or config['signature']['ablation_code_hashes'] != task['code_hashes']
                    or not base.math.isfinite(score) or not 0 <= score <= 100):
                raise ValueError('Invalid cross-architecture probe: %s' % path)
            current = (expected['train_fingerprint'], expected['test_fingerprint'])
            if signature is not None and (signature != config['signature'] or fingerprints != current):
                raise ValueError('Different SSL recipe/data across seeds')
            signature, fingerprints = config['signature'], current
            samples.append(dict(ssl_seed=seed, probe_seed=probe, accuracy_percent=score))
        complete = len(samples) == 4
        scores = [s['accuracy_percent'] for s in samples]
        rows.append(dict(target=target, labels=labels, complete=complete, completed_probes=len(samples),
            expected_probes=4, mean_percent=base.statistics.mean(scores) if complete else None,
            std_percent=base.statistics.stdev(scores) if complete else None, samples=samples))
    base.common.atomic_json(group / 'summary.json', dict(source=task['source'], model=task['model'],
        kind='ours', subset=1, std_ddof=1, cells=rows, complete=all(r['complete'] for r in rows),
        synthetic_file=task['synthetic_file'], synthetic_sha256=task['synthetic_sha256'],
        table='CVPR_tables/ablations/cross_arch_combined.tex'))
    # Leave current table values intact until the replacement cells are complete.


def main():
    manifest = Path(sys.argv[1]).resolve()
    assert manifest.is_relative_to(RUN / 'batches')
    task = base.read(manifest)
    assert task['kind'] == 'ours' and task['size'] == 1
    assert task['seed'] in (0, 1) and task['model'] in ('ResNet34', 'ResNet50')
    assert base.os.environ.get('SLURM_JOB_ACCOUNT') == task['account']
    assert task['account'] in ('aip-boyuwang', 'aip-yiweilu')
    assert base.common.file_hash(RUN / 'calibration.json') == task['calibration_sha256']

    def progress(*values, **kwargs):
        if len(values) == 1 and isinstance(values[0], str) and values[0].startswith('SSL '):
            prefix, line = values[0].split(': ', 1)
            stats = base.json.loads(line)
            if stats['epoch'] not in (1, 2000) and stats['epoch'] % 200:
                return
            stats.pop('projector_std', None)
            stats.pop('seconds', None)
            values = (prefix + ': ' + base.json.dumps(stats),)
        print(*values, **kwargs)

    base.train_ssl.print = progress
    base.HERE, base.summarize = RUN, summarize
    base.main()


if __name__ == '__main__':
    main()
