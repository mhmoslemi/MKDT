"""CIFAR100 main-only entry point; reuse the unchanged CIFAR10 SSL/resume worker."""
import csv
import fcntl
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('cifar_finish_base', HERE / 'worker.py')
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


def load_task(manifest, task):
    with Path(manifest).open() as stream:
        rows = [row for row in csv.DictReader(stream, delimiter='\t') if row['task'] == task]
    if len(rows) != 1:
        raise ValueError('Unknown or duplicate CIFAR100 seed job')
    row = rows[0]
    row['seed'], row['epochs'] = int(row['seed']), int(row['epochs'])
    percentage = float(row['subset'])
    row['subset'] = int(percentage) if percentage.is_integer() else percentage
    if (row['source'] != 'CIFAR100' or row['kind'] not in ('ours', 'mkdt')
            or row['model'] not in worker.mkdt.MODELS or row['seed'] not in (0, 1)
            or row['subset'] not in (0.2, 1, 2) or row['scope'] != 'main'
            or row['ssl'] not in ('simclr', 'barlowtwins')
            or (row['kind'] == 'mkdt' and row['subset'] == 0.2)):
        raise ValueError('Unsupported CIFAR100 main-table task')
    expected = (3000 if row['subset'] == 2 else 2000) if row['ssl'] == 'simclr' else 10000
    if row['epochs'] != expected:
        raise ValueError('CIFAR100 must use the same requested schedule as CIFAR10')
    row['checkpoint_dir'] = worker.inside_bench(row['checkpoint_dir'])
    row['output_dir'] = (row['checkpoint_dir'] if row['output_dir'] == '-'
                         else worker.inside_bench(row['output_dir']))
    row['parent_dir'] = None if row['parent_dir'] == '-' else worker.inside_bench(row['parent_dir'])
    return row


def write_report(manifest):
    directory = HERE / 'reports'
    directory.mkdir(exist_ok=True)
    with (directory / '.cifar100.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with Path(manifest).open() as stream:
            tasks = list(csv.DictReader(stream, delimiter='\t'))
        rows, seen = [], set()
        for task in tasks:
            folder = task['checkpoint_dir'] if task['output_dir'] == '-' else task['output_dir']
            path = worker.inside_bench(folder).parent / 'summary.json'
            if path in seen or not path.exists():
                continue
            seen.add(path)
            rows.extend({**row, 'method': task['kind'], 'model': task['model']}
                        for row in worker.read(path)['cells']
                        if row['source'] == row['target'] == 'CIFAR100')
        worker.ours.write_tsv(directory / 'cifar100_main_summary.tsv', [
            'source', 'model', 'method', 'subset', 'ssl', 'target', 'labels',
            'completed_probes', 'expected_probes', 'complete', 'mean_percent', 'std_percent'], rows)


worker.load_task = load_task
worker.write_report = write_report

if __name__ == '__main__':
    worker.main()
