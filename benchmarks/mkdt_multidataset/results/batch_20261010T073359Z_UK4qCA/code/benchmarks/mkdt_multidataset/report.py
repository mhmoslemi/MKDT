"""Fill MKDT rows for CIFAR-100/Tiny main tables and ResNet18 downstream tables."""

import argparse
import csv
import json
from pathlib import Path
import re
import statistics


MODELS = ('ConvNet', 'VGG11', 'ResNet18')
LOSSES = ('simclr', 'barlowtwins')
SIZES = (1, 2, 5)
LABELS = (1, 5)
RUNS = 10
SOURCES = {
    'CIFAR100': {
        'token': 'cifar100',
        'targets': ('TinyImageNet', 'CIFAR10', 'Aircraft', 'CUB2011', 'Dogs', 'Flowers'),
    },
    'TinyImageNet': {
        'token': 'tinyimagenet',
        'targets': ('CIFAR10', 'CIFAR100', 'Aircraft', 'CUB2011', 'Dogs', 'Flowers'),
    },
}


def collect(batch):
    rows = []
    scores = {}
    for source, metadata in SOURCES.items():
        for model in MODELS:
            targets = (source,) + (metadata['targets'] if model == 'ResNet18' else ())
            for size in SIZES:
                for loss in LOSSES:
                    config = f'{source}_{model}_size{size}_{loss}'
                    for target in targets:
                        for labels in LABELS:
                            path = (batch / 'runs' / config / 'evaluations' /
                                    f'{target}_lbl{labels}' / 'summary.json')
                            summary = json.loads(path.read_text()) if path.is_file() else None
                            complete = bool(
                                summary and summary.get('complete')
                                and summary.get('completed_runs') == RUNS
                                and summary.get('expected_runs') == RUNS
                                and summary.get('seeds') == list(range(RUNS)))
                            row = {
                                'source': source, 'target': target, 'model': model,
                                'ssl_method': loss, 'subset_percentage': size,
                                'label_percentage': labels,
                                'completed_runs': summary.get('completed_runs', 0) if summary else 0,
                                'status': 'complete' if complete else 'incomplete',
                                'mean_percent': '', 'std_percent': '', 'summary': str(path),
                            }
                            if complete:
                                values = summary['accuracies_percent']
                                mean, std = statistics.mean(values), statistics.stdev(values)
                                row['mean_percent'], row['std_percent'] = mean, std
                                scores[(source, target, model, size, labels, loss)] = (
                                    f'\\dsacc{{{mean:.2f}}}{{{std:.2f}}}')
                            rows.append(row)
    return scores, rows


def fill_rows(text, cells):
    output = []
    labels = subset = None
    replacements = 0
    for line in text.splitlines():
        match = re.search(r'\\multirow\{17\}\{\*\}\{([15])\\%\}', line)
        if match:
            labels = int(match.group(1))
        match = re.search(r'\\multirow\{5\}\{\*\}\{([125])\\%\}', line)
        if match:
            subset = int(match.group(1))
        if line.strip().startswith('& & MKDT') and labels and subset:
            values = cells(labels, subset)
            if values and all(values):
                line = '& & MKDT & ' + ' & '.join(values) + ' \\\\'
                replacements += 1
        output.append(line)
    return '\n'.join(output) + '\n', replacements


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', required=True)
    parser.add_argument('--templates', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    batch = Path(args.batch)
    templates = Path(args.templates)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    scores, rows = collect(batch)
    with (output / 'results.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    rendered = {}
    counts = {}
    for source, metadata in SOURCES.items():
        token = metadata['token']

        def main_cells(labels, subset):
            return [scores.get((source, source, model, subset, labels, loss), '')
                    for model in MODELS for loss in LOSSES]

        def downstream_cells(labels, subset):
            return [scores.get((source, target, 'ResNet18', subset, labels, loss), '')
                    for target in metadata['targets'] for loss in LOSSES]

        for kind, factory in (('main', main_cells), ('downstream', downstream_cells)):
            name = f'{token}_{kind}.tex'
            content, count = fill_rows((templates / name).read_text(), factory)
            rendered[name] = content
            counts[name] = count
            (output / name).write_text(content)

    complete = sum(row['status'] == 'complete' for row in rows)
    if args.apply and complete == len(rows):
        for name, content in rendered.items():
            destination = templates / name
            temporary = destination.with_suffix('.tmp')
            temporary.write_text(content)
            temporary.replace(destination)
        print(f'Applied all {complete} MKDT result cells to {templates}.')
    elif args.apply:
        print(f'Not applying incomplete results ({complete}/{len(rows)} cells).')
    print(f'Complete result cells: {complete}/{len(rows)}; row replacements: {counts}')


if __name__ == '__main__':
    main()
