"""Collect images-only MKDT results and fill every MKDT table cell."""

import argparse
import csv
import json
from pathlib import Path
import re
import statistics


SIZES = (1, 2, 5)
LOSSES = ('simclr', 'barlowtwins')
LABELS = (1, 5)
MODELS = ('ConvNet', 'VGG11', 'ResNet18')
DOWNSTREAM = ('CIFAR100', 'Aircraft', 'CUB2011', 'Dogs', 'Flowers', 'TinyImageNet')
RUNS = 10


def collect(batch):
    scores = {}
    rows = []
    for model in MODELS:
        for size in SIZES:
            for loss in LOSSES:
                config = f'{model}_size{size}_{loss}'
                for target in ('CIFAR10',) + DOWNSTREAM:
                    for labels in LABELS:
                        path = batch / 'runs' / config / 'evaluations' / f'{target}_lbl{labels}' / 'summary.json'
                        summary = json.loads(path.read_text()) if path.is_file() else None
                        complete = bool(
                            summary and summary.get('complete') and
                            summary.get('completed_runs') == RUNS and
                            summary.get('expected_runs') == RUNS and
                            summary.get('seeds') == list(range(RUNS))
                        )
                        row = {
                            'source': 'CIFAR10', 'target': target, 'model': model,
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
                            scores[(model, target, size, labels, loss)] = f'\\dsacc{{{mean:.2f}}}{{{std:.2f}}}'
                        rows.append(row)
    return scores, rows


def fill_main(text, scores):
    lines = text.splitlines()
    labels = subset = None
    replacements = 0
    output = []
    for line in lines:
        match = re.search(r'\\multirow\{17\}\{\*\}\{([15])\\%\}', line)
        if match:
            labels = int(match.group(1))
        match = re.search(r'\\multirow\{5\}\{\*\}\{([125])\\%\}', line)
        if match:
            subset = int(match.group(1))
        if line.strip().startswith('& & MKDT') and labels and subset:
            values = [scores.get((model, 'CIFAR10', subset, labels, loss), '')
                      for model in MODELS for loss in LOSSES]
            if all(values):
                line = '& & MKDT & ' + ' & '.join(values) + ' \\\\'
                replacements += 1
        output.append(line)
    return '\n'.join(output) + '\n', replacements


def fill_downstream(text, scores):
    lines = text.splitlines()
    table_models = {'CONVNET': 'ConvNet', 'VGG11': 'VGG11', 'RESNET18': 'ResNet18'}
    model = None
    labels = subset = None
    replacements = 0
    output = []
    for line in lines:
        table = re.search(r'% TABLE \d+: (CONVNET|VGG11|RESNET18)', line)
        if table:
            model = table_models[table.group(1)]
            labels = subset = None
        match = re.search(r'\\multirow\{17\}\{\*\}\{([15])\\%\}', line)
        if match:
            labels = int(match.group(1))
        match = re.search(r'\\multirow\{5\}\{\*\}\{([125])\\%\}', line)
        if match:
            subset = int(match.group(1))
        if line.strip().startswith('& & MKDT') and model and labels and subset:
            values = [scores.get((model, target, subset, labels, loss), '')
                      for target in DOWNSTREAM for loss in LOSSES]
            if all(values):
                line = '& & MKDT & ' + ' & '.join(values) + ' \\\\'
                replacements += 1
        output.append(line)
    return '\n'.join(output) + '\n', replacements


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', required=True)
    parser.add_argument('--templates', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--apply', action='store_true',
                        help='Replace source tables only when all 252 cells are complete')
    args = parser.parse_args()
    batch, templates, output = Path(args.batch), Path(args.templates), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    scores, rows = collect(batch)
    with (output / 'results.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    main, main_count = fill_main((templates / 'cifar10_main.tex').read_text(), scores)
    downstream, downstream_count = fill_downstream(
        (templates / 'cifar10_downstream.tex').read_text(), scores)
    (output / 'cifar10_main.tex').write_text(main)
    (output / 'cifar10_downstream.tex').write_text(downstream)
    complete = sum(row['status'] == 'complete' for row in rows)
    if args.apply:
        if complete != len(rows):
            print(f'Not applying incomplete results to CVPR_tables ({complete}/{len(rows)} cells).')
        else:
            for name, content in (('cifar10_main.tex', main),
                                  ('cifar10_downstream.tex', downstream)):
                destination = templates / name
                temporary = destination.with_suffix('.tmp')
                temporary.write_text(content)
                temporary.replace(destination)
            print(f'Applied complete results to {templates}.')
    print(f'Complete result cells: {complete}/{len(rows)}; updated MKDT rows: '
          f'main={main_count}/6 downstream={downstream_count}/18; output={output}')


if __name__ == '__main__':
    main()
