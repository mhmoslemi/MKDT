"""Summarize complete 15-run target configurations and fill a copy of res.tex."""

import argparse
import csv
import json
from pathlib import Path
import re


DATASETS = ('CIFAR100', 'Aircraft', 'CUB2011', 'Dogs', 'Flowers')
MODELS = {'convnet': 'ConvNet', 'vgg11': 'VGG11', 'resnet18': 'ResNet18'}
METHODS = {'No Pre-Training': 'no_pretrain', 'Random Subset': 'random', 'K-means Subset': 'kmeans', 'Full Data': 'full'}


def collect(root):
    with (root / 'configs.tsv').open() as handle:
        configs = list(csv.DictReader(handle, delimiter='\t'))
    requested = (root / 'targets.txt').read_text().strip().split(',')
    results, rows = {}, []
    for config in configs:
        if config['target'] not in requested:
            continue
        paths = sorted((root / 'evaluations' / config['config_id']).glob('job_*/summary.json'))
        if len(paths) > 1:
            raise ValueError(f'Multiple submissions for {config["config_id"]}; choose one before reporting.')
        summary = json.loads(paths[0].read_text()) if paths else None
        complete = bool(summary and summary['complete'] and summary['completed_runs'] == 15
                        and summary['expected_runs'] == 15 and sorted(summary['seeds']) == list(range(15)))
        key = (config['target'], config['model'], config['ssl_method'], config['method'],
               int(config['subset_percentage']), int(config['label_percentage']))
        if complete:
            results[key] = f'${summary["mean_percent"]:.2f} \\pm {summary["std_percent"]:.2f}$'
        rows.append(dict(config, completed_runs=summary['completed_runs'] if summary else 0,
                         mean_percent=summary['mean_percent'] if complete else '',
                         std_percent=summary['std_percent'] if complete else '',
                         status='complete' if complete else 'incomplete'))
    return results, rows


def fill_latex(template, results):
    model = ssl = subset = labels = None
    lines = []
    for line in template.splitlines():
        match = re.search(r'\\label\{tab:cifar10-(convnet|vgg11|resnet18)-(simclr|barlowtwins)\}', line)
        if match:
            model, ssl = MODELS[match[1]], match[2]
        match = re.search(r'\\textbf\{([125])\\% Pretraining Subset Size\}', line)
        if match:
            subset = int(match[1])
        match = re.search(r'\\multirow\{\d+\}\{\*\}\{([15])\\%\}', line)
        if match:
            labels = int(match[1])
        if line.startswith('& ') and not line.startswith('& &'):
            cells = line.removesuffix('\\\\').split('&')
            if len(cells) != 8:
                raise ValueError('Expected eight columns in res.tex')
            method = METHODS.get(cells[1].strip())
            if method:
                if None in (model, ssl, subset, labels):
                    raise ValueError('Cannot determine model/SSL/subset/labels for table row')
                for index, target in enumerate(DATASETS, start=3):
                    key = (target, model, 'none' if method == 'no_pretrain' else ssl, method,
                           0 if method == 'no_pretrain' else 100 if method == 'full' else subset, labels)
                    # Preserve manually entered cells where no complete run is available.
                    if key in results:
                        cells[index] = f' {results[key]} '
                line = '&'.join(cells) + '\\\\'
        lines.append(line)
    return '\n'.join(lines) + '\n'


def render(root):
    root = Path(root)
    results, rows = collect(root)
    with (root / 'tables.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    template = (root / 'res_template.tex').read_text()
    (root / 'res.tex').write_text(fill_latex(template, results))
    print(f'Completed {len(results)}/{len(rows)} configurations. Wrote {root / "tables.csv"} and {root / "res.tex"}.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', required=True)
    render(parser.parse_args().results)
