"""Summarize complete target configurations and fill a copy of res.tex."""

import argparse
import csv
import json
from pathlib import Path
import re


DATASETS = ('CIFAR100', 'Aircraft', 'CUB2011', 'Dogs', 'Flowers', 'TinyImageNet')
MODELS = ('ConvNet', 'VGG11', 'ResNet18')
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
        if config['method'] == 'no_pretrain':
            paths = [path for path in paths if json.loads(path.read_text())['config'].get('no_pretrain_protocol') == 'frozen_random_encoder']
        if len(paths) > 1:
            raise ValueError(f'Multiple submissions for {config["config_id"]}; choose one before reporting.')
        summary = json.loads(paths[0].read_text()) if paths else None
        expected = int(config['runs'])
        complete = bool(summary and summary['complete'] and summary['completed_runs'] == expected and summary['expected_runs'] == expected and sorted(summary['seeds']) == list(range(expected)))
        key = (config['target'], config['model'], config['ssl_method'], config['method'], int(config['subset_percentage']), int(config['label_percentage']))
        if complete:
            results[key] = f'${summary["mean_percent"]:.2f} \\pm {summary["std_percent"]:.2f}$'
        rows.append(dict(config, completed_runs=summary['completed_runs'] if summary else 0, mean_percent=summary['mean_percent'] if complete else '', std_percent=summary['std_percent'] if complete else '', status='complete' if complete else 'incomplete'))
    return results, rows


def fill_latex(template, results, targets=DATASETS, runs=10):
    source = template.split('% Downstream results')[0].rstrip().replace('15 independent runs', f'{runs} independent runs')
    lines = source.splitlines()
    tables = [source, '% Downstream results']
    for target in targets:
        output, index, labels, subset = [], 0, None, None
        while index < len(lines):
            line = lines[index]
            if line.startswith('% Results batch:'):
                index += 1
                continue
            match = re.search(r'\\multirow\{17\}\{\*\}\{([15])\\%\}', line)
            if match:
                labels = int(match[1])
            match = re.search(r'\\multirow\{5\}\{\*\}\{([125])\\%\}', line)
            if match:
                subset = int(match[1])
            method = next((key for name, key in METHODS.items() if line.endswith(name)), None)
            line = line.replace('Consolidated CIFAR-10', f'Consolidated {target}').replace('tab:cifar10-all', f'tab:downstream-{target.lower()}').replace('random encoder.}', 'random encoder. Pretraining uses CIFAR-10.}')
            output.append(line)
            if method:
                size = 0 if method == 'no_pretrain' else 100 if method == 'full' else subset
                for model in MODELS:
                    cells = [results.get((target, model, ssl, method, size, labels), '') for ssl in (('none',) if method == 'no_pretrain' else ('simclr', 'barlowtwins'))]
                    content = '\\multicolumn{2}{c}{' + cells[0] + '}' if method == 'no_pretrain' else ' & '.join(cells)
                    output.append('& ' + content + (' \\\\' if model == 'ResNet18' else ''))
                index += 1
                while not lines[index].rstrip().endswith('\\\\'):
                    index += 1
            index += 1
        tables.append('\n'.join(output))
    return '\n\n'.join(tables) + '\n'


def render(root):
    root = Path(root)
    results, rows = collect(root)
    with (root / 'tables.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    template = (root / 'res_template.tex').read_text()
    targets = (root / 'targets.txt').read_text().strip().split(',')
    runs = int(rows[0]['runs'])
    (root / 'res.tex').write_text(fill_latex(template, results, targets, runs))
    print(f'Completed {len(results)}/{len(rows)} configurations. Wrote {root / "tables.csv"} and {root / "res.tex"}.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', required=True)
    render(parser.parse_args().results)
