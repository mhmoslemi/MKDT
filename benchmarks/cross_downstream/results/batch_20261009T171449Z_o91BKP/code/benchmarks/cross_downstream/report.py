"""Render ResNet18 cross-dataset results into two LaTeX tables."""

import argparse
import csv
import json
from pathlib import Path


SOURCE_META = {
    'CIFAR100': {
        'display': 'CIFAR-100', 'token': 'cifar100',
        'targets': ('TinyImageNet', 'CIFAR10', 'Aircraft', 'CUB2011', 'Dogs', 'Flowers'),
    },
    'TinyImageNet': {
        'display': 'Tiny ImageNet', 'token': 'tinyimagenet',
        'targets': ('CIFAR10', 'CIFAR100', 'Aircraft', 'CUB2011', 'Dogs', 'Flowers'),
    },
}
DISPLAY = {
    'CIFAR10': 'CIFAR-10', 'CIFAR100': 'CIFAR-100',
    'TinyImageNet': 'Tiny ImageNet', 'Aircraft': 'Aircraft',
    'CUB2011': 'CUB-2011', 'Dogs': 'Dogs', 'Flowers': 'Flowers',
}


def collect(root):
    root = Path(root)
    with (root / 'configs.tsv').open() as handle:
        configs = list(csv.DictReader(handle, delimiter='\t'))
    scores = {}
    flat = []
    cache = {}
    for config in configs:
        result_id = config['result_id']
        if result_id not in cache:
            paths = sorted((root / 'evaluations' / result_id).glob('job_*/summary.json'))
            if len(paths) > 1:
                raise ValueError(f'Multiple submissions for {result_id}')
            cache[result_id] = json.loads(paths[0].read_text()) if paths else None
        summary = cache[result_id]
        expected = int(config['runs'])
        expected_seeds = list(range(expected))
        complete = bool(
            summary and summary['complete']
            and summary['completed_runs'] == expected
            and summary['expected_runs'] == expected
            and sorted(set(summary['seeds'])) == expected_seeds)
        key = (config['source'], config['target'], config['ssl_method'],
               config['method'], int(config['subset_percentage']),
               int(config['label_percentage']))
        scores[key] = (f'\\dsacc{{{summary["mean_percent"]:.2f}}}'
                       f'{{{summary["std_percent"]:.2f}}}') if complete else ''
        flat.append(dict(config,
                         completed_runs=summary['completed_runs'] if summary else 0,
                         mean_percent=summary['mean_percent'] if complete else '',
                         std_percent=summary['std_percent'] if complete else '',
                         status='complete' if complete else 'incomplete'))
    return scores, flat


def render(source, scores, runs=10):
    meta = SOURCE_META[source]
    targets = meta['targets']

    def value(target, ssl, method, subset, labels):
        return scores.get((source, target, ssl, method, subset, labels), '')

    lines = [
        '\\begin{table*}[t]', '\\centering',
        '\\def\\dsacc#1#2{$#1${\\footnotesize$\\pm #2$}}',
        f'\\caption{{Transfer accuracy (\\%) on six downstream datasets using ResNet18 representations pretrained on {meta["display"]}. Results are mean $\\pm$ sample standard deviation over {runs} independent runs. SimCLR and Barlow Twins (BT) use frozen linear probes; No Pre-Training uses a random encoder. PT denotes pretraining subset size. Empty entries indicate unavailable results.}}',
        f'\\label{{tab:{meta["token"]}-downstream-resnet18}}', '',
        '\\setlength{\\tabcolsep}{2pt}',
        '\\renewcommand{\\arraystretch}{1.1}',
        '\\resizebox{\\textwidth}{!}{%',
        '\\begin{tabular}{@{}D{7mm}D{8mm}l*{12}{c}@{}}',
        '\\toprule',
        '\\multirow{2}{*}{\\shortstack{Labeled\\\\(\\%)}}',
        '& \\multirow{2}{*}{\\shortstack{PT\\\\(\\%)}}',
        '& \\multirow{2}{*}{Method}',
    ]
    lines.extend(f'& \\multicolumn{{2}}{{c}}{{{DISPLAY[target]}}}' for target in targets)
    lines[-1] += ' \\\\'
    lines.extend([
        '\\cmidrule(lr){4-5} \\cmidrule(lr){6-7} \\cmidrule(lr){8-9}',
        '\\cmidrule(lr){10-11} \\cmidrule(lr){12-13} \\cmidrule(l){14-15}',
        '& & & SimCLR & BT & SimCLR & BT & SimCLR & BT & SimCLR & BT & SimCLR & BT & SimCLR & BT \\\\',
        '\\midrule', '',
    ])
    for block, labels in enumerate((1, 5)):
        baseline = [value(target, 'none', 'no_pretrain', 0, labels)
                    for target in targets]
        lines.extend([
            f'\\multirow{{17}}{{*}}{{{labels}\\%}}',
            '& -- & No Pre-Training',
            *[f'& \\multicolumn{{2}}{{c}}{{{cell}}}' for cell in baseline],
        ])
        lines[-1] += ' \\\\'
        lines.extend(['\\cmidrule(lr){2-15}', '', '& 100\\% & Full Data'])
        full = []
        for target in targets:
            full.extend((value(target, 'simclr', 'full', 100, labels),
                         value(target, 'barlowtwins', 'full', 100, labels)))
        lines.append('& ' + ' & '.join(full) + ' \\\\')
        lines.extend(['\\cmidrule(lr){2-15}', ''])
        for subset in (1, 2, 5):
            for row_index, (method, title) in enumerate(
                    (('random', 'Random Subset'), ('kmeans', 'K-means Subset'))):
                prefix = (f'& \\multirow{{5}}{{*}}{{{subset}\\%}} & {title}'
                          if row_index == 0 else f'& & {title}')
                cells = []
                for target in targets:
                    cells.extend((value(target, 'simclr', method, subset, labels),
                                  value(target, 'barlowtwins', method, subset, labels)))
                lines.extend([prefix, '& ' + ' & '.join(cells) + ' \\\\'])
            lines.extend([
                '& & KRR-ST & \\multicolumn{12}{c}{} \\\\',
                '& & MKDT & \\multicolumn{12}{c}{} \\\\',
                '& & Ours & \\multicolumn{12}{c}{} \\\\',
            ])
            if subset != 5:
                lines.extend(['\\cmidrule(lr){2-15}', ''])
        if block == 0:
            lines.extend(['', '\\midrule', ''])
    lines.extend(['', '\\bottomrule', '\\end{tabular}%', '}', '\\end{table*}', ''])
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--empty', action='store_true')
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if args.empty:
        scores, flat, runs = {}, [], 10
    else:
        if not args.results:
            parser.error('--results is required unless --empty is used')
        scores, flat = collect(args.results)
        runs = int(flat[0]['runs'])
        with (output / 'cross_downstream.csv').open('w') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
            writer.writeheader()
            writer.writerows(flat)
    for source, meta in SOURCE_META.items():
        path = output / f'{meta["token"]}_downstream.tex'
        path.write_text(render(source, scores, runs))
        completed = sum(row['source'] == source and row['status'] == 'complete'
                        for row in flat)
        total = sum(row['source'] == source for row in flat)
        print(f'Wrote {path}: {completed}/{total} complete configurations.')


if __name__ == '__main__':
    main()
