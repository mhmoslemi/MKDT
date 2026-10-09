"""Render a consolidated LaTeX table from complete main-dataset benchmark runs."""

import argparse
import csv
import json
from pathlib import Path


DISPLAY = {'CIFAR100': 'CIFAR-100', 'TinyImageNet': 'Tiny ImageNet'}
TOKEN = {'CIFAR100': 'cifar100', 'TinyImageNet': 'tinyimagenet'}


def load_results(results):
    results = Path(results)
    with (results / 'configs.tsv').open() as handle:
        configs = list(csv.DictReader(handle, delimiter='\t'))
    scores = {}
    flat = []
    for config in configs:
        candidates = sorted((results / config['config_id']).glob('job_*/summary.json'))
        if len(candidates) > 1:
            raise ValueError(f'Multiple submissions for {config["config_id"]}')
        summary = json.loads(candidates[0].read_text()) if candidates else None
        expected = int(config['runs'])
        complete = bool(summary and summary['complete'] and summary['completed_runs'] == expected
                        and summary['expected_runs'] == expected
                        and sorted(set(summary['seeds'])) == list(range(expected)))
        key = (config['model'], config['ssl_method'], config['method'],
               int(config['subset_percentage']), int(config['label_percentage']))
        scores[key] = (f'\\dsacc{{{summary["mean_percent"]:.2f}}}'
                       f'{{{summary["std_percent"]:.2f}}}') if complete else ''
        flat.append(dict(config, completed_runs=summary['completed_runs'] if summary else 0,
                         mean_percent=summary['mean_percent'] if complete else '',
                         std_percent=summary['std_percent'] if complete else '',
                         status='complete' if complete else 'incomplete'))
    return scores, flat


def render(dataset, scores, runs):
    display, token = DISPLAY[dataset], TOKEN[dataset]

    def value(model, ssl, method, size, labels):
        return scores.get((model, ssl, method, size, labels), '')

    lines = [
        '', '', '\\begin{table*}[t]', '\\centering',
        '\\def\\dsacc#1#2{$#1${\\footnotesize$\\pm #2$}}',
        f'\\caption{{Consolidated {display} accuracy (\\%) across backbones,',
        'pretraining objectives, and subset sizes. Results are reported as',
        f'mean $\\pm$ sample standard deviation over {runs} independent runs.',
        'All methods use frozen linear probes; No Pre-Training uses',
        'a random encoder.}',
        f'\\label{{tab:{token}-all}}', '',
        '\\setlength{\\tabcolsep}{3pt}',
        '\\renewcommand{\\arraystretch}{1}',
        '\\resizebox{0.8\\textwidth}{!}{%',
        '\\begin{tabular}{@{}cclcccccc@{}}', '\\toprule',
        '\\multirow{2}{*}{\\shortstack{Labeled data\\\\(\\%)}}',
        '& \\multirow{2}{*}{\\shortstack{Pretraining subset\\\\(\\%)}}',
        '& \\multirow{2}{*}{Method}',
        '& \\multicolumn{2}{c}{ConvNet}',
        '& \\multicolumn{2}{c}{VGG11}',
        '& \\multicolumn{2}{c}{ResNet18} \\\\',
        '\\cmidrule(lr){4-5} \\cmidrule(lr){6-7} \\cmidrule(l){8-9}',
        '& & & SimCLR & Barlow Twins',
        '& SimCLR & Barlow Twins',
        '& SimCLR & Barlow Twins \\\\', '\\midrule', ''
    ]
    for block_index, labels in enumerate((1, 5)):
        lines += [
            f'\\multirow{{17}}{{*}}{{{labels}\\%}}',
            '& -- & No Pre-Training',
            f'& \\multicolumn{{2}}{{c}}{{{value("ConvNet", "none", "no_pretrain", 0, labels)}}}',
            f'& \\multicolumn{{2}}{{c}}{{{value("VGG11", "none", "no_pretrain", 0, labels)}}}',
            f'& \\multicolumn{{2}}{{c}}{{{value("ResNet18", "none", "no_pretrain", 0, labels)}}} \\\\',
            '\\cmidrule(lr){2-9}', '',
            '& 100\\% & Full Data',
            f'& {value("ConvNet", "simclr", "full", 100, labels)} & {value("ConvNet", "barlowtwins", "full", 100, labels)}',
            f'& {value("VGG11", "simclr", "full", 100, labels)} & {value("VGG11", "barlowtwins", "full", 100, labels)}',
            f'& {value("ResNet18", "simclr", "full", 100, labels)} & {value("ResNet18", "barlowtwins", "full", 100, labels)} \\\\',
            '\\cmidrule(lr){2-9}', ''
        ]
        for subset in (1, 2, 5):
            for row_index, (method, title) in enumerate((('random', 'Random Subset'), ('kmeans', 'K-means Subset'))):
                prefix = f'& \\multirow{{5}}{{*}}{{{subset}\\%}} & {title}' if row_index == 0 else f'& & {title}'
                lines += [prefix,
                          f'& {value("ConvNet", "simclr", method, subset, labels)} & {value("ConvNet", "barlowtwins", method, subset, labels)}',
                          f'& {value("VGG11", "simclr", method, subset, labels)} & {value("VGG11", "barlowtwins", method, subset, labels)}',
                          f'& {value("ResNet18", "simclr", method, subset, labels)} & {value("ResNet18", "barlowtwins", method, subset, labels)} \\\\']
            lines += [
                '& & KRR-ST & & & & & & \\\\',
                '& & MKDT & & & & & & \\\\',
                '& & Ours & & & & & & \\\\'
            ]
            if subset != 5:
                lines += ['\\cmidrule(lr){2-9}', '']
        if block_index == 0:
            lines += ['', '\\midrule', '']
    lines += ['', '\\bottomrule', '\\end{tabular}%', '}', '\\end{table*}', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, choices=DISPLAY)
    parser.add_argument('--results', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    scores, flat = load_results(args.results)
    runs = int(flat[0]['runs'])
    output = Path(args.output)
    output.write_text(render(args.dataset, scores, runs))
    with output.with_suffix('.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    completed = sum(row['status'] == 'complete' for row in flat)
    print(f'Wrote {output}: {completed}/{len(flat)} complete configurations.')


if __name__ == '__main__':
    main()
