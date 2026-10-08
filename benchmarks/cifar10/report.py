"""Build the requested table panels, counting only complete 15-seed configurations."""

import argparse
import csv
import json
from pathlib import Path


def render(results, output):
    results = Path(results)
    manifest = Path(__file__).with_name('configs.tsv')
    if (results / 'configs.tsv').exists():
        manifest = results / 'configs.tsv'
    with manifest.open() as handle:
        configs = list(csv.DictReader(handle, delimiter='\t'))
    available = {}
    flat = []
    for config in configs:
        candidates = sorted((results / config['config_id']).glob('job_*/summary.json'))
        if len(candidates) > 1:
            raise ValueError(f'Multiple submissions for {config["config_id"]}; select one before reporting.')
        summary = json.loads(candidates[0].read_text()) if candidates else None
        complete = bool(summary and summary['complete'] and summary['completed_runs'] == 15 and len(set(summary['seeds'])) == 15)
        key = (config['model'], config['ssl_method'], config['method'], int(config['subset_percentage']), int(config['label_percentage']))
        if complete:
            text = f'{summary["mean_percent"]:.2f} ± {summary["std_percent"]:.2f}'
        elif summary:
            text = f'Incomplete ({summary["completed_runs"]}/15)'
        else:
            text = '—'
        available[key] = text
        flat.append(dict(config, completed_runs=summary['completed_runs'] if summary else 0,
                         mean_percent=summary['mean_percent'] if complete else '',
                         std_percent=summary['std_percent'] if complete else '',
                         status='complete' if complete else 'incomplete'))
    lines = []
    for model in ('ConvNet', 'VGG11', 'ResNet18'):
        for ssl in ('simclr', 'barlowtwins'):
            for subset in (1, 2, 5):
                lines += [f'### {model} · {ssl} · {subset}% pretraining subset', '',
                          '| Labeled data | Method | CIFAR-10 accuracy (%) |', '|---|---|---:|']
                for labels in (1, 5):
                    for method, display, method_ssl, size in (
                        ('no_pretrain', 'No pretraining', 'none', 0),
                        ('random', 'Random subset', ssl, subset),
                        ('kmeans', 'K-means subset', ssl, subset),
                        ('full', 'Full data', ssl, 100),
                    ):
                        score = available.get((model, method_ssl, method, size, labels), '—')
                        lines.append(f'| {labels}% | {display} | {score} |')
                lines.append('')
    output = Path(output)
    output.write_text('\n'.join(lines))
    with output.with_suffix('.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    print(f'Wrote {output}: {sum(row["status"] == "complete" for row in flat)}/{len(flat)} complete configurations.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    render(args.results, args.output)
