"""Write a compact completion/results CSV for a full-data refresh batch."""

import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', required=True)
    args = parser.parse_args()
    batch = Path(args.batch)
    rows = []
    for path in sorted((batch / 'runs').glob('*/evaluations/*/job_*/summary.json')):
        summary = json.loads(path.read_text())
        config = summary['config']
        rows.append({
            'source': config['source'], 'target': config['target'],
            'model': config['model'], 'ssl_method': config['ssl_method'],
            'method': 'full', 'subset_percentage': 100,
            'label_percentage': config['label_percentage'],
            'ssl_epochs': config['ssl_epochs'],
            'probe_epochs': config['probe_epochs'],
            'completed_runs': summary['completed_runs'],
            'complete': summary['complete'],
            'mean_percent': summary['mean_percent'],
            'std_percent': summary['std_percent'],
            'summary': str(path),
        })
    destination = batch / 'full_data_results.csv'
    fields = list(rows[0]) if rows else [
        'source', 'target', 'model', 'ssl_method', 'method',
        'subset_percentage', 'label_percentage', 'ssl_epochs', 'probe_epochs',
        'completed_runs', 'complete', 'mean_percent', 'std_percent', 'summary']
    with destination.open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f'Wrote {len(rows)} result cells to {destination}.')


if __name__ == '__main__':
    main()
