"""Generate portable runners and check every table cell has a runner.

Standard library only. Commands: generate, check, inputs. Does not launch training.
"""

import argparse
import csv
import io
import json
from pathlib import Path
import re
import shlex

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DISPLAY = {'No Pre-Training': 'none', 'Full Data': 'full', 'Random Subset': 'random',
           'K-means Subset': 'kmeans', 'MKDT': 'mkdt', 'Ours': 'ours'}
SLUGS = {'cifar10': 'CIFAR10', 'cifar100': 'CIFAR100', 'tinyimagenet': 'TinyImageNet'}
TARGET_NAMES = {'CIFAR-10': 'CIFAR10', 'CIFAR-100': 'CIFAR100', 'Tiny ImageNet': 'TinyImageNet',
                'Aircraft': 'Aircraft', 'CUB-2011': 'CUB2011', 'Dogs': 'Dogs', 'Flowers': 'Flowers'}


def key(value):
    return format(float(value), 'g')


def runner_name(source, model, method, subset, ssl, seed=None):
    suffix = f'seed{seed}' if method == 'full' else '10seeds'
    return f'{source}_{model}_{method}_pt{key(subset).replace(".", "p")}_{ssl}_{suffix}.sh'


def build_runners(protocol):
    result = []
    for source in protocol['sources']:
        for model in protocol['models']:
            specs = [('none', 0, 'none', protocol['subset_seeds'])]
            for ssl in protocol['methods']:
                specs += [('full', 100, ssl, [seed]) for seed in protocol['full_seeds']]
                specs += [(method, subset, ssl, protocol['subset_seeds'])
                          for method in ('random', 'kmeans', 'mkdt', 'ours')
                          for subset in protocol['subset_percentages']]
            for method, subset, ssl, seeds in specs:
                name = runner_name(source, model, method, subset, ssl, seeds[0])
                args = ['--source', source, '--model', model, '--method', method, '--subset', key(subset),
                        '--ssl', ssl, '--seeds', *map(str, seeds)]
                result.append({'script': 'scripts/' + name, 'source': source, 'model': model,
                               'method': method, 'subset': key(subset), 'ssl': ssl,
                               'seeds': ','.join(map(str, seeds)), 'arguments': args})
    return result


def script_text(row):
    return ('#!/bin/bash\nset -euo pipefail\n'
            'runner_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"\n'
            'source "$runner_dir/../common.sh"\n'
            f'run_benchmark "${{BASH_SOURCE[0]}}" {shlex.join(row["arguments"])} "$@"\n')


def table_cells(path):
    slug, scope = path.stem.rsplit('_', 1)
    if slug not in SLUGS or scope not in ('main', 'downstream'):
        raise ValueError(f'Unrecognized table requires an explicit mapping: {path}')
    source = SLUGS[slug]
    lines = path.read_text().splitlines()
    headings = re.findall(r'\\multicolumn\{2\}\{c\}\{([^{}]+)\}', '\n'.join(lines).split('\\midrule')[0])
    if scope == 'main':
        if headings != ['ConvNet', 'VGG11', 'ResNet18']:
            raise ValueError(f'Unexpected model columns: {path}: {headings}')
        columns = [(model, source) for model in headings]
    else:
        columns = [('ResNet18', TARGET_NAMES[title]) for title in headings]
        if len(columns) != 6:
            raise ValueError(f'Expected six downstream datasets: {path}')
    label = subset = None
    for number, raw in enumerate(lines, 1):
        disabled = raw.lstrip().startswith('%')
        line = re.sub(r'^\s*%\s?', '', raw) if disabled else raw
        match = re.match(r'^\s*\\multirow\{\d+\}\{\*\}\{([\d.]+)\\%\}', line)
        if match:
            label = key(match.group(1))
        match = re.search(r'&\s*\\multirow\{\d+\}\{\*\}\{([\d.]+)\\%\}', line)
        if match:
            subset = key(match.group(1))
        for title, method in DISPLAY.items():
            if not re.search(r'&\s*' + re.escape(title) + r'(?=\s*(?:&|$))', line):
                continue
            if label is None:
                raise ValueError(f'Missing label block: {path}:{number}')
            size = '100' if method == 'full' else '0' if method == 'none' else subset
            if size is None:
                raise ValueError(f'Missing subset percentage: {path}:{number}')
            for model, target in columns:
                for ssl in (('none',) if method == 'none' else ('simclr', 'barlowtwins')):
                    yield {'table': path.name, 'line': number, 'disabled': disabled, 'source': source,
                           'model': model, 'target': target, 'method': method, 'subset': size,
                           'ssl': ssl, 'labels': label}


def coverage(runners):
    lookup = {}
    for row in runners:
        identifier = tuple(row[k] for k in ('source', 'model', 'method', 'subset', 'ssl'))
        lookup.setdefault(identifier, []).append(row['script'])
    rows = []
    for path in sorted((ROOT / 'CVPR_tables').glob('*.tex')):
        for cell in table_cells(path):
            identifier = tuple(cell[k] for k in ('source', 'model', 'method', 'subset', 'ssl'))
            if identifier not in lookup:
                raise ValueError(f'Uncovered cell: {cell}')
            rows.append({**cell, 'scripts': ';'.join(lookup[identifier])})
    if not rows:
        raise ValueError('No table cells discovered')
    return rows


def tsv_text(rows):
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=list(rows[0]), delimiter='\t', lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('generate', 'check', 'inputs'))
    args = parser.parse_args()
    protocol = json.loads((HERE / 'protocol.json').read_text())
    runners = build_runners(protocol)
    cells = coverage(runners)
    if args.command == 'generate':
        (HERE / 'scripts').mkdir(exist_ok=True)
        for row in runners:
            path = HERE / row['script']
            path.write_text(script_text(row))
            path.chmod(0o755)
        (HERE / 'runners.tsv').write_text(tsv_text([{k: v for k, v in row.items() if k != 'arguments'} for row in runners]))
        (HERE / 'table_coverage.tsv').write_text(tsv_text(cells))
    elif args.command == 'check':
        for row in runners:
            path = HERE / row['script']
            if not path.exists() or path.read_text() != script_text(row):
                raise ValueError(f'Missing or changed runner: {path}')
        expected = tsv_text([{k: v for k, v in row.items() if k != 'arguments'} for row in runners])
        if (HERE / 'runners.tsv').read_text() != expected or (HERE / 'table_coverage.tsv').read_text() != tsv_text(cells):
            raise ValueError('Stale runner or table coverage manifest; regenerate')
    else:
        entries = json.loads((HERE / 'inputs.json').read_text())
        missing = []
        for method in ('mkdt', 'ours'):
            for source in protocol['sources']:
                for size in protocol['subset_percentages']:
                    entry = entries[method][source][key(size)]
                    path = Path(entry['path']) if entry.get('path') else None
                    if path is not None and not path.is_absolute():
                        path = ROOT / path
                    if path is None or not path.is_file():
                        missing.append(f'{method}/{source}/{key(size)}%: {path or "NOT REGISTERED"}')
        print('\n'.join(missing) if missing else 'All synthetic input paths exist.')
        if missing:
            raise SystemExit(2)
    print(f'{len(runners)} runners; {len(cells)} logical table cells covered '
          f'({sum(not row["disabled"] for row in cells)} active).')


if __name__ == '__main__':
    main()
