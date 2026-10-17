"""Atomic coverage and mean/sample-SD reports; never edit the TeX tables."""
import csv
import fcntl
import json
import math
import os
from pathlib import Path
import statistics

from common import atomic_json, digest_json, file_hash


def write_tsv(path, columns, rows):
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    with temporary.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, columns, delimiter='\t', extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def read(path):
    return json.loads(path.read_text())


def report(root, plan, hashes, driver_hashes):
    with (root / '.report.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        values, fingerprints, signatures = {}, {}, {}
        errors, checklist, timings = [], [], []
        completed = 0
        for job in plan['jobs']:
            for unit in job['units']:
                folder = root / unit['relative']
                marker_path = folder / 'complete.json'
                scores, times, status = {}, {}, 'pending'
                if marker_path.exists():
                    try:
                        marker = read(marker_path)
                        config = read(folder / 'config.json')
                        identity = config['identity']
                        signature = identity['signature']
                        expected_probes = [[c['target'], int(c['labels'])] for c in unit['cells']]
                        if (not marker.get('complete') or not marker['checkpoint_retained']
                                or marker['identity'] != identity
                                or digest_json(config['signature']) != signature
                                or identity['seed'] != unit['seed']
                                or identity['selection_method'] != plan['method']
                                or marker['evaluations'] != expected_probes
                                or marker['evaluation_code_hashes'] != hashes
                                or marker['driver_hashes'] != driver_hashes):
                            raise ValueError('Invalid encoder evaluation marker')
                        if unit['group'] in signatures and signatures[unit['group']] != signature:
                            raise ValueError('Mixed SSL signatures in a seed group')
                        candidate_fingerprints = {}
                        for cell in unit['cells']:
                            path = folder / 'evaluations' / f"{cell['target']}_lbl{cell['labels']}.json"
                            row = read(path)
                            expected = row['expected']
                            score = row['accuracy_percent']
                            if (not row.get('complete') or not row['encoder_unchanged']
                                    or row['encoder_sha256'] != marker['encoder_sha256']
                                    or expected['identity'] != identity or expected['seed'] != unit['seed']
                                    or expected['source'] != unit['source'] or expected['target'] != cell['target']
                                    or expected['label_percentage'] != int(cell['labels'])
                                    or expected['evaluation'] != config['signature']['protocol']['evaluation']
                                    or expected['evaluation_code_hashes'] != hashes
                                    or file_hash(path) != marker['result_hashes'][path.name]
                                    or not math.isfinite(score) or not 0 <= score <= 100):
                                raise ValueError(f'Invalid probe result: {path}')
                            pair = (expected['train_fingerprint'], expected['test_fingerprint'])
                            if cell['target'] in fingerprints and fingerprints[cell['target']] != pair:
                                raise ValueError('Mixed target datasets')
                            candidate_fingerprints[cell['target']] = pair
                            scores[(cell['target'], cell['labels'])] = score
                            times[(cell['target'], cell['labels'])] = row['seconds']
                        fingerprints.update(candidate_fingerprints)
                        signatures[unit['group']] = signature
                        completed += 1
                        status = 'complete'
                    except Exception as error:
                        scores, times, status = {}, {}, 'invalid'
                        errors.append({'encoder': unit['relative'], 'error': str(error)})
                for cell in unit['cells']:
                    key = (unit['group'], cell['target'], cell['labels'])
                    score = scores.get((cell['target'], cell['labels']))
                    if score is not None:
                        values.setdefault(key, {})[unit['seed']] = score
                        timings.append({**cell, 'task': job['task'], 'seed': unit['seed'],
                                        'seconds': times[(cell['target'], cell['labels'])]})
                    checklist.append({**cell, 'task': job['task'], 'seed': unit['seed'], 'status': status,
                        'accuracy_percent': '' if score is None else score,
                        'result': str(folder / 'evaluations' / f"{cell['target']}_lbl{cell['labels']}.json")})
        summaries = []
        for group in plan['groups']:
            group_rows = []
            for cell in group['cells']:
                scores = values.get((group['group'], cell['target'], cell['labels']), {})
                complete = set(scores) == set(plan['seeds'])
                row = {**cell, 'complete': complete, 'completed_seeds': len(scores),
                       'expected_seeds': len(plan['seeds']), 'mean_percent': '', 'std_percent': '', 'latex': ''}
                if complete:
                    mean, std = statistics.mean(scores.values()), statistics.stdev(scores.values())
                    row.update(mean_percent=mean, std_percent=std, latex=f'\\dsacc{{{mean:.2f}}}{{{std:.2f}}}')
                summaries.append(row)
                group_rows.append({**row, 'seeds': sorted(scores),
                                   'accuracies_percent': [scores[s] for s in sorted(scores)]})
            atomic_json(root / group['group'] / 'summary.json', {
                'complete': all(r['complete'] for r in group_rows), 'std_ddof': 1,
                'signature': signatures.get(group['group']), 'results': group_rows})
        write_tsv(root / 'checklist.tsv', ['task', 'table', 'line', 'source', 'model', 'method',
                  'subset', 'ssl', 'seed', 'target', 'labels', 'status', 'accuracy_percent', 'result'], checklist)
        write_tsv(root / 'table_summary.tsv', ['table', 'line', 'source', 'model', 'method', 'subset',
                  'ssl', 'target', 'labels', 'completed_seeds', 'expected_seeds', 'complete',
                  'mean_percent', 'std_percent', 'latex'], summaries)
        write_tsv(root / 'probe_times.tsv', ['task', 'source', 'model', 'method', 'subset',
                  'ssl', 'seed', 'target', 'labels', 'seconds'], timings)
        expected_encoders = sum(len(j['units']) for j in plan['jobs'])
        expected_probes = sum(j['probes'] for j in plan['jobs'])
        complete_cells = sum(r['complete'] for r in summaries)
        status = {'complete': completed == expected_encoders and complete_cells == len(summaries) and not errors,
            'completed_encoders': completed, 'expected_encoders': expected_encoders,
            'completed_probes': sum(len(v) for v in values.values()), 'expected_probes': expected_probes,
            'completed_table_cells': complete_cells, 'expected_table_cells': len(summaries),
            'invalid_encoders': errors}
        atomic_json(root / 'progress.json', status)
        return status
