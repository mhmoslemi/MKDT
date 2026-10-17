"""Build verified seed checklists and mean/sample-standard-deviation table rows."""
import csv
import fcntl
import json
import os
from pathlib import Path
import statistics

from common import atomic_json, file_hash


def write_tsv(path, columns, rows):
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    with temporary.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, columns, delimiter='\t', extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def report(root, plan):
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.report.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        values, fingerprints, errors, checklist = {}, {}, [], []
        valid_encoders = 0
        signatures = {}
        for job in plan['jobs']:
            for unit in job['units']:
                folder = root / unit['relative']
                marker_path = folder / 'complete.json'
                rows, status = {}, 'pending'
                if marker_path.exists():
                    try:
                        marker = json.loads(marker_path.read_text())
                        if not marker.get('complete') or not marker['checkpoint_retained']:
                            raise ValueError('Incomplete marker or missing retention guarantee')
                        signature = marker['identity']['signature']
                        if unit['group'] in signatures and signatures[unit['group']] != signature:
                            raise ValueError('Mixed checkpoint signatures in a ten-seed group')
                        candidate_fingerprints = {}
                        for cell in unit['cells']:
                            name = f"{cell['target']}_lbl{cell['labels']}.json"
                            file = folder / 'evaluations' / name
                            row = json.loads(file.read_text())
                            expected = row['expected']
                            if (not row.get('complete') or not row['encoder_unchanged']
                                    or row['encoder_sha256'] != marker['encoder_sha256']
                                    or expected['identity'] != marker['identity']
                                    or expected['seed'] != unit['seed'] or expected['source'] != unit['source']
                                    or expected['target'] != cell['target']
                                    or expected['label_percentage'] != int(cell['labels'])
                                    or expected['evaluation_code_hashes'] != marker['evaluation_code_hashes']
                                    or file_hash(file) != marker['result_hashes'][name]):
                                raise ValueError('Invalid result: ' + str(file))
                            key = (cell['target'],)
                            pair = (expected['train_fingerprint'], expected['test_fingerprint'])
                            if key in fingerprints and fingerprints[key] != pair:
                                raise ValueError('Mixed target datasets: ' + cell['target'])
                            candidate_fingerprints[key] = pair
                            rows[(cell['target'], cell['labels'])] = row['accuracy_percent']
                        fingerprints.update(candidate_fingerprints)
                        signatures[unit['group']] = signature
                        valid_encoders += 1
                        status = 'complete'
                    except Exception as error:
                        rows = {}
                        status = 'invalid'
                        errors.append({'encoder': unit['relative'], 'error': str(error)})
                for cell in unit['cells']:
                    key = (unit['group'], cell['target'], cell['labels'])
                    score = rows.get((cell['target'], cell['labels']))
                    if score is not None:
                        values.setdefault(key, {})[unit['seed']] = score
                    checklist.append({**cell, 'task': job['task'], 'seed': unit['seed'],
                        'status': status, 'accuracy_percent': '' if score is None else score,
                        'result': str(folder / 'evaluations' / f"{cell['target']}_lbl{cell['labels']}.json")})
        summaries = []
        for group in plan['groups']:
            group_rows = []
            for cell in group['cells']:
                scores = values.get((group['group'], cell['target'], cell['labels']), {})
                complete = set(scores) == set(range(10))
                row = {**cell, 'completed_seeds': len(scores), 'expected_seeds': 10,
                    'complete': complete, 'mean_percent': '', 'std_percent': '', 'latex': ''}
                if complete:
                    mean, std = statistics.mean(scores.values()), statistics.stdev(scores.values())
                    row.update(mean_percent=mean, std_percent=std, latex=f'\\dsacc{{{mean:.2f}}}{{{std:.2f}}}')
                summaries.append(row)
                group_rows.append({**row, 'accuracies_percent': [scores[s] for s in sorted(scores)],
                                   'seeds': sorted(scores)})
            summary = root / group['group'] / 'summary.json'
            if all(row['complete'] for row in group_rows):
                atomic_json(summary, {'complete': True, 'signature': signatures[group['group']],
                    'std_ddof': 1, 'results': group_rows})
            else:
                # Do not leave an old successful summary after corruption is found.
                summary.unlink(missing_ok=True)
        write_tsv(root / 'checklist.tsv', ['task','table','source','model','subset','ssl','seed',
                  'target','labels','status','accuracy_percent','result'], checklist)
        write_tsv(root / 'table_summary.tsv', ['table','line','source','model','subset','ssl','target','labels',
                  'completed_seeds','expected_seeds','complete','mean_percent','std_percent','latex'], summaries)
        complete_cells = sum(row['complete'] for row in summaries)
        status = {'complete': valid_encoders == 540 and complete_cells == 324 and not errors,
            'completed_encoders': valid_encoders, 'expected_encoders': 540,
            'completed_probes': sum(len(v) for v in values.values()), 'expected_probes': 3240,
            'completed_table_cells': complete_cells, 'expected_table_cells': 324,
            'invalid_encoders': errors}
        atomic_json(root / 'progress.json', status)
        return status
