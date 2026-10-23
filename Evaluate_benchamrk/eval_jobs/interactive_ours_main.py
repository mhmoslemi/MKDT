"""Main-table scheduling only; preserve the original SSL/probe checkpoint identities."""
import argparse
import fcntl
import itertools
import sys

import interactive_ours as ours

ALL_MODELS = ours.MODELS
original_exclusive_run = ours.exclusive_run
original_atomic_json = ours.atomic_json


def main_targets(source, model):
    return (source,)


def write_main_report(args):
    with (args.output / args.source / '.main_report.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        rows = []
        for model, percentage, method in itertools.product(ALL_MODELS, ours.SIZES, ours.METHODS):
            path = args.output / args.source / model / ('pt%g' % percentage) / method / 'summary.json'
            if path.is_file():
                rows.extend(row for row in ours.read(path)['cells'] if row['target'] == args.source)
        assert all(row['target'] == args.source for row in rows)
        ours.write_tsv(args.output / args.source / 'table_summary.tsv',
                       ['source', 'model', 'subset', 'ssl', 'target', 'labels', 'completed_probes',
                        'expected_probes', 'complete', 'mean_percent', 'std_percent'], rows)


def network_session_lock(folder):
    if folder.name == 'interactive_session':
        folder = folder.parent / ours.MODELS[0] / folder.name
    return original_exclusive_run(folder)


def network_session_json(path, value):
    if path.name == 'session.json':
        path = path.parent / ours.MODELS[0] / path.name
    return original_atomic_json(path, value)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--model', choices=ALL_MODELS, default=None)
    selection, remaining = parser.parse_known_args()
    sys.argv[1:] = remaining
    if selection.model is not None:
        ours.MODELS = (selection.model,)
        ours.exclusive_run = network_session_lock
        ours.atomic_json = network_session_json
        print('NETWORK:', selection.model, flush=True)
    # Scheduling is outside the hashed evaluator so existing encoders/probes remain reusable.
    ours.evaluation_targets = main_targets
    ours.write_report = write_main_report
    print('MAIN TABLE ONLY: no downstream datasets or probes.', flush=True)
    raise SystemExit(ours.main())
