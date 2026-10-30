"""Add five probes per label budget to the completed K=8 SVHN encoder."""
import json
import math
import os
from pathlib import Path
import re
import statistics
import sys

import torch

HERE = Path(__file__).resolve().parent
BENCH, ROOT = HERE.parent, HERE.parent.parent
sys.path.insert(0, str(ROOT / 'OT-SSL-DD'))
import eval as reference_eval


def read(path):
    return json.loads(Path(path).read_text())


def update_table(summaries, output, common):
    table = ROOT / 'CVPR_tables/table.tex'
    with common.exclusive_run(HERE / 'table_update'):
        original = table.read_text()
        lines = original.splitlines(keepends=True)
        rows = [i for i, line in enumerate(lines) if line.startswith(r'\multirow{3}{*}{SVHN}')]
        assert len(rows) == 1
        index = rows[0]
        cells = lines[index].split('&')
        assert len(cells) == 12 and float(cells[1].strip()) == 0.2
        for labels, column in ((1, 5), (5, 10)):
            summary = summaries[labels]
            cells[column] = ' \\dsacc{%.2f}{%.2f} ' % (summary['mean_percent'], summary['std_percent'])
        for columns in ((2, 3, 4, 5), (7, 8, 9, 10)):
            scores = {}
            for column in columns:
                cells[column] = cells[column].replace(r'\bestacc', r'\dsacc')
                match = re.search(r'\\dsacc\{([\d.]+)\}\{([\d.]+)\}', cells[column])
                if match:
                    scores[column] = float(match[1])
            assert scores
            for column, score in scores.items():
                if score == max(scores.values()):
                    cells[column] = cells[column].replace(r'\dsacc', r'\bestacc')
        previous = lines[index]
        lines[index] = '&'.join(cells)
        # Never alter another method's reported numbers while refreshing bolding.
        for column in (2, 3, 4, 6, 7, 8, 9, 11):
            numbers = lambda cell: re.findall(r'\\(?:dsacc|bestacc)\{([\d.]+)\}\{([\d.]+)\}', cell)
            assert numbers(previous.split('&')[column]) == numbers(cells[column])
        updated = ''.join(lines)
        note = '% SVHN MIPA 0.2%: K=8, one 2k-epoch encoder, 10 probes per label budget; ' + str(output.relative_to(ROOT)) + '/summary_SVHN_lbl{1,5}_10probes.json\n'
        if note not in updated:
            updated = updated.replace('% MIPA results:', note + '% MIPA results:', 1)
        backup = output / 'table_before_extra_probes.tex'
        if not backup.exists():
            backup.write_text(original)
        temporary = table.with_name('.table.tex.%d.tmp' % os.getpid())
        temporary.write_text(updated)
        if table.read_text() != original:
            temporary.unlink()
            raise RuntimeError('Table changed during update; rerun to apply the completed results')
        temporary.replace(table)
        print('UPDATED table.tex: ' + lines[index].strip(), flush=True)


def main():
    manifest = Path(sys.argv[1]).resolve()
    assert manifest.is_relative_to(HERE / 'batches')
    job = read(manifest)
    assert os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_JOB_ACCOUNT') == job['account'] == 'aip-yiweilu'
    assert os.environ.get('SLURM_CLUSTER_NAME', os.environ.get('CC_CLUSTER')) == 'vulcan'
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    reference = reference_eval.reference_modules()
    common = reference['common']
    for file, sha in job['code_hashes'].items():
        assert common.file_hash(ROOT / file) == sha, 'Code changed: ' + file
    folder = ROOT / job['folder']
    assert folder.is_relative_to(HERE / 'batches')
    assert common.file_hash(folder / 'config.json') == job['config_sha256']
    checkpoint = folder / 'encoder.pt'
    assert common.file_hash(checkpoint) == job['encoder_file_sha256']
    config, marker = read(folder / 'config.json'), read(folder / 'ssl_complete.json')
    signature, identity = config['signature'], config['identity']
    assert identity['signature'] == common.digest_json(signature)
    assert marker['complete'] and marker['epochs'] == 2000 and marker['identity'] == identity
    assert marker['checkpoint_file_sha256'] == job['encoder_file_sha256']
    assert signature['source'] == 'SVHN' and signature['subset'] == 0.2 and signature['ssl'] == 'simclr'
    assert signature['generation_args']['cosine_views'] == 8
    protocol = signature['protocol']
    assert protocol['evaluation']['epochs'] == 100
    train_set, test_set = (reference['datasets'].DigitDataset(job['data_root'], 'SVHN', flag) for flag in (True, False))
    summaries = {}
    with common.exclusive_run(manifest.parent / 'session'):
        for labels in (1, 5):
            samples = []
            for seed in range(10):
                result_file = folder / 'evaluations' / ('SVHN_lbl%d_probe%d.json' % (labels, seed))
                result = reference['evaluate'].linear_probe(checkpoint, identity, protocol, 'SVHN', 'SVHN',
                    train_set, test_set, labels, seed, result_file, workers=0, device='cuda')
                assert result['complete'] and result['encoder_unchanged'] and result['encoder_sha256'] == marker['encoder_sha256']
                assert math.isfinite(result['accuracy_percent'])
                samples.append(dict(ssl_seed=0, probe_seed=seed, accuracy_percent=result['accuracy_percent'], result_file=str(result_file)))
            scores = [sample['accuracy_percent'] for sample in samples]
            summary = dict(read(folder / ('summary_SVHN_lbl%d.json' % labels)),
                samples=samples, probe_seeds=list(range(10)), mean_percent=statistics.mean(scores),
                std_percent=statistics.stdev(scores), std_ddof=1, complete=True, existing_probes=5, added_probes=5)
            common.atomic_json(folder / ('summary_SVHN_lbl%d_10probes.json' % labels), summary)
            summaries[labels] = summary
            print('SVHN 0.2%% lbl=%d: %.2f +/- %.2f (10 probes, one fixed encoder)' %
                  (labels, summary['mean_percent'], summary['std_percent']), flush=True)
        assert common.file_hash(checkpoint) == job['encoder_file_sha256'], 'SSL checkpoint changed'
        update_table(summaries, folder, common)
        common.atomic_json(manifest.parent / 'complete.json', dict(complete=True, labels=[1, 5],
            probe_seeds=list(range(10)), encoder_unchanged=True, table='CVPR_tables/table.tex'))


if __name__ == '__main__':
    main()
