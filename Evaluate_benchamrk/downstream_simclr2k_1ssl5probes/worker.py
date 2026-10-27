"""Five standard frozen probes per downstream cell; no SSL training."""
import contextlib
import fcntl
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent
ROOT = BENCH.parent
sys.path.insert(0, str(BENCH))
import torch
from common import atomic_json, code_hashes, digest_json, exclusive_run, file_hash, load_encoder
from data import RawDataset
from evaluate import linear_probe


def read(path):
    return json.loads(Path(path).read_text())


def inside(value):
    folder = (BENCH / value).resolve()
    if not folder.is_relative_to(BENCH) or folder == BENCH:
        raise ValueError('Path must stay inside Evaluate_benchamrk')
    return folder


def run(cell, data_root):
    source, target, subset, kind, labels = (cell[k] for k in ('source', 'target', 'subset', 'kind', 'labels'))
    if (source not in ('CIFAR10', 'CIFAR100', 'TinyImageNet') or target == source
            or target not in ('CIFAR10', 'CIFAR100', 'TinyImageNet', 'Aircraft', 'CUB2011', 'Dogs')
            or subset not in (0.2, 1) or kind not in ('random', 'kmeans') or labels not in (1, 5)
            or (source == 'CIFAR10' and subset == 1 and labels == 5)):
        raise ValueError('Cell is outside the requested downstream replacement scope')
    folder, output = inside(cell['checkpoint_dir']), inside(cell['output_dir'])
    expected_root = 'subset_simclr2k_1ssl5probes' if source == 'TinyImageNet' else 'cifar_small_simclr2k_1ssl5probes'
    relative = Path(source) / 'ResNet18' / kind / ('pt%g' % subset) / 'simclr/ssl_seed_000'
    if folder != BENCH / expected_root / 'results' / relative or output != HERE / 'results' / relative:
        raise ValueError('Unexpected encoder or output directory')
    with contextlib.ExitStack() as locks:
        lock = locks.enter_context((folder / '.lock').open('a'))
        fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        if (file_hash(folder / 'config.json') != cell['config_sha256']
                or file_hash(folder / 'ssl_complete.json') != cell['marker_sha256']):
            raise ValueError('Encoder metadata changed since submission')
        config, marker = read(folder / 'config.json'), read(folder / 'ssl_complete.json')
        signature, identity = config['signature'], config['identity']
        protocol = signature['protocol']
        if (digest_json(signature) != identity['signature'] or signature['code_hashes'] != code_hashes()
                or [signature['source'], signature['model'], signature['selection_method'], signature['subset'], signature['ssl'], identity['seed']]
                != [source, 'ResNet18', kind, subset, 'simclr', 0]
                or not marker['complete'] or marker['identity'] != identity or marker['epochs'] != 2000
                or protocol['ssl']['subset_epochs'] != 2000 or protocol['evaluation']['epochs'] != 100
                or protocol['evaluation'] != read(BENCH / 'protocol.json')['evaluation']):
            raise ValueError('Encoder identity or standard evaluation recipe mismatch')
        checkpoint = folder / 'encoder.pt'
        if file_hash(checkpoint) != marker['checkpoint_file_sha256']:
            raise ValueError('Checkpoint file differs from completion marker')
        encoder, _, saved = load_encoder(checkpoint, identity, 'cpu')
        if (saved['epochs'] != 2000 or saved['encoder_sha256'] != marker['encoder_sha256']
                or saved['model'] != 'ResNet18' or saved['method'] != 'simclr' or saved['source'] != source):
            raise ValueError('Not the requested trained 2k SimCLR encoder')
        del encoder, saved
        with exclusive_run(output / ('locks/%s_lbl%d' % (target, labels))):
            train_set, test_set = RawDataset(data_root, target, True), RawDataset(data_root, target, False)
            samples = []
            for seed in range(5):
                path = output / 'evaluations' / ('%s_lbl%d_probe%d.json' % (target, labels, seed))
                print('SKIP saved probe:' if path.exists() else 'RUN probe:', source, kind, subset, target, labels, seed, flush=True)
                result = linear_probe(checkpoint, identity, protocol, source, target, train_set, test_set,
                                      labels, seed, path, workers=0, device='cuda')
                score = result['accuracy_percent']
                if (not result['complete'] or not result['encoder_unchanged']
                        or result['encoder_sha256'] != marker['encoder_sha256']
                        or not math.isfinite(score) or not 0 <= score <= 100):
                    raise ValueError('Invalid frozen probe result')
                samples.append({'ssl_seed': 0, 'probe_seed': seed, 'accuracy_percent': score,
                                'encoder_sha256': marker['encoder_sha256'], 'result_file': str(path)})
            scores = [s['accuracy_percent'] for s in samples]
            summary = {key: cell[key] for key in ('source', 'target', 'subset', 'kind', 'labels')}
            summary.update(complete=True, model='ResNet18', ssl='simclr', ssl_seed=0, ssl_epochs=2000,
                           linear_epochs=100, std_ddof=1, expected_probes=5, completed_probes=5, samples=samples,
                           mean_percent=statistics.mean(scores), std_percent=statistics.stdev(scores),
                           table_scope='CVPR_tables/cifar_downstream.tex', checkpoint=str(checkpoint))
            summary['tex_value'] = '\\dsacc{%.1f}{%.1f}' % (summary['mean_percent'], summary['std_percent'])
            summary['cell_key'] = '|'.join(map(str, (source, 'ResNet18', kind, subset, 'simclr', target, labels)))
            path = output / ('summary_%s_lbl%d.json' % (target, labels))
            atomic_json(path, summary)
            with (ROOT / 'CVPR_tables/.probe_completion.lock').open('a') as table_lock:
                fcntl.flock(table_lock, fcntl.LOCK_EX)
                subprocess.run(['node', str(HERE / 'jobs.js'), '--update', str(path)], check=True)


def main():
    if (not os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_JOB_ACCOUNT') != 'aip-boyuwang'
            or os.environ.get('SLURM_CLUSTER_NAME', os.environ.get('CC_CLUSTER')) != 'vulcan'
            or not torch.cuda.is_available()):
        raise RuntimeError('Use a Wang GPU job on Vulcan')
    manifest = Path(sys.argv[1]).resolve()
    if not manifest.is_relative_to(HERE / 'batches'):
        raise ValueError('Unexpected bundle manifest')
    job = read(manifest)
    if job['stage'] != 'downstream_probe_bundle' or not 1 <= len(job['cells']) <= 4:
        raise ValueError('Expected at most 20 frozen probes per bundle')
    for relative, expected in job['code_hashes'].items():
        if file_hash(ROOT / relative) != expected:
            raise ValueError('Code changed since submission: ' + relative)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    for cell in job['cells']:
        run(cell, Path(sys.argv[2]).resolve())
    print('BUNDLE COMPLETE:', job['name'], flush=True)


if __name__ == '__main__':
    main()
