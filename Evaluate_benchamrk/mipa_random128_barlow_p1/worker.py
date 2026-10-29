"""Repeat the existing 1% Barlow evaluation with the finalized batch-128 MIPA sets."""
import fcntl
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent
ROOT = BENCH.parent
sys.path[:0] = [str(BENCH / 'eval_jobs'), str(BENCH)]

import torch
import interactive_ours as ours
import train_ssl
from common import atomic_json, code_hashes, digest_json, exclusive_run, file_hash
from data import RawDataset
from evaluate import linear_probe


def read(path):
    return json.loads(Path(path).read_text())


def prepare(job, data_root):
    assert job['source'] in ('CIFAR10', 'CIFAR100') and job['seed'] in (0, 1)
    for name, sha in job['code_hashes'].items():
        assert file_hash(ROOT / name) == sha, 'Code changed: ' + name
    path = ROOT / job['synthetic_file']
    assert file_hash(path) == job['synthetic_sha256']
    contract = read(path.parent / 'pipeline_config.json')
    state = read(path.parent / 'generation_state.json')
    assert state['complete'] and state['iteration'] == 8000 and state['contract'] == contract
    assert contract['generation_args']['batch_real'] == 128
    assert contract['generation_args']['percentage'] == 1
    saved = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    assert saved['run_args'] == contract['generation_args'] and saved['initialization'] == contract['selection']
    del saved
    loaded = ours.load_synthetic(path, job['source'], 'ResNet18BN', 1)
    assert loaded is not None
    synthetic, metadata = loaded
    metadata['file_sha256'] = job['synthetic_sha256']
    old = read(ROOT / job['recipe_file'])
    assert file_hash(ROOT / job['recipe_file']) == job['recipe_sha256']
    signature = dict(old['signature'])
    assert signature['code_hashes'] == code_hashes()
    assert signature['source'] == job['source'] and signature['model'] == 'ResNet18'
    assert signature['subset'] == 1 and signature['ssl'] == 'barlowtwins'
    assert signature['protocol']['ssl']['subset_epochs'] == 10000
    assert signature['protocol']['ssl']['checkpoint_every_epochs'] == 500
    assert signature['protocol']['evaluation']['epochs'] == 100
    assert signature['ssl_seeds'] == [0, 1] and signature['probe_seeds'] == [0, 1, 2]
    source = RawDataset(data_root, job['source'], True)
    assert source.fingerprint == signature['source_fingerprint'] == contract['selection']['source_fingerprint']
    signature.update(synthetic=metadata, experiment=HERE.name, driver_sha256=file_hash(__file__),
                     replacement_code_hashes=job['code_hashes'])
    identity = dict(signature=digest_json(signature), seed=job['seed'], selection_method='ours')
    return synthetic, dict(signature=signature, identity=identity)


def main():
    manifest = Path(sys.argv[1]).resolve()
    assert manifest.is_relative_to(HERE / 'batches')
    job, data_root = read(manifest), Path(sys.argv[2]).resolve()
    assert os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_JOB_ACCOUNT') == job['account']
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    synthetic, config = prepare(job, data_root)
    if len(sys.argv) == 4 and sys.argv[3] == '--check':
        assert len(synthetic) == 500 and synthetic[0][0].size == (32, 32)
        from common import build_encoder
        encoder, dimension = build_encoder('ResNet18', 32)
        encoder.eval()
        with torch.no_grad():
            features = encoder(torch.zeros(2, 3, 32, 32))
        assert features.shape == (2, dimension) and torch.isfinite(features).all()
        print('CHECK PASSED:', job['source'], 'new 500-image set, unchanged 10k Barlow/100-epoch probe recipe')
        return
    assert torch.cuda.is_available(), 'CUDA required'
    source = job['source']
    group = HERE / 'results' / source / 'ResNet18BN/pt1/barlowtwins'
    folder = group / ('ssl_seed_%03d' % job['seed'])
    protocol, identity = config['signature']['protocol'], config['identity']

    def progress(*values, **kwargs):
        if len(values) == 1 and isinstance(values[0], str) and values[0].startswith('SSL '):
            prefix, line = values[0].split(': ', 1)
            stats = json.loads(line)
            if stats['epoch'] not in (1, 10000) and stats['epoch'] % 200:
                return
            stats.pop('projector_std', None)
            stats.pop('seconds', None)
            values = (prefix + ': ' + json.dumps(stats),)
        print(*values, **kwargs)

    train_ssl.print = progress
    targets = (source, 'CIFAR100' if source == 'CIFAR10' else 'CIFAR10', 'TinyImageNet')
    with exclusive_run(folder):
        cfg = folder / 'config.json'
        if cfg.exists() and read(cfg) != config:
            raise ValueError('Inputs changed; refusing to mix evaluation results')
        atomic_json(cfg, config)
        checkpoint, saved = train_ssl.train(synthetic, protocol, source, 'ResNet18', 'barlowtwins',
            job['seed'], folder, identity, workers=0, device='cuda')
        assert saved['complete'] and saved['epochs'] == 10000
        marker = dict(complete=True, identity=identity, epochs=10000,
            encoder_sha256=saved['encoder_sha256'], checkpoint_file_sha256=file_hash(checkpoint))
        marker_path = folder / 'ssl_complete.json'
        if marker_path.exists() and read(marker_path) != marker:
            raise ValueError('Completed encoder changed')
        atomic_json(marker_path, marker)
        for target in targets:
            train_set, test_set = RawDataset(data_root, target, True), RawDataset(data_root, target, False)
            for labels in (1, 5):
                for probe_seed in (0, 1, 2):
                    output = folder / 'evaluations' / ('%s_lbl%d_probe%d.json' % (target, labels, probe_seed))
                    result = linear_probe(checkpoint, identity, protocol, source, target, train_set, test_set,
                        labels, probe_seed, output, workers=0, device='cuda')
                    assert result['complete'] and result['encoder_unchanged']
                    assert result['encoder_sha256'] == marker['encoder_sha256']
                    with (group / '.summary.lock').open('a') as lock:
                        fcntl.flock(lock, fcntl.LOCK_EX)
                        ours.summarize(group, identity['signature'], source, 'ResNet18BN', 1, 'barlowtwins', targets)
            del train_set, test_set
        assert file_hash(checkpoint) == marker['checkpoint_file_sha256']
        atomic_json(folder / 'complete.json', dict(marker, targets=targets, probe_seeds=[0, 1, 2]))


if __name__ == '__main__':
    main()
