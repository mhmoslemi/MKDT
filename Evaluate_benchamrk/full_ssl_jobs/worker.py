"""One full-data SSL seed; no probing, selection, or synthetic generation.

Lives below full_ssl_jobs/ to preserve existing shared-code fingerprints.
"""
import csv
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent
sys.path.insert(0, str(BENCH))

import torch
import torchvision

from common import atomic_json, code_hashes, digest_json, exclusive_run, file_hash, load_encoder
from data import RawDataset, SOURCE_COUNTS
from run import validate_protocol
from train_ssl import train


def read(path):
    return json.loads(path.read_text())


def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Run through the supplied Slurm launcher, never on a login node')
    if os.environ.get('SLURM_CLUSTER_NAME', os.environ.get('CC_CLUSTER')) != 'vulcan':
        raise RuntimeError('Full-data jobs belong on Vulcan')
    task, data_root, output_root = sys.argv[1:]
    with (HERE / 'tasks.tsv').open() as stream:
        matches = [row for row in csv.DictReader(stream, delimiter='\t') if row['task'] == task]
    if len(matches) != 1:
        raise ValueError('Unknown or duplicate task')
    row = matches[0]
    source, model, method = row['source'], row['model'], row['ssl']
    seed = int(row['seed'])
    if (source not in SOURCE_COUNTS or model not in ('ConvNet', 'VGG11', 'ResNet18')
            or method not in ('simclr', 'barlowtwins') or seed not in (0, 1, 2)
            or int(row['epochs']) != 1000):
        raise ValueError('Task does not match the full-data three-seed benchmark')
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise RuntimeError('Use one process and one GPU per seed')
    root = Path(output_root).resolve()
    if not root.is_relative_to(BENCH) or root == BENCH:
        raise ValueError('Keep checkpoints in a dedicated directory inside Evaluate_benchamrk')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable: no CPU fallback')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    protocol = read(BENCH / 'protocol.json')
    # Per-launcher overrides only. Never rewrite shared protocol.json while
    # random/k-means/MKDT training or evaluation is using its fingerprint.
    protocol['full_seeds'] = [0, 1, 2]
    protocol['ssl']['checkpoint_every_epochs'] = 50
    validate_protocol(protocol)
    if protocol['ssl']['full_epochs'] != 1000 or protocol['ssl']['batch_size'] != 256:
        raise ValueError('Expected the agreed 1000-epoch, batch-256 full-data recipe')
    base = RawDataset(data_root, source, True)
    if len(base) != SOURCE_COUNTS[source]:
        raise ValueError('Unexpected source training split size')
    signature = {'protocol': protocol, 'code_hashes': code_hashes(),
        'ssl_worker_sha256': file_hash(Path(__file__)), 'ssl_worker': 'full_ssl_jobs/worker.py',
        'source': source, 'source_fingerprint': base.fingerprint, 'model': model,
        'selection_method': 'full', 'ssl': method, 'subset': 100,
        'input_scope': 'entire_source_training_split', 'image_count': len(base)}
    identity = {'signature': digest_json(signature), 'seed': seed,
                'selection': {'scope': 'full'}, 'selection_method': 'full'}
    folder = root / source / model / 'full' / 'pt100' / method / f'seed_{seed:03d}'
    with exclusive_run(folder):
        config = {'identity': identity, 'signature': signature}
        config_path = folder / 'config.json'
        if config_path.exists() and read(config_path) != config:
            raise ValueError('Existing full-data configuration differs; do not mix runs')
        atomic_json(config_path, config)
        checkpoint, marker_path = folder / 'encoder.pt', folder / 'ssl_complete.json'
        if marker_path.exists() and not checkpoint.exists():
            raise FileNotFoundError('Completion marker exists but its final encoder is missing')
        already_saved = checkpoint.exists()
        started = time.monotonic()
        if already_saved:
            checked, _, saved = load_encoder(checkpoint, identity, 'cpu')
            metadata = {key: value for key, value in saved.items() if key != 'encoder'}
            del checked, saved
        else:
            atomic_json(folder / 'environment.json', {
                'python': sys.version, 'torch': str(torch.__version__),
                'torchvision': str(torchvision.__version__), 'gpu': torch.cuda.get_device_name(),
                'job_id': os.environ['SLURM_JOB_ID'], 'cluster': 'vulcan',
                'account': os.environ.get('SLURM_JOB_ACCOUNT'), 'workers': 0, 'threads': 1})
            print(f'FULL SSL {source}/{model}/{method} seed={seed}: '
                  f'{len(base)} training images, 1000 epochs, checkpoint every 50', flush=True)
            checkpoint, metadata = train(base, protocol, source, model, method,
                                         seed, folder, identity, 0, 'cuda', full=True)
        if (not metadata['complete'] or metadata['identity'] != identity
                or metadata['epochs'] != 1000 or metadata['source'] != source
                or metadata['model'] != model or metadata['method'] != method
                or metadata['seed'] != seed or metadata['effective_batch'] != 256
                or metadata['steps'] != 1000 * (len(base) // 256)):
            raise ValueError('Final full-data encoder metadata mismatch')
        checksum = file_hash(checkpoint)
        if marker_path.exists():
            marker = read(marker_path)
            if (not marker.get('complete') or marker['identity'] != identity
                    or marker['epochs'] != 1000 or marker['encoder_sha256'] != metadata['encoder_sha256']
                    or marker['checkpoint_file_sha256'] != checksum):
                raise ValueError('Invalid final completion marker')
        else:
            atomic_json(marker_path, {**metadata, 'checkpoint_file_sha256': checksum,
                'recovered_existing_encoder': already_saved,
                'training_wall_seconds': None if already_saved else time.monotonic() - started})
        # train_ssl atomically replaces ONE training.pt at epochs 50,100,...,1000.
        # Retain that resumable checkpoint and the final encoder.pt; delete nothing.
        print('VERIFIED FULL-DATA ENCODER', checkpoint, flush=True)


if __name__ == '__main__':
    main()
