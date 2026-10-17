"""Six independent SSL initializations on saved MKDT images, not distillation.

Use the same train_ssl.train and SyntheticDataset as the checked benchmark.
This subdirectory keeps the existing top-level SSL code-hash inventory intact.
"""
import csv
import gc
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
from data import RawDataset, SOURCE_COUNTS, SyntheticDataset
from run import validate_protocol
from train_ssl import train


def read(path):
    return json.loads(path.read_text())


def main():
    if 'SLURM_JOB_ID' not in os.environ:
        raise RuntimeError('Run through the Slurm launcher, never on a login node')
    if os.environ.get('SLURM_CLUSTER_NAME', os.environ.get('CC_CLUSTER')) != 'killarney':
        raise RuntimeError('This launcher is for Killarney')
    task, data_root, synthetic_root, output_root = sys.argv[1:]
    with (HERE / 'tasks.tsv').open() as stream:
        matches = [row for row in csv.DictReader(stream, delimiter='\t') if row['task'] == task]
    if len(matches) != 1:
        raise ValueError('Unknown/duplicate task')
    row = matches[0]
    source, model, method = row['source'], row['model'], row['ssl']
    percentage, seed_start = int(row['subset']), int(row['seed_start'])
    seed_count, epochs = int(row['seed_count']), int(row['epochs'])
    generator = 'VGG11' if model == 'VGG11' else 'ConvNet'
    budget = ({1: 5000, 2: 4000} if source == 'TinyImageNet' else {1: 8000, 2: 6000})
    if (source not in SOURCE_COUNTS or model not in ('ConvNet', 'VGG11', 'ResNet18')
            or method not in ('simclr', 'barlowtwins') or percentage not in budget
            or epochs != budget[percentage] or generator != row['generator']
            or seed_count not in (1, 2) or not 0 <= seed_start < 6 or seed_start + seed_count > 6
            or (seed_count == 2 and seed_start % 2)):
        raise ValueError('Task does not match the six-seed MKDT benchmark')
    root = Path(output_root).resolve()
    if not root.is_relative_to(BENCH) or root == BENCH:
        raise ValueError('Keep checkpoints in their own directory inside Evaluate_benchamrk')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable: no CPU fallback')
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise RuntimeError('Use one process and one GPU per job')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    protocol = read(BENCH / 'protocol.json')
    protocol['ssl']['subset_epochs'] = epochs
    protocol['ssl']['checkpoint_every_epochs'] = 500
    protocol['subset_seeds'] = list(range(6))
    validate_protocol(protocol)
    if protocol['ssl']['batch_size'] != 256:
        raise ValueError('The runtime estimates require the existing batch-256 recipe')

    token, iteration = ('Tiny', 10000) if source == 'TinyImageNet' else (source, 5000)
    expected_image = f'{generator}/{source}/{token}_{percentage}pct/images_{iteration}.pt'
    if row['images_relative'] != expected_image:
        raise ValueError('Synthetic artifact path does not match its generator/source/size')
    image_path = (Path(synthetic_root) / expected_image).resolve()
    entry = {'path': str(image_path), 'key': None, 'space': 'repository_normalized',
             'pixel_policy': 'clamp_to_unit_rgb', 'generator': generator}
    # Validates count, finite float NCHW data and native 32/64px resolution;
    # inverts GENERATION normalization, clamps explicitly, then supplies PIL RGB.
    # train_ssl subsequently applies the SAME fresh-view augmentation and SSL
    # normalization used for real random/kmeans subsets. No legacy 64->32 resize.
    selected = SyntheticDataset(entry, source, percentage, protocol['input_size'][source])
    print('SYNTHETIC INPUT', json.dumps(selected.metadata, sort_keys=True), flush=True)
    print('Images only: labels_*.pt and synlr_*.pt are NOT used.', flush=True)
    # Read the real TRAIN split only to bind later evaluations to the same source
    # dataset. It is NOT passed into the SSL trainer and does not supply labels.
    base = RawDataset(data_root, source, True)
    if len(base) != SOURCE_COUNTS[source]:
        raise ValueError('Unexpected source training split size')
    source_fingerprint = base.fingerprint
    del base
    signature = {'protocol': protocol, 'code_hashes': code_hashes(),
        'ssl_worker_sha256': file_hash(Path(__file__)), 'ssl_worker': 'mkdt_ssl_jobs/worker.py',
        'source': source, 'source_fingerprint': source_fingerprint, 'model': model,
        'selection_method': 'mkdt', 'ssl': method, 'subset': percentage,
        'generator': generator, 'synthetic': selected.metadata,
        'input_scope': 'saved_synthetic_images_only', 'seed_scope': 'ssl_initialization_not_distillation'}
    group = root / source / model / 'mkdt' / f'pt{percentage}' / method
    print(f'MKDT SSL: {source}/{model}/{percentage}%/{method}; generator={generator}; '
          f'seeds={seed_start}..{seed_start + seed_count - 1}; epochs/seed={epochs}', flush=True)
    for seed in range(seed_start, seed_start + seed_count):
        folder = group / f'seed_{seed:03d}'
        with exclusive_run(folder):
            identity = {'signature': digest_json(signature), 'seed': seed,
                        'selection': selected.metadata, 'selection_method': 'mkdt'}
            config = {'identity': identity, 'signature': signature}
            if (folder / 'config.json').exists() and read(folder / 'config.json') != config:
                raise ValueError(f'Configuration/synthetic images changed: {folder}')
            atomic_json(folder / 'config.json', config)
            checkpoint = folder / 'encoder.pt'
            started = time.monotonic()
            already_saved = checkpoint.exists()
            if already_saved:
                checked, _, saved = load_encoder(checkpoint, identity, 'cpu')
                metadata = {key: value for key, value in saved.items() if key != 'encoder'}
                del checked, saved
            else:
                atomic_json(folder / 'environment.json', {
                    'python': sys.version, 'torch': str(torch.__version__),
                    'torchvision': str(torchvision.__version__), 'gpu': torch.cuda.get_device_name(),
                    'job_id': os.environ['SLURM_JOB_ID'], 'cluster': 'killarney',
                    'account': os.environ.get('SLURM_JOB_ACCOUNT'), 'workers': 0, 'threads': 1})
                checkpoint, metadata = train(selected, protocol, source, model, method,
                                              seed, folder, identity, 0, 'cuda', full=False)
            if (not metadata['complete'] or metadata['identity'] != identity or metadata['epochs'] != epochs
                    or metadata['source'] != source or metadata['model'] != model
                    or metadata['method'] != method or metadata['seed'] != seed):
                raise ValueError(f'Final encoder metadata mismatch: {checkpoint}')
            checksum = file_hash(checkpoint)
            marker_path = folder / 'ssl_complete.json'
            if marker_path.exists():
                marker = read(marker_path)
                if (not marker.get('complete') or marker['identity'] != identity or marker['epochs'] != epochs
                        or marker['checkpoint_file_sha256'] != checksum
                        or marker['encoder_sha256'] != metadata['encoder_sha256']):
                    raise ValueError(f'Invalid final completion marker: {marker_path}')
            else:
                # Also repairs a crash between final encoder save and marker save.
                atomic_json(marker_path, {**metadata, 'checkpoint_file_sha256': checksum,
                    'synthetic': selected.metadata, 'recovered_existing_encoder': already_saved,
                    'training_wall_seconds': None if already_saved else time.monotonic() - started})
            print('VERIFIED SAVED ENCODER', checkpoint, flush=True)
        gc.collect()
        torch.cuda.empty_cache()
    # Deliberately retain the last rolling training.pt as well as encoder.pt.
    # At most one rolling file per seed; train_ssl atomically replaces it every
    # 500 epochs. No generated image, label, learning-rate or encoder is deleted.
    print(f'COMPLETE task={task}: {seed_count} encoders retained', flush=True)


if __name__ == '__main__':
    main()
