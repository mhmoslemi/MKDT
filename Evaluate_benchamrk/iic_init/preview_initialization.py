"""Random versus saved-IIC initialization previews; no training."""
import argparse
import json
from pathlib import Path
import sys
import tempfile
import time

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import ROOT, atomic_json, atomic_save, digest_json, file_hash, load_encoder, state_hash
from data import RawDataset, probe_transform


def balanced_representatives(assignments, confidence, clusters, count):
    # Cycle through nonempty learned clusters, taking their most confident unused image.
    order = np.argsort(-confidence, kind='stable')
    groups = [order[assignments[order] == index] for index in range(clusters)]
    chosen, depth = [], 0
    while len(chosen) < count:
        previous = len(chosen)
        for group in groups:
            if len(group) > depth:
                chosen.append(int(group[depth]))
                if len(chosen) == count:
                    break
        assert len(chosen) > previous, 'Not enough distinct training images'
        depth += 1
    chosen.sort(key=lambda index: int(assignments[index]))
    assert len(chosen) == count == len(set(chosen))
    return chosen


def grid(pixels, indices, title):
    tile, padding = 96, 4
    image = Image.new('RGB', (10 * (tile + padding) + padding, 10 * (tile + padding) + 40), 'white')
    ImageDraw.Draw(image).text((padding, 10), title, fill='black')
    for position, index in enumerate(indices):
        thumbnail = Image.fromarray(pixels[index]).resize((tile, tile), Image.Resampling.NEAREST)
        image.paste(thumbnail, (padding + position % 10 * (tile + padding), 36 + position // 10 * (tile + padding)))
    return image


@torch.inference_mode()
def preview(run, device):
    config = json.loads((run / 'config.json').read_text())
    dataset_name = config['dataset']
    assert dataset_name in ('CIFAR10', 'CIFAR100')
    checkpoint = run / 'training.pt'
    checkpoint_sha = file_hash(checkpoint)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    assert saved['config_sha256'] == digest_json(config)
    epoch = saved['epoch']
    assert 0 < epoch <= config['epochs']
    rows = [json.loads(line) for line in (run / 'training.jsonl').read_text().splitlines() if line]
    stats = next(row for row in reversed(rows) if row['epoch'] == epoch)
    for name in ('networks.py', 'benchmarks/downstream_data.py', 'Evaluate_benchamrk/common.py', 'Evaluate_benchamrk/data.py'):
        assert file_hash(ROOT / name) == config['code_hashes'][name], 'Inference code changed: ' + name
    teacher = Path(config['teacher']['path'])
    assert file_hash(teacher) == config['teacher']['file_sha256']
    assert file_hash(teacher.parent / 'ssl_complete.json') == config['teacher']['marker_sha256']
    encoder, dimension, teacher_saved = load_encoder(teacher, device=device)
    assert (teacher_saved['source'], teacher_saved['model'], teacher_saved['method'], teacher_saved['seed']) == (dataset_name, 'ResNet18', 'simclr', 0)
    assert teacher_saved['encoder_sha256'] == config['teacher']['encoder_sha256']
    heads = nn.ModuleDict({name: nn.ModuleList([nn.Linear(dimension, count) for _ in range(config['subheads'])])
                           for name, count in config['cluster_counts'].items()})
    heads.load_state_dict(saved['heads'], strict=True)
    heads.to(device).eval().requires_grad_(False)
    # Reuse the saved training MI to choose heads; avoid extra augmented inference passes for a preview.
    selected = {name: int(np.argmax(stats['subhead_mi'][name])) for name in heads}
    dataset = RawDataset(config['data_root'], dataset_name, True)
    assert len(dataset) == 50000 and dataset.fingerprint == config['source_fingerprint']
    pixels = dataset.base.data
    mean, std = config['protocol']['normalization'][dataset_name]
    mean, std = torch.tensor(mean)[None, :, None, None], torch.tensor(std)[None, :, None, None]
    assert config['protocol']['input_size'][dataset_name] == 32
    transform = probe_transform(config['protocol'], dataset_name, False)
    assert np.array_equal(np.asarray(dataset[0][0]), pixels[0])
    assert torch.allclose(transform(dataset[0][0]), (torch.from_numpy(pixels[0].copy()).permute(2, 0, 1).float() / 255 - mean[0]) / std[0])
    mean, std = mean.to(device), std.to(device)
    assignment_parts, confidence_parts = {name: [] for name in heads}, {name: [] for name in heads}
    print(f'{dataset_name}: saved epoch {epoch}; {device} inference on {len(dataset)} training images; heads {selected}', flush=True)
    started = time.monotonic()
    for start in range(0, len(dataset), 128):
        images = torch.from_numpy(pixels[start:start + 128].copy()).permute(0, 3, 1, 2).float().div_(255)
        features = encoder((images.to(device) - mean) / std)
        for name, group in heads.items():
            probabilities = group[selected[name]](features).softmax(1)
            assert torch.isfinite(probabilities).all()
            confidence, assignment = probabilities.max(1)
            assignment_parts[name].append(assignment.cpu())
            confidence_parts[name].append(confidence.cpu())
        if start // 128 % 40 == 0 or start + 128 >= len(dataset):
            print(f'{dataset_name}: {min(start + 128, len(dataset))}/{len(dataset)}, {time.monotonic() - started:.1f}s', flush=True)
    clusters = {name: dict(assignments=torch.cat(assignment_parts[name]), confidence=torch.cat(confidence_parts[name]),
                          subhead=selected[name], cluster_count=config['cluster_counts'][name]) for name in heads}
    count = int(len(dataset) * 0.2 / 100)
    assert count == 100
    random_indices = np.random.default_rng(config['seed'] + 1000).permutation(len(dataset))[:count].tolist()
    main = clusters['main']
    iic_indices = balanced_representatives(main['assignments'].numpy(), main['confidence'].numpy(), main['cluster_count'], count)
    assert not encoder.training and all(not parameter.requires_grad and parameter.grad is None for parameter in encoder.parameters())
    assert state_hash(encoder) == teacher_saved['encoder_sha256']
    assert file_hash(teacher) == config['teacher']['file_sha256'] and file_hash(checkpoint) == checkpoint_sha
    output = Path(tempfile.mkdtemp(prefix=f'{dataset_name}_epoch{epoch:03d}_pt0.2_', dir=HERE / 'previews'))
    random_grid = grid(pixels, random_indices, f'{dataset_name} | 0.2% (100 images) | Random, seed {config["seed"]}')
    iic_grid = grid(pixels, iic_indices, f'{dataset_name} | 0.2% (100 images) | IIC epoch {epoch}, balanced main-cluster representatives')
    random_grid.save(output / 'random.png')
    iic_grid.save(output / 'iic.png')
    combined = Image.new('RGB', (random_grid.width * 2 + 16, random_grid.height), 'white')
    combined.paste(random_grid, (0, 0))
    combined.paste(iic_grid, (random_grid.width + 16, 0))
    combined.save(output / 'random_vs_iic.png')
    for name, value in clusters.items():
        value['counts'] = torch.bincount(value['assignments'], minlength=value['cluster_count'])
    metadata = dict(dataset=dataset_name, percentage=0.2, epoch=epoch, count=count, seed=config['seed'],
                    run=str(run), checkpoint_sha256=checkpoint_sha, config_sha256=saved['config_sha256'],
                    teacher_sha256=config['teacher']['file_sha256'], train_fingerprint=dataset.fingerprint,
                    device=device, preview_implementation_sha256=file_hash(Path(__file__)),
                    selected_subheads=selected, subhead_selection='highest saved-epoch minibatch training MI; not final global-MI selection',
                    image_selection='round-robin highest-confidence images in nonempty main clusters, then grouped by cluster',
                    random_indices=random_indices, iic_indices=iic_indices, teacher_unchanged=True,
                    nonempty_main_clusters=int((main['counts'] > 0).sum()),
                    selected_cluster_counts=torch.bincount(main['assignments'][iic_indices], minlength=main['cluster_count']).tolist())
    atomic_save(output / 'clusters.pt', dict(kind='iic_initialization_preview_clusters', metadata=metadata, clusters=clusters))
    atomic_json(output / 'indices.json', metadata)
    assert Image.open(output / 'random.png').size == random_grid.size
    assert Image.open(output / 'iic.png').size == iic_grid.size
    print('SAVED: ' + str(output), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('runs', type=Path, nargs='*', default=[
        HERE / 'runs/CIFAR10_ResNet18_e100_20261030T173733Z_ueNMxo_lu_l40s',
        HERE / 'runs/CIFAR100_ResNet18_e100_20261030T174140Z_OcY02E_lu_l40s',
    ])
    args = parser.parse_args()
    if args.device == 'cuda':
        assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    assert balanced_representatives(np.array([0, 0, 2, 2]), np.array([.9, .7, .6, .8]), 3, 3) == [0, 1, 3]
    (HERE / 'previews').mkdir(exist_ok=True)
    for run in args.runs:
        preview(run.resolve(), args.device)
