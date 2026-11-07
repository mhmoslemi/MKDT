"""Authors' CLIP grouping and unmodified SAS selector, with approved CIFAR10 settings."""
from functools import partial
import argparse
import json
import os
from pathlib import Path
import random
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE.parent))
import numpy as np
import torch
from torchvision import transforms as T
from common import atomic_json, atomic_save, file_hash, seed_all
from data import RawDataset, ImageView, loader
from train_proxy import build
sys.path.insert(0, str(HERE / 'upstream/sas/sas-pip'))
sys.path.insert(0, str(HERE / 'upstream/clip'))
import clip
from sas.approx_latent_classes import clip_approx
from sas.subset_dataset import SASSubsetDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--groups-only', action='store_true')
    args = parser.parse_args()
    assert os.environ.get('SLURM_JOB_ACCOUNT') == 'aip-yiweilu' and torch.cuda.device_count() == 1
    torch.set_num_threads(1)
    seed_all(0)
    manifest = json.loads((HERE / 'manifest.json').read_text())
    for relative, checksum in manifest['code_hashes'].items():
        assert file_hash(ROOT / relative) == checksum, 'Code changed: ' + relative
    folder = HERE / 'selections/TinyImageNet'
    folder.mkdir(parents=True, exist_ok=True)
    base = RawDataset(manifest['data_root'], 'TinyImageNet', True)
    labeled = random.sample(range(len(base)), int(0.01 * len(base)))
    clip.load = partial(clip.load, download_root=str(HERE / 'assets/clip'))
    groups_path = folder / 'groups.pt'
    if groups_path.is_file():
        saved = torch.load(groups_path, map_location='cpu', weights_only=True)
        assert saved['source_fingerprint'] == base.fingerprint
        partition, labeled = saved['partition'], saved['labeled_indices']
    else:
        partition = clip_approx(ImageView(base, T.ToTensor()), labeled,
                               [int(base.targets[i]) for i in labeled], 200, 'cuda', verbose=True)
        partition = {int(k): [int(i) for i in values] for k, values in partition.items()}
        assert sorted(i for values in partition.values() for i in values) == list(range(len(base)))
        atomic_save(groups_path, dict(source_fingerprint=base.fingerprint,
                    partition=partition, labeled_indices=labeled))
    print(f'CLIP grouping ready: {len(partition)} occupied groups, {len(labeled)} training labels used', flush=True)
    if args.groups_only:
        return
    marker = json.loads((HERE / 'proxy/TinyImageNet/complete.json').read_text())
    proxy_path = HERE / 'proxy/TinyImageNet/training.pt'
    assert marker['complete'] and marker['epochs'] == 400
    assert marker['source_fingerprint'] == base.fingerprint and file_hash(proxy_path) == marker['checkpoint_sha256']
    cache = folder / 'features.pt'
    if cache.is_file():
        saved = torch.load(cache, map_location='cpu', weights_only=True)
        assert saved['source_fingerprint'] == base.fingerprint and saved['proxy_sha256'] == marker['checkpoint_sha256']
        features, partition, labeled = saved['features'].cuda(), saved['partition'], saved['labeled_indices']
    else:
        net, critic = build()
        saved = torch.load(proxy_path, map_location='cpu', weights_only=True)
        net.load_state_dict(saved['encoder'])
        critic.load_state_dict(saved['critic'])
        net, critic = net.cuda().eval(), critic.cuda().eval()
        chunks = []
        with torch.no_grad():
            for images, _ in loader(ImageView(base, T.ToTensor()), 256, 4, 0):
                chunks.append(critic.project(net(images.cuda())).cpu())
        features = torch.cat(chunks).cuda()
        assert features.shape == (len(base), 128) and torch.isfinite(features).all()
        atomic_save(cache, dict(source_fingerprint=base.fingerprint,
                    proxy_sha256=marker['checkpoint_sha256'], features=features.cpu(),
                    partition=partition, labeled_indices=labeled))
        del net, critic, saved, chunks
        torch.cuda.empty_cache()
    similarities = {group: SASSubsetDataset.pairwise_distance(features[indices], features[indices])
                    for group, indices in partition.items()}
    for percentage in (0.2, 1, 2):
        subset = SASSubsetDataset(ImageView(base, T.ToTensor()), percentage / 100, 200,
            'cuda', partition, augmentation_distance=similarities, threshold=0.5, verbose=True)
        indices = [int(i) for i in subset.subset_indices]
        expected = sum(int(percentage / 100 * len(group)) for group in partition.values())
        assert len(indices) == len(set(indices)) == expected and len(indices) > 1
        assert all(0 <= i < len(base) for i in indices)
        result = dict(complete=True, source='TinyImageNet', percentage=percentage,
            requested_count=int(len(base) * percentage / 100), actual_count=len(indices),
            indices=indices, source_fingerprint=base.fingerprint,
            origin='official_SAS_code_CIFAR10_recipe_extended_to_TinyImageNet',
            recipe=manifest['tiny_recipe'], proxy_sha256=marker['checkpoint_sha256'],
            labeled_indices=labeled, grouping_labels_used=len(labeled),
            budget_rounding='unaltered_per_group_floor', selection_transform='ToTensor_only_as_official_notebook')
        atomic_json(folder / f'pt{percentage:g}.json', result)
        print(f'Saved Tiny SAS {percentage}%: {len(indices)} images (requested {result["requested_count"]}); official floor rounding retained', flush=True)
    atomic_json(folder / 'complete.json', dict(complete=True))


if __name__ == '__main__':
    main()
