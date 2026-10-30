"""Two comparable t-SNE figures using the exact saved Random and IIC selections."""
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import ROOT, atomic_json, file_hash, load_encoder, state_hash
from data import RawDataset

PREVIEWS = {
    'CIFAR10': HERE / 'previews/CIFAR10_epoch080_pt0.2_mksx5ygw',
    'CIFAR100': HERE / 'previews/CIFAR100_epoch080_pt0.2_xgmlvv8b',
}


@torch.inference_mode()
def extract(output):
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    for name, preview in PREVIEWS.items():
        selection = json.loads((preview / 'indices.json').read_text())
        config = json.loads((Path(selection['run']) / 'config.json').read_text())
        for source in ('networks.py', 'benchmarks/downstream_data.py', 'Evaluate_benchamrk/common.py', 'Evaluate_benchamrk/data.py'):
            assert file_hash(ROOT / source) == config['code_hashes'][source], 'Inference code changed: ' + source
        teacher = Path(config['teacher']['path'])
        assert config['teacher']['file_sha256'] == selection['teacher_sha256'] == file_hash(teacher)
        encoder, dimension, saved = load_encoder(teacher, device='cuda')
        assert (saved['source'], saved['model'], saved['method'], saved['seed']) == (name, 'ResNet18', 'simclr', 0)
        assert saved['encoder_sha256'] == config['teacher']['encoder_sha256']
        dataset = RawDataset(config['data_root'], name, True)
        assert len(dataset) == 50000 and dataset.fingerprint == selection['train_fingerprint']
        pixels = dataset.base.data
        mean, std = [torch.tensor(value, device='cuda')[None, :, None, None]
                     for value in config['protocol']['normalization'][name]]
        features = np.empty((len(dataset), dimension), dtype=np.float32)
        for start in range(0, len(dataset), 128):
            images = torch.from_numpy(pixels[start:start + 128].copy()).permute(0, 3, 1, 2).cuda().float().div_(255)
            features[start:start + len(images)] = F.normalize(encoder((images - mean) / std), dim=1).cpu().numpy()
        assert np.isfinite(features).all() and np.allclose(np.linalg.norm(features, axis=1), 1, atol=1e-5)
        assert not encoder.training and all(not parameter.requires_grad and parameter.grad is None for parameter in encoder.parameters())
        assert state_hash(encoder) == saved['encoder_sha256'] and file_hash(teacher) == selection['teacher_sha256']
        folder = output / name
        folder.mkdir()
        np.save(folder / 'features.npy', features, allow_pickle=False)
        np.save(folder / 'labels.npy', dataset.targets, allow_pickle=False)
        atomic_json(folder / 'selection.json', dict(selection, feature_space='L2-normalized frozen full-data SimCLR encoder features',
                    preview_indices_sha256=file_hash(preview / 'indices.json'), features_sha256=file_hash(folder / 'features.npy'),
                    teacher_unchanged=True))
        print('FEATURES SAVED: ' + str(folder), flush=True)
        del encoder, dataset, features
        torch.cuda.empty_cache()


def plot(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap, hsv_to_rgb
    from matplotlib.lines import Line2D
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    workers = int(os.environ.get('SLURM_CPUS_PER_TASK', '1'))
    embeddings, labels, selections = {}, {}, {}
    for name in PREVIEWS:
        folder = output / name
        selection = json.loads((folder / 'selection.json').read_text())
        assert file_hash(folder / 'features.npy') == selection['features_sha256']
        features = np.load(folder / 'features.npy', allow_pickle=False)
        labels[name] = np.load(folder / 'labels.npy', allow_pickle=False)
        selections[name] = selection
        assert features.shape == (50000, 512) and labels[name].shape == (50000,)
        for key in ('random_indices', 'iic_indices'):
            indices = selection[key]
            assert len(indices) == len(set(indices)) == 100 and min(indices) >= 0 and max(indices) < len(features)
        coordinates = folder / 'tsne.npy'
        if coordinates.exists():
            embeddings[name] = np.load(coordinates, allow_pickle=False)
        else:
            print(name + ': PCA to 50 dimensions, then one shared t-SNE for all 50,000 images', flush=True)
            reduced = PCA(n_components=50, svd_solver='randomized', random_state=0).fit_transform(features)
            model = TSNE(n_components=2, perplexity=30, learning_rate='auto', init='pca',
                         max_iter=1000, random_state=0, method='barnes_hut', angle=0.5, n_jobs=workers, verbose=1)
            embeddings[name] = model.fit_transform(reduced)
            np.save(coordinates, embeddings[name], allow_pickle=False)
            atomic_json(folder / 'tsne.json', dict(seed=0, perplexity=30, max_iter=1000, pca_dimensions=50,
                        kl_divergence=float(model.kl_divergence_), points=len(features),
                        features_sha256=selection['features_sha256'], labels_used_for_embedding=False))
        assert embeddings[name].shape == (50000, 2) and np.isfinite(embeddings[name]).all()
        del features

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.titlesize': 10, 'savefig.facecolor': 'white'})
    for method, key, filename in [('Random selection', 'random_indices', 'tsne_random.png'),
                                   ('IIC initialization', 'iic_indices', 'tsne_iic.png')]:
        figure, axes = plt.subplots(1, 2, figsize=(7.0, 3.2), constrained_layout=True)
        for axis, name in zip(axes, PREVIEWS):
            points = embeddings[name]
            classes = int(labels[name].max()) + 1
            palette = hsv_to_rgb(np.column_stack((np.linspace(0.12, 0.80, classes),
                                                  np.full(classes, 0.6), np.full(classes, 0.8))))
            assert not np.any(np.all(np.isclose(palette, (1, 0, 0)), axis=1))
            cmap = ListedColormap(palette)
            norm = BoundaryNorm(np.arange(classes + 1) - 0.5, classes)
            axis.scatter(points[:, 0], points[:, 1], c=labels[name], cmap=cmap, norm=norm,
                         s=1.2, alpha=0.30, linewidths=0, rasterized=True)
            chosen = np.asarray(selections[name][key])
            axis.scatter(points[chosen, 0], points[chosen, 1], c='#ff0000', s=32, alpha=1,
                         edgecolors='#850000', linewidths=0.5, zorder=10)
            extent = np.ptp(points, axis=0)
            axis.set_xlim(points[:, 0].min() - extent[0] * 0.03, points[:, 0].max() + extent[0] * 0.03)
            axis.set_ylim(points[:, 1].min() - extent[1] * 0.03, points[:, 1].max() + extent[1] * 0.03)
            axis.set_aspect('equal', adjustable='box')
            axis.set_title(name)
            axis.set_axis_off()
            colorbar = figure.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axis,
                                      fraction=0.035, pad=0.01, ticks=np.arange(classes) if classes == 10 else [0, 25, 50, 75, 99])
            colorbar.ax.tick_params(labelsize=7, length=2)
            colorbar.set_label('True class', fontsize=8)
            colorbar.outline.set_linewidth(0.4)
        figure.suptitle(method + ' | 0.2% selected (100 images per dataset)', fontsize=11)
        figure.legend(handles=[Line2D([], [], marker='o', color='none', markerfacecolor='#ff0000',
                                     markeredgecolor='#850000', markersize=6, label='Selected')],
                      loc='outside lower center', frameon=False)
        figure.savefig(output / filename, dpi=350, bbox_inches='tight')
        plt.close(figure)
        print('SAVED: ' + str(output / filename), flush=True)
    assert (output / 'tsne_random.png').stat().st_size > 1000 and (output / 'tsne_iic.png').stat().st_size > 1000


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['extract', 'plot'])
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', '1')))
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.stage == 'extract':
        extract(args.output.resolve())
    else:
        plot(args.output.resolve())
