"""Current-table MIPA versus its exact Random initialization, in one teacher space.

extract: GPU features for all six sets; embed: CPU t-SNE and figures for one set;
plot: CPU-only restyling from saved coordinates, without teachers or datasets.
"""
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUTPUT = ROOT / 'OT-SSL-DD/tsne_mipa_random'


def extract():
    import torch
    from torch.nn import functional as F
    sys.path.insert(0, str(HERE.parent))
    from common import atomic_json, file_hash, load_encoder, state_hash
    from data import RawDataset

    assert os.environ.get('SLURM_JOB_ID') and torch.cuda.is_available()
    torch.set_num_threads(int(os.environ['SLURM_CPUS_PER_TASK']))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    manifest = HERE.parent / 'mipa_cvpr_color_crop/manifest.json'
    jobs = json.loads(manifest.read_text())['jobs']
    with torch.inference_mode():
        for name in ('CIFAR10', 'CIFAR100'):
            entries = [job for job in jobs if job['generation_args']['dataset'] == name]
            assert len(entries) == 3
            reference = entries[0]
            teacher = Path(reference['teacher'])
            teacher_hash = file_hash(teacher)
            assert teacher_hash == reference['teacher_sha256']
            encoder, dimension, weights = load_encoder(teacher)
            assert (weights['source'], weights['model'], weights['method'], weights['seed']) == (name, 'ResNet18', 'simclr', 0)
            dataset = RawDataset(reference['data_root'], name, True)
            cache = HERE.parent / 'iic_init/tsne/comparison_iaTEXR' / name
            cached = json.loads((cache / 'selection.json').read_text())
            assert cached['teacher_sha256'] == teacher_hash
            assert cached['train_fingerprint'] == dataset.fingerprint
            assert file_hash(cache / 'features.npy') == cached['features_sha256']
            real = np.load(cache / 'features.npy', allow_pickle=False)
            labels = np.load(cache / 'labels.npy', allow_pickle=False)
            assert real.shape == (50000, dimension) and np.isfinite(real).all()
            assert np.allclose(np.linalg.norm(real, axis=1), 1, atol=1e-5)
            assert np.array_equal(labels, dataset.targets)
            folder = OUTPUT / name
            folder.mkdir(parents=True, exist_ok=True)
            np.save(folder / 'real_features.npy', real, allow_pickle=False)
            np.save(folder / 'real_labels.npy', labels, allow_pickle=False)
            teacher_mean, teacher_std = [torch.tensor(values, device='cuda')[None, :, None, None] for values in weights['normalization']]

            def features(images):
                return torch.cat([F.normalize(encoder((part.cuda() - teacher_mean) / teacher_std), dim=1).cpu()
                                  for part in images.split(128)]).numpy()

            for job in entries:
                args = job['generation_args']
                size = args['percentage']
                count = int(len(dataset) * size / 100)
                assert args['seed'] == 0 and job['teacher_sha256'] == teacher_hash
                checkpoint = Path(job['synthetic_dir']) / f'res_IIC_{name}_ResNet18BN_{size:g}percent_final.pt'
                checkpoint_hash = file_hash(checkpoint)
                saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
                assert saved['complete'] and saved['iteration'] == args['Iteration'] and saved['run_args'] == args
                assert saved['teacher_seed'] == 0 and saved['init'] == 'random'
                indices = np.random.default_rng(1000).permutation(len(dataset))[:count]
                mean, std = [torch.tensor(values)[None, :, None, None] for values in saved['normalization']]
                initial = (saved['initial_data'] * std + mean).clamp(0, 1)
                final = (saved['data'] * std + mean).clamp(0, 1)
                original = torch.from_numpy(dataset.base.data[indices].copy()).permute(0, 3, 1, 2).float() / 255
                assert initial.shape == final.shape == (count, 3, 32, 32) and torch.isfinite(final).all()
                assert torch.allclose(initial, original, atol=1e-6)
                assert np.allclose(features(initial), real[indices], atol=2e-4), 'Cached feature space mismatch'
                synthetic = features(final)
                assert np.isfinite(synthetic).all() and np.allclose(np.linalg.norm(synthetic, axis=1), 1, atol=1e-5)
                out = folder / f'pt{size:g}'
                out.mkdir(exist_ok=True)
                identity = dict(dataset=name, percentage=size, count=count, iteration=saved['iteration'],
                                checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_hash,
                                teacher=str(teacher), teacher_sha256=teacher_hash, teacher_seed=0,
                                real_features_sha256=cached['features_sha256'], train_fingerprint=dataset.fingerprint,
                                generation_args=args, random_seed=0, random_selection_rng_seed=1000,
                                feature_space='L2-normalized frozen full-data SimCLR encoder, last layer, no augmentation',
                                initial_pixels_verified=True, labels_used_for_embedding=False,
                                pixel_policy='Denormalize saved float pixels and clamp to [0, 1]; no enhancement')
                if (out / 'inputs.json').exists():
                    assert json.loads((out / 'inputs.json').read_text()) == identity, 'Refusing to replace a different comparison'
                np.savez(out / 'features.npz', random=real[indices], mipa=synthetic,
                         random_indices=indices, initial_labels=labels[indices])
                atomic_json(out / 'inputs.json', identity)
                assert file_hash(checkpoint) == checkpoint_hash
                print('FEATURES SAVED:', out, flush=True)
            assert state_hash(encoder) == weights['encoder_sha256'] and file_hash(teacher) == teacher_hash
            assert not encoder.training and all(not p.requires_grad and p.grad is None for p in encoder.parameters())
            del encoder, dataset, real
            torch.cuda.empty_cache()


def embed(folder):
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    import sklearn
    sys.path.insert(0, str(HERE.parent))
    from common import atomic_json, file_hash

    identity = json.loads((folder / 'inputs.json').read_text())
    assert file_hash(folder.parent / 'real_features.npy') == identity['real_features_sha256']
    real = np.load(folder.parent / 'real_features.npy', allow_pickle=False)
    labels = np.load(folder.parent / 'real_labels.npy', allow_pickle=False)
    with np.load(folder / 'features.npz', allow_pickle=False) as saved:
        synthetic, random, indices = saved['mipa'], saved['random'], saved['random_indices']
    count = identity['count']
    assert real.shape == (50000, 512) and synthetic.shape == random.shape == (count, 512)
    assert len(np.unique(indices)) == count and np.array_equal(random, real[indices])
    protocol = dict(seed=0, perplexity=30, max_iter=1000, pca_dimensions=50, pca_solver='randomized',
                    learning_rate='auto', init='pca', method='barnes_hut', angle=0.5,
                    sklearn_version=sklearn.__version__, features_sha256=file_hash(folder / 'features.npz'),
                    real_features_sha256=identity['real_features_sha256'], labels_used_for_embedding=False,
                    points=len(real) + count, random_points='Existing real rows; never duplicated in the embedding')
    if (folder / 'projection.json').exists():
        previous = json.loads((folder / 'projection.json').read_text())
        assert previous['protocol'] == protocol, 'Projection inputs changed'
        assert file_hash(folder / 'tsne.npz') == previous['coordinates_sha256']
    else:
        print(f"{identity['dataset']} {identity['percentage']}%: joint t-SNE of all 50,000 real images and {count} prototypes", flush=True)
        features = np.concatenate((real, synthetic))
        reduced = PCA(n_components=50, svd_solver='randomized', random_state=0).fit_transform(features)
        model = TSNE(n_components=2, perplexity=30, learning_rate='auto', init='pca', max_iter=1000,
                     random_state=0, method='barnes_hut', angle=0.5,
                     n_jobs=int(os.environ.get('SLURM_CPUS_PER_TASK', '1')), verbose=1)
        coordinates = model.fit_transform(reduced)
        assert coordinates.shape == (len(real) + count, 2) and np.isfinite(coordinates).all()
        np.savez(folder / 'tsne.npz', real=coordinates[:len(real)], random=coordinates[indices],
                 mipa=coordinates[len(real):], real_labels=labels, random_indices=indices)
        atomic_json(folder / 'projection.json', dict(protocol=protocol, kl_divergence=float(model.kl_divergence_),
                    coordinates_sha256=file_hash(folder / 'tsne.npz')))
    plot(folder)


def plot(folder):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import hsv_to_rgb
    from matplotlib.lines import Line2D

    identity = json.loads((folder / 'inputs.json').read_text())
    with np.load(folder / 'tsne.npz', allow_pickle=False) as saved:
        real, random, mipa, labels, indices = [saved[key] for key in ('real', 'random', 'mipa', 'real_labels', 'random_indices')]
    assert real.shape == (50000, 2) and random.shape == mipa.shape == (identity['count'], 2)
    assert np.array_equal(real[indices], random) and np.isfinite(np.concatenate((real, mipa))).all()
    classes = int(labels.max()) + 1
    palette = hsv_to_rgb(np.column_stack((np.linspace(.12, .80, classes), np.full(classes, .6), np.full(classes, .8))))
    points = np.concatenate((real, mipa))
    lo, hi = points.min(axis=0), points.max(axis=0)
    margin = (hi - lo) * .03
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'pdf.fonttype': 42, 'savefig.facecolor': 'white'})
    for filename, methods in (('comparison', [('Random', random), ('MIPA', mipa)]),
                              ('random', [('Random', random)]), ('mipa', [('MIPA', mipa)])):
        figure, axes = plt.subplots(1, len(methods), figsize=(3.35 * len(methods), 3.2), squeeze=False)
        for axis, (name, selected) in zip(axes.flat, methods):
            axis.scatter(real[:, 0], real[:, 1], c=palette[labels], s=.6, alpha=.3, linewidths=0, rasterized=True)
            axis.scatter(selected[:, 0], selected[:, 1], c='#ff0000', s=9, edgecolors='#850000',
                         linewidths=.25, zorder=10, rasterized=True)
            axis.set(xlim=(lo[0] - margin[0], hi[0] + margin[0]), ylim=(lo[1] - margin[1], hi[1] + margin[1]),
                     aspect='equal', title=name)
            axis.set_axis_off()
        figure.suptitle(f"{identity['dataset']} | {identity['percentage']:g}% data ({identity['count']} images)", fontsize=10)
        figure.legend(handles=[Line2D([], [], marker='o', color='none', markerfacecolor='#ff0000', markeredgecolor='#850000',
                                     markersize=4, label='Selected images / prototypes')], loc='lower center', frameon=False)
        figure.subplots_adjust(left=.01, right=.99, bottom=.09, top=.86, wspace=.02)
        for extension in ('png', 'pdf'):
            figure.savefig(folder / f'{filename}.{extension}', dpi=400, bbox_inches='tight')
        plt.close(figure)
    print('PLOTS SAVED:', folder, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('extract', 'embed', 'plot'))
    parser.add_argument('--dataset', choices=('CIFAR10', 'CIFAR100'))
    parser.add_argument('--percentage', type=float, choices=(0.2, 1, 2))
    args = parser.parse_args()
    if args.stage == 'extract':
        extract()
    else:
        if args.dataset is None or args.percentage is None:
            parser.error('embed/plot require --dataset and --percentage')
        folder = OUTPUT / args.dataset / f'pt{args.percentage:g}'
        if args.stage == 'embed':
            embed(folder)
        else:
            plot(folder)
