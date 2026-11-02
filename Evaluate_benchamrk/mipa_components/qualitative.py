"""CIFAR10 MIPA initialization/final images and a shared teacher-space t-SNE."""
import argparse
import json
import os
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch
from torch.nn import functional as F
from torchvision.utils import save_image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import ROOT, atomic_json, file_hash, load_encoder, state_hash
from data import RawDataset


def selected_examples(cleaner=False, cifar100=False, percentage=0.2):
    from PIL import Image

    assert os.environ.get('SLURM_JOB_ID')
    output = ROOT / 'CVPR_tables/ablations/figures' / ('mipa_components_cifar100' if cifar100 else 'mipa_components_cifar10')
    count = 500 if cifar100 and percentage == 1 else 100
    if count == 500:
        output = output.with_name(output.name + '_pt1')
    if cifar100:
        torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', '1')))
        plan = json.loads((ROOT / 'Evaluate_benchamrk/mipa_cvpr_color_crop/manifest.json').read_text())
        job = next(value for value in plan['jobs'] if value['name'] == ('CIFAR100_pt1' if count == 500 else 'CIFAR100_pt0p2'))
        checkpoint = Path(job['synthetic_dir']) / ('res_IIC_CIFAR100_ResNet18BN_%gpercent_final.pt' % percentage)
        checkpoint_hash = file_hash(checkpoint)
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        assert saved['complete'] and saved['iteration'] == job['generation_args']['Iteration'] and saved['run_args'] == job['generation_args']
        assert saved['teacher_seed'] == 0 and saved['init'] == 'random'
        dataset = RawDataset(job['data_root'], 'CIFAR100', True)
        indices = np.random.default_rng(1000).permutation(len(dataset))[:count]
        labels = np.asarray(dataset.targets)[indices]
        classes = [name.replace('_', ' ').capitalize() for name in dataset.base.classes]
        mean, std = (torch.tensor(values).view(1, 3, 1, 1) for values in saved['normalization'])
        initial = (saved['initial_data'] * std + mean).clamp(0, 1)
        final = (saved['data'] * std + mean).clamp(0, 1)
        original = torch.from_numpy(dataset.base.data[indices].copy()).permute(0, 3, 1, 2).float() / 255
        assert initial.shape == final.shape == (count, 3, 32, 32) and torch.isfinite(final).all()
        assert torch.allclose(initial, original, atol=1e-6)
        output.mkdir(parents=True, exist_ok=True)
        save_image(initial, output / 'initial_all.png', nrow=10)
        save_image(final, output / 'final_all.png', nrow=10)
        assert file_hash(checkpoint) == checkpoint_hash
        atomic_json(output / 'inputs.json', dict(checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_hash,
                    generation_args=saved['run_args'], iteration=saved['iteration'],
                    initialization_indices=indices.tolist(), initial_labels=labels.tolist(),
                    initial_pixels_verified_against_real_data=True, classes=classes))
    paths = [output / 'initial_all.png', output / 'final_all.png']
    grids = [Image.open(path).convert('RGB') for path in paths]
    assert all(grid.size == (342, 2 + 34 * (count // 10)) for grid in grids)
    selected = [31, 70, 67, 53, 58, 60]
    candidates = [7, 16, 25, 31, 38, 41, 46, 53, 58, 60, 67, 70, 76, 87, 91, 93]
    if cleaner:
        selected = [3, 31, 67, 56, 60]
        candidates = list(range(count))
    if cifar100:
        selected = [64, 192, 279, 22, 332, 376, 453, 483] if count == 500 else [1, 37, 65, 74, 99]
    stem = 'paper_examples' if cleaner else 'selected_examples'
    if not cifar100:
        indices = np.load(output / 'features_and_indices.npz', allow_pickle=False)['initialization_indices']
        labels = np.load(ROOT / 'Evaluate_benchamrk/iic_init/tsne/comparison_iaTEXR/CIFAR10/labels.npy', allow_pickle=False)[indices]
        classes = ('Airplane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck')
    assert indices.shape == labels.shape == (count,) and np.all((labels >= 0) & (labels < len(classes)))
    if cleaner:
        assert len(selected) == len(set(labels[selected])) == (8 if count == 500 else 5)

    def tile(grid, index):
        left, top = 2 + 34 * (index % 10), 2 + 34 * (index // 10)
        pixels = np.asarray(grid.crop((left, top, left + 32, top + 32)))
        assert pixels.shape == (32, 32, 3)
        return pixels

    assert len(set(selected)) == len(selected) and set(selected).issubset(candidates)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'pdf.fonttype': 42, 'savefig.facecolor': 'white'})
    columns = 4 if count == 500 else len(selected)
    rows = 4 if count == 500 else 2
    figure, axes = plt.subplots(rows, columns, figsize=(3.35, 2.7 if count == 500 else (1.28 if cleaner else 1.15)))
    for row, grid in enumerate(grids):
        for position, index in enumerate(selected):
            block, column = divmod(position, columns)
            axis = axes[2 * block + row, column]
            axis.imshow(tile(grid, index), interpolation='nearest')
            axis.set_axis_off()
            if cleaner and row == 0:
                axis.set_title(classes[labels[index]], fontsize=7, pad=3)
    figure.subplots_adjust(left=.13, right=1, top=.93 if count == 500 else (.88 if cleaner else .98), bottom=.02, wspace=.05, hspace=.30 if count == 500 else .06)
    for row, title in enumerate(('Before', 'After') * (rows // 2)):
        position = axes[row, 0].get_position()
        figure.text(.005, (position.y0 + position.y1) / 2, title, va='center', fontsize=7)
    for extension in ('png', 'pdf'):
        figure.savefig(output / (stem + '.' + extension), dpi=600, bbox_inches='tight')
    plt.close(figure)

    page_size = 50 if count == 500 else 16
    for start in range(0, len(candidates), page_size):
        figure, axes = plt.subplots(10 if count == 500 else 4, 10 if count == 500 else 8,
                                    figsize=(12, 13) if count == 500 else (10, 7.2 if cifar100 else 5.7))
        for axis in axes.flat:
            axis.set_axis_off()
        for position, index in enumerate(candidates[start:start + page_size]):
            row, column = divmod(position, 5 if count == 500 else 4)
            for version, grid in enumerate(grids):
                axis = axes[row, 2 * column + version]
                axis.imshow(tile(grid, index), interpolation='nearest')
                axis.set_title('#%d %s\n%s' % (index, classes[labels[index]], 'before' if version == 0 else 'after'), fontsize=6 if count == 500 else 8)
        figure.tight_layout(pad=.4, w_pad=.35, h_pad=.6)
        name = 'cleaner_candidate_pairs' if cleaner else 'candidate_pairs'
        if start:
            name += '_%d' % (start // page_size + 1)
        figure.savefig(output / (name + '.png'), dpi=120 if count == 500 else 200, bbox_inches='tight')
        plt.close(figure)
    drift = {str(index): float(np.sqrt(np.mean((tile(grids[1], index).astype(float) - tile(grids[0], index)) ** 2))) for index in selected}
    atomic_json(output / (stem + '.json'), dict(indices_zero_based=selected, candidate_indices_zero_based=candidates,
                initial_class_labels=[classes[labels[index]] for index in selected],
                selection='Visually selected illustrative pairs, not a representative sample or a reconstruction-quality claim.',
                source_grid_sha256=[file_hash(path) for path in paths], pixel_drift_rms_255=drift,
                display='Unmodified saved RGB tiles; nearest-neighbor display only.'))
    print('SAVED selected examples: ' + str(output / (stem + '.png')), flush=True)


@torch.inference_mode()
def main():
    assert os.environ.get('SLURM_JOB_ID') and torch.cuda.is_available()
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', '1')))
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    bench = ROOT / 'Evaluate_benchamrk'
    plan = json.loads((bench / 'mipa_cvpr_color_crop/manifest.json').read_text())
    job = next(value for value in plan['jobs'] if value['name'] == 'CIFAR10_pt0p2')
    checkpoint = Path(job['synthetic_dir']) / 'res_IIC_CIFAR10_ResNet18BN_0.2percent_final.pt'
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    assert saved['complete'] and saved['iteration'] == 10000 and saved['run_args'] == job['generation_args']
    assert saved['teacher_seed'] == 0 and saved['init'] == 'random'
    teacher = Path(job['teacher'])
    assert file_hash(teacher) == job['teacher_sha256']
    cache = bench / 'iic_init/tsne/comparison_iaTEXR/CIFAR10'
    selection = json.loads((cache / 'selection.json').read_text())
    assert selection['teacher_sha256'] == job['teacher_sha256']
    assert file_hash(cache / 'features.npy') == selection['features_sha256']
    real_features = np.load(cache / 'features.npy', allow_pickle=False)
    labels = np.load(cache / 'labels.npy', allow_pickle=False)
    dataset = RawDataset(job['data_root'], 'CIFAR10', True)
    assert dataset.fingerprint == selection['train_fingerprint']
    assert real_features.shape == (len(dataset), 512) and labels.shape == (len(dataset),)
    assert np.array_equal(labels, np.asarray(dataset.targets))
    assert np.isfinite(real_features).all() and np.allclose(np.linalg.norm(real_features, axis=1), 1, atol=1e-5)

    mean, std = (torch.tensor(values).view(1, 3, 1, 1) for values in saved['normalization'])
    initial = (saved['initial_data'] * std + mean).clamp(0, 1)
    final = (saved['data'] * std + mean).clamp(0, 1)
    indices = np.random.default_rng(1000).permutation(len(dataset))[:100]
    original = torch.from_numpy(dataset.base.data[indices].copy()).permute(0, 3, 1, 2).float() / 255
    assert initial.shape == final.shape == (100, 3, 32, 32)
    assert torch.allclose(initial, original, atol=1e-6)
    encoder, _, teacher_saved = load_encoder(teacher, device='cuda')
    teacher_mean, teacher_std = (torch.tensor(values, device='cuda').view(1, 3, 1, 1) for values in teacher_saved['normalization'])
    prototype_features = []
    for images in (initial, final):
        prototype_features.append(F.normalize(encoder((images.cuda() - teacher_mean) / teacher_std), dim=1).cpu().numpy())
    assert np.allclose(prototype_features[0], real_features[indices], atol=2e-4)
    assert state_hash(encoder) == teacher_saved['encoder_sha256']
    assert all(not parameter.requires_grad and parameter.grad is None for parameter in encoder.parameters())

    output = ROOT / 'CVPR_tables/ablations/figures/mipa_components_cifar10'
    output.mkdir(parents=True, exist_ok=True)
    identity = dict(checkpoint_sha256=file_hash(checkpoint), teacher_sha256=job['teacher_sha256'],
                    real_features_sha256=selection['features_sha256'], seed=0, background_count=10000,
                    prototype_count=100, perplexity=30, max_iter=1000, labels_used_for_embedding=False)
    identity_path = output / 'inputs.json'
    if identity_path.exists():
        assert json.loads(identity_path.read_text()) == identity, 'Qualitative inputs changed'
    atomic_json(identity_path, identity)
    save_image(initial, output / 'initial_all.png', nrow=10)
    save_image(final, output / 'final_all.png', nrow=10)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'pdf.fonttype': 42, 'savefig.facecolor': 'white'})
    figure, axes = plt.subplots(2, 10, figsize=(3.35, 0.95))
    for row, images in enumerate((initial, final)):
        for column in range(10):
            axes[row, column].imshow(images[column].permute(1, 2, 0).numpy(), interpolation='nearest')
            axes[row, column].set_axis_off()
    figure.subplots_adjust(left=.11, right=1, top=1, bottom=0, wspace=.03, hspace=.08)
    figure.text(.005, .74, 'Initial', va='center', fontsize=7)
    figure.text(.005, .25, 'MIPA', va='center', fontsize=7)
    for extension in ('pdf', 'png'):
        figure.savefig(output / ('prototype_evolution.' + extension), dpi=350, bbox_inches='tight')
    plt.close(figure)

    background = np.sort(np.random.default_rng(0).choice(len(real_features), identity['background_count'], replace=False))
    features = np.concatenate((real_features[background], *prototype_features))
    coordinates = output / 'tsne.npy'
    if coordinates.exists():
        points = np.load(coordinates, allow_pickle=False)
    else:
        print('One joint t-SNE: 10,000 real images, 100 initial and 100 final prototypes.', flush=True)
        reduced = PCA(n_components=50, svd_solver='randomized', random_state=0).fit_transform(features)
        points = TSNE(n_components=2, perplexity=30, learning_rate='auto', init='pca', max_iter=1000,
                      random_state=0, method='barnes_hut', n_jobs=torch.get_num_threads(), verbose=1).fit_transform(reduced)
        np.save(coordinates, points, allow_pickle=False)
    assert points.shape == (10200, 2) and np.isfinite(points).all()
    np.savez(output / 'features_and_indices.npz', initial=prototype_features[0], final=prototype_features[1],
             background_indices=background, initialization_indices=indices)
    figure, axes = plt.subplots(1, 2, figsize=(3.35, 1.75))
    for index, (axis, title) in enumerate(zip(axes, ('Random initialization', 'After MIPA'))):
        axis.scatter(points[:10000, 0], points[:10000, 1], c=labels[background], cmap='tab10',
                     vmin=0, vmax=9, s=.8, alpha=.25, linewidths=0, rasterized=True)
        selected = points[10000 + 100 * index:10100 + 100 * index]
        axis.scatter(selected[:, 0], selected[:, 1], c='#d62728', marker='*', s=17,
                     edgecolors='black', linewidths=.2, zorder=3)
        axis.set_title(title, fontsize=8)
        axis.set_xlim(points[:, 0].min() - 1, points[:, 0].max() + 1)
        axis.set_ylim(points[:, 1].min() - 1, points[:, 1].max() + 1)
        axis.set_aspect('equal', adjustable='box')
        axis.set_axis_off()
    figure.legend(handles=[Line2D([], [], marker='*', linestyle='none', color='#d62728', markeredgecolor='black',
                                  markeredgewidth=.3, markersize=6, label='Prototypes')],
                  loc='lower center', frameon=False, fontsize=7)
    figure.subplots_adjust(left=0, right=1, top=.85, bottom=.13, wspace=.06)
    for extension in ('pdf', 'png'):
        figure.savefig(output / ('prototype_tsne.' + extension), dpi=350, bbox_inches='tight')
    plt.close(figure)

    metrics = {}
    for name, features in zip(('initial', 'final'), prototype_features):
        similarity = real_features @ features.T
        pairwise = features @ features.T
        off_diagonal = pairwise[~np.eye(len(features), dtype=bool)]
        metrics[name] = dict(mean_nearest_prototype_cosine_distance=float((1 - similarity.max(axis=1)).mean()),
                             mean_pairwise_prototype_cosine=float(off_diagonal.mean()),
                             maximum_pairwise_prototype_cosine=float(off_diagonal.max()),
                             distinct_nearest_real_images=int(len(np.unique(similarity.argmax(axis=0)))))
    assert file_hash(teacher) == identity['teacher_sha256'] and file_hash(checkpoint) == identity['checkpoint_sha256']
    atomic_json(output / 'metrics.json', dict(metrics=metrics, identity=identity, teacher_unchanged=True,
                note='t-SNE illustrates geometry; coverage metrics use all 50,000 real teacher features.'))
    print('SAVED: ' + str(output), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--examples', action='store_true', help='Make selected before/after panels from the existing saved grids; no feature extraction or training')
    group.add_argument('--cleaner-examples', action='store_true', help='Show the less-speckled alternative shortlist without modifying the saved pixels')
    group.add_argument('--cifar100-examples', action='store_true', help='Inspect CIFAR100 original and distilled saved pixels on CPU')
    group.add_argument('--cifar100-one-percent', action='store_true', help='Inspect all 500 CIFAR100 1% pairs and render eight selected examples')
    args = parser.parse_args()
    if args.examples or args.cleaner_examples or args.cifar100_examples or args.cifar100_one_percent:
        selected_examples(cleaner=args.cleaner_examples or args.cifar100_examples or args.cifar100_one_percent,
                          cifar100=args.cifar100_examples or args.cifar100_one_percent, percentage=1 if args.cifar100_one_percent else 0.2)
    else:
        main()
