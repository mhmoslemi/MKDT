"""Matched CIFAR10 initialization ablation: Gaussian noise, standard or 3x iterations."""
from pathlib import Path

import torch

import worker as base


def noise_images(count, seed, normalization):
    generator = torch.Generator().manual_seed(seed)
    images = torch.randn(count, 3, 32, 32, generator=generator)
    mean, std = [torch.tensor(v)[None, :, None, None] for v in normalization]
    return (((images * std + mean).clamp(0, 1) - mean) / std).contiguous()


def self_check(_data):
    normalization = base.reference_eval.SYNTHETIC_NORMALIZATION['CIFAR10']
    rng = torch.get_rng_state()
    images = noise_images(3, 0, normalization)
    assert torch.equal(images, noise_images(3, 0, normalization))
    assert torch.equal(images[:1], noise_images(1, 0, normalization))
    assert not torch.equal(images, noise_images(3, 1, normalization))
    assert torch.equal(rng, torch.get_rng_state()), 'Initialization changed the distillation RNG'
    mean, std = [torch.tensor(v)[None, :, None, None] for v in normalization]
    pixels = images * std + mean
    assert images.shape == (3, 3, 32, 32) and torch.isfinite(images).all() and images.std() > .5
    assert pixels.min() >= -1e-7 and pixels.max() <= 1 + 1e-7
    print('PASS: paired noise equality, nested budgets, seed independence, unchanged RNG, valid pixels', flush=True)


@torch.no_grad()
def choose_images(method, data, common, dataset, count, device, seed, variant, run, job):
    assert variant in ('noise1x', 'noise3x') and seed == 0
    normalization = base.reference_eval.SYNTHETIC_NORMALIZATION['CIFAR10']
    images = noise_images(count, seed, normalization)
    selection = dict(method='gaussian_noise', seed=seed, count=count,
                     distribution='N(0,1) in normalized image space', normalization=normalization,
                     pixel_policy='denormalize_clamp_0_1_then_normalize',
                     source_fingerprint=dataset.base.fingerprint, labels_used=False, real_images_used=False,
                     initial_images_sha256=base.tensor_hash(images))
    path = run / 'initialization.json'
    if path.exists():
        assert base.read(path) == selection, 'Saved noise initialization changed'
    common.atomic_json(path, selection)
    print('GAUSSIAN INITIALIZATION: %d images, seed=%d, sha256=%s' %
          (count, seed, selection['initial_images_sha256']), flush=True)
    return images.to(device)


def main():
    # Reuse the completed initialization runner; only its initializer and iteration budget change.
    runner = base.frozen_module(Path(base.__file__), 'matched_noise_initialization', [
        ("choices=('kmeans100', 'iic')", "choices=('noise1x', 'noise3x')"),
        ('job = read(baseline_manifest)',
         "job = read(baseline_manifest)\n"
         "    job['noise_iteration_multiplier'] = 3 if args.variant == 'noise3x' else 1\n"
         "    job['generation_args']['Iteration'] *= job['noise_iteration_multiplier']\n"
         "    job['code_hashes'][NOISE_DRIVER] = common.file_hash(NOISE_DRIVER)")])
    runner.NOISE_DRIVER = str(Path(__file__).resolve())
    runner.choose_images = choose_images
    runner.self_check = self_check
    runner.main()


if __name__ == '__main__':
    main()
