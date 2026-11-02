"""CIFAR100 0.2% noise initialization, 50k MIPA iterations, then third-row PNG export."""
from pathlib import Path
import fcntl
import re
import sys

from PIL import Image
import torch

import noise
import worker as base

VARIANT = 'noise5x_cifar100'
DESTINATION = base.ROOT / 'OT-SSL-DD/cifar100_1percent_pairs'


def self_check(_data):
    normalization = base.reference_eval.SYNTHETIC_NORMALIZATION['CIFAR100']
    rng = torch.get_rng_state()
    images = noise.noise_images(3, 0, normalization)
    assert torch.equal(images, noise.noise_images(3, 0, normalization))
    assert not torch.equal(images, noise.noise_images(3, 1, normalization))
    assert torch.equal(rng, torch.get_rng_state())
    mean, std = [torch.tensor(v)[None, :, None, None] for v in normalization]
    pixels = images * std + mean
    assert images.shape == (3, 3, 32, 32) and torch.isfinite(images).all() and images.std() > .5
    assert pixels.min() >= -1e-7 and pixels.max() <= 1 + 1e-7
    print('PASS: deterministic CIFAR100 Gaussian noise, unchanged RNG, finite valid pixels', flush=True)


@torch.no_grad()
def choose_images(method, data, common, dataset, count, device, seed, variant, run, job):
    assert variant == VARIANT and count == 100 and seed == 0
    assert job['generation_args']['dataset'] == 'CIFAR100'
    normalization = base.reference_eval.SYNTHETIC_NORMALIZATION['CIFAR100']
    images = noise.noise_images(count, seed, normalization)
    selection = dict(method='gaussian_noise', dataset='CIFAR100', percentage=.2, count=count, seed=seed,
                     distribution='N(0,1) in normalized image space', normalization=normalization,
                     pixel_policy='denormalize_clamp_0_1_then_normalize', labels_used=False, real_images_used=False,
                     source_fingerprint=dataset.base.fingerprint, initial_images_sha256=base.tensor_hash(images))
    path = run / 'initialization.json'
    if path.exists():
        assert base.read(path) == selection
    common.atomic_json(path, selection)
    print('CIFAR100 GAUSSIAN INITIALIZATION: ' + base.json.dumps(selection), flush=True)
    return images.to(device)


def export_images():
    common = base.reference_eval.reference_modules()['common']
    run = base.HERE / 'runs' / VARIANT / 'pt0.2'
    job = base.read(run / 'job.json')
    path = Path(job['synthetic_dir']) / 'res_IIC_CIFAR100_ResNet18BN_0.2percent_final.pt'
    saved = torch.load(path, map_location='cpu', weights_only=True)
    assert saved['complete'] and saved['iteration'] == 50000 and saved['dataset'] == 'CIFAR100'
    assert saved['percentage'] == .2 and saved['data'].shape == (100, 3, 32, 32)
    assert saved['run_args'] == job['generation_args'] and saved['init'] == VARIANT
    assert saved['initialization'] == base.read(run / 'initialization.json')
    assert base.tensor_hash(saved['initial_data']) == saved['initialization']['initial_images_sha256']
    assert torch.isfinite(saved['data']).all()
    mean, std = [torch.tensor(v)[None, :, None, None] for v in saved['normalization']]
    pixels = (saved['data'] * std + mean).clamp(0, 1)
    indices = list(range(0, 100, 10))
    files = {}
    for position, index in enumerate(indices, 1):
        output = DESTINATION / ('noise_%02d.png' % position)
        rgb = (pixels[index] * 255 + .5).clamp(0, 255).to(torch.uint8).permute(1, 2, 0).contiguous().numpy()
        if output.exists():
            with Image.open(output) as old:
                assert old.mode == 'RGB' and old.size == (32, 32) and old.tobytes() == rgb.tobytes(), output
        else:
            temporary = output.with_suffix('.tmp.png')
            Image.fromarray(rgb).save(temporary)
            temporary.replace(output)
        with Image.open(output) as checked:
            assert checked.mode == 'RGB' and checked.size == (32, 32) and checked.tobytes() == rgb.tobytes()
        files[output.name] = common.file_hash(output)
    references = re.findall(r'\\includegraphics\[[^\]]+\]\{([^}]+)\}', (DESTINATION / 'figure.tex').read_text())
    assert len(references) == 30 and len(set(references)) == 30
    assert all((base.ROOT / 'OT-SSL-DD' / name).is_file() for name in references)
    assert [Path(name).name for name in references[20:]] == list(files)
    provenance = dict(complete=True, dataset='CIFAR100', percentage=.2, iteration=50000, initialization='Gaussian noise',
                      checkpoint=str(path), checkpoint_sha256=common.file_hash(path), generation_args=saved['run_args'],
                      prototype_indices=indices, selection='Fixed indices 0,10,...,90; no labels or visual selection',
                      corresponding_to_original_rows=False, display='32x32 RGB, denormalized and rounded to uint8; no enhancement',
                      output_sha256=files)
    common.atomic_json(DESTINATION / 'noise_sources.json', provenance)
    common.atomic_json(run / 'figure_export_complete.json', provenance)
    print('EXPORTED TEN NOISE PROTOTYPES: ' + str(DESTINATION), flush=True)


def main():
    runner = base.frozen_module(Path(base.__file__), 'cifar100_noise5x', [
        ("os.environ.get('SLURM_JOB_ACCOUNT') == 'aip-yiweilu'", "os.environ.get('SLURM_JOB_ACCOUNT') in ('aip-boyuwang', 'aip-yiweilu')"),
        ("account='aip-yiweilu'", "account=read(run / 'job.json')['account'] if (run / 'job.json').exists() else os.environ['SLURM_JOB_ACCOUNT']"),
        ("choices=('kmeans100', 'iic')", "choices=(%r,)" % VARIANT),
        ("'CIFAR10_pt%s.json'", "'CIFAR100_pt%s.json'"),
        ("name='CIFAR10_%s_pt%g'", "name='CIFAR100_%s_pt%g'"),
        ("'res_IIC_CIFAR10_ResNet18BN_%gpercent_final.pt'", "'res_IIC_CIFAR100_ResNet18BN_%gpercent_final.pt'"),
        ('job = read(baseline_manifest)',
         "job = read(baseline_manifest)\n"
         "    job['generation_args']['Iteration'] *= 5\n"
         "    job['noise_iteration_multiplier'] = 5\n"
         "    job['code_hashes'][FIGURE_DRIVER] = common.file_hash(FIGURE_DRIVER)\n"
         "    job['code_hashes'][NOISE_HELPER] = common.file_hash(NOISE_HELPER)")])
    runner.FIGURE_DRIVER = str(Path(__file__).resolve())
    runner.NOISE_HELPER = str(Path(noise.__file__).resolve())
    runner.self_check = self_check
    runner.choose_images = choose_images
    sys.argv = [str(Path(__file__)), VARIANT, '0.2', 'generation']
    runner.main()
    export_images()


if __name__ == '__main__':
    with (base.HERE / 'cifar100_noise5x_run.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print('SKIP: the other account is already running this CIFAR100 experiment.', flush=True)
            sys.exit(75)  # Stop the batch shell before it renders an unfinished figure.
        main()
