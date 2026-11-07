"""Run the frozen CIFAR10 table recipe from saved SAS or ProbCover images."""
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'Evaluate_benchamrk/mipa_table_initialization'))
import worker as base

SELECTIONS = ROOT / 'Evaluate_benchamrk/selection_baselines/selections/CIFAR10'


def selection_file(variant, percentage):
    return SELECTIONS / ('%s_pt%g.json' % (variant, percentage))


def check_selection(common, variant, percentage):
    marker = base.read(SELECTIONS / 'complete.json')
    path = selection_file(variant, percentage)
    saved = base.read(path)
    assert marker['complete'] and saved['complete']
    assert marker['files'][path.name] == common.file_hash(path)
    assert saved['expected'] == marker['expected']
    assert saved['method'] == variant and saved['percentage'] == percentage
    assert saved['expected']['source'] == 'CIFAR10' and not saved['expected']['ground_truth_labels_used']
    indices = saved['indices']
    assert len(indices) == len(set(indices)) == saved['count'] == int(50000 * percentage / 100)
    assert all(isinstance(index, int) and 0 <= index < 50000 for index in indices)
    return saved


def self_check(_data):
    common = base.reference_eval.reference_modules()['common']
    for variant in ('sas', 'probcover'):
        for percentage in (0.2, 1, 2):
            check_selection(common, variant, percentage)
    print('PASS: six completed, label-free selections; hashes, identities, exact budgets and unique indices.', flush=True)


@torch.no_grad()
def choose_images(method, data, common, dataset, count, device, seed, variant, run, job):
    assert seed == 0 and variant in ('sas', 'probcover')
    saved = check_selection(common, variant, job['generation_args']['percentage'])
    assert saved['expected']['source_fingerprint'] == dataset.base.fingerprint
    assert (ROOT / saved['expected']['teacher']).resolve() == Path(job['teacher']).resolve()
    assert saved['expected']['teacher_sha256'] == job['teacher_sha256']
    assert saved['count'] == count
    rng, numpy_rng = common.rng_state(), np.random.get_state()
    try:
        images = torch.stack([method._images_from_batch(dataset[index]) for index in saved['indices']])
        selection = dict(method=variant, seed=seed, count=count, labels_used=False,
                         indices=saved['indices'], source_fingerprint=dataset.base.fingerprint,
                         selection_file=str(selection_file(variant, job['generation_args']['percentage'])),
                         selection_file_sha256=common.file_hash(selection_file(variant, job['generation_args']['percentage'])),
                         initial_images_sha256=base.tensor_hash(images))
        path = run / 'initialization.json'
        if path.exists():
            assert base.read(path) == selection, 'Saved initialization changed'
        common.atomic_json(path, selection)
        print('INITIALIZATION: %s, %d exact saved real images, sha256=%s' %
              (variant, count, selection['initial_images_sha256']), flush=True)
        return images.to(device)
    finally:
        common.restore_rng(rng)
        np.random.set_state(numpy_rng)


def main():
    runner = base.frozen_module(Path(base.__file__), 'matched_sas_probcover_mipa', [
        ("choices=('kmeans100', 'iic')", "choices=('sas', 'probcover')"),
        ("== 'aip-yiweilu'", "== 'aip-boyuwang'"),
        ("account='aip-yiweilu'", "account='aip-boyuwang'"),
        ('job = read(baseline_manifest)',
         "job = read(baseline_manifest)\n"
         "    selection_path = selection_file(args.variant, args.percentage)\n"
         "    for source in (DRIVER, LAUNCHER, selection_path, SELECTION_MARKER):\n"
         "        job['code_hashes'][str(source)] = common.file_hash(source)")])
    runner.HERE = HERE
    runner.DRIVER = Path(__file__).resolve()
    runner.LAUNCHER = HERE / 'job.sh'
    runner.SELECTION_MARKER = SELECTIONS / 'complete.json'
    runner.selection_file = selection_file
    runner.choose_images = choose_images
    runner.self_check = self_check
    runner.main()


if __name__ == '__main__':
    main()
