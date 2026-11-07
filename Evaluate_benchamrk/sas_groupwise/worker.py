"""Rebudget the original SAS ranking across groups, then use unchanged evaluation."""
import argparse
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
BENCH, ROOT = HERE.parent, HERE.parent.parent
sys.path.insert(0, str(BENCH))
sys.path.insert(0, str(BENCH / 'selection_baselines'))
from sas_mkdt_indices.worker import IndexUnpickler, reference


def block_prefixes(indices, group_count, budget):
    """Preserve the ordering within each equal-sized block of this artifact."""
    assert group_count > 0 and 0 < budget <= len(indices)
    assert len(indices) % group_count == budget % group_count == 0
    assert len(set(indices)) == len(indices)
    width, quota = len(indices) // group_count, budget // group_count
    return [index for start in range(0, len(indices), width) for index in indices[start:start + quota]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    assert os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_JOB_ACCOUNT') == 'aip-yiweilu'
    reference.torch.set_num_threads(1)
    reference.torch.set_num_interop_threads(1)
    reference.torch.backends.cuda.matmul.allow_tf32 = False
    reference.torch.backends.cudnn.allow_tf32 = False
    reference.seed_all(0)
    manifest = reference.read(HERE / 'manifest.json')
    for relative, checksum in manifest['code_hashes'].items():
        assert reference.file_hash(ROOT / relative) == checksum, 'Code changed: ' + relative
    artifacts = []
    for item in manifest['inputs']:
        path = HERE / item['path']
        assert reference.file_hash(path) == item['sha256']
        with path.open('rb') as handle:
            artifacts.append(IndexUnpickler(handle).load())
    ranking, example = artifacts[0]['indices'], artifacts[1]
    assert all(type(index) is int for index in ranking + example)
    assert len(ranking) == 50000 and set(ranking) == set(range(50000))
    assert len(example) == 2500
    # This pinned ranking stores 250-image blocks in rounds, not whole groups.
    assert example == ranking[:2500], 'Published 5% selection must match this ranking'
    assert block_prefixes(list(range(60)), 3, 6) == [0, 1, 20, 21, 40, 41]
    assert block_prefixes(list(range(60)), 3, 60) == list(range(60))
    for invalid in ([0, 0, 2, 3], [0, 1, 2]):
        try:
            block_prefixes(invalid, 2, 2)
        except AssertionError:
            pass
        else:
            raise AssertionError('Invalid block input accepted')
    indices = block_prefixes(example, 10, 500)
    assert len(indices) == len(set(indices)) == 500
    assert indices != ranking[:500] and indices != example[:500]
    dataset = reference.RawDataset(manifest['data_root'], 'CIFAR10', True)
    expected = dict(source='CIFAR10', source_fingerprint=dataset.fingerprint,
                    selection_origin='original_SAS_ranking_groupwise_rebudget',
                    upstream_revision=manifest['upstream_revision'], upstream_files=manifest['inputs'],
                    group_sizes=[250] * 10, group_quotas=[50] * 10,
                    group_recovery='artifact-specific ten 250-image blocks in published 5% selection',
                    labels_used_for_rebudgeting=False, original_selection_label_usage='not_established',
                    code_hashes=manifest['code_hashes'])
    folder = HERE / 'selections/CIFAR10'
    result = dict(complete=True, expected=expected, method='sas_groupwise', percentage=1, count=500, indices=indices)
    with reference.exclusive_run(folder):
        target = folder / 'sas_groupwise_pt1.json'
        if target.exists():
            assert reference.read(target) == result
        else:
            reference.atomic_json(target, result)
        reference.atomic_json(folder / 'complete.json', dict(complete=True, expected=expected,
                              files={target.name: reference.file_hash(target)}))
    # Labels are inspected only after selection, never used to change it.
    counts = [int((dataset.targets[indices] == label).sum()) for label in range(10)]
    print('PASS: official 5% prefix identity, block-prefix checks, 500 unique corrected indices', flush=True)
    print('Group sizes:', expected['group_sizes'], 'quotas:', expected['group_quotas'], flush=True)
    print('Diagnostic class counts (not selection inputs):', counts, flush=True)
    if not args.prepare_only:
        assert reference.torch.cuda.is_available() and reference.torch.cuda.device_count() == 1
        reference.HERE = HERE
        reference.evaluate(manifest, 'CIFAR10', 'sas_groupwise', 1)


if __name__ == '__main__':
    main()
