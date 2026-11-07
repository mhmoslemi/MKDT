"""Evaluate MKDT's supplied SAS indices; do not regenerate selections."""
import argparse
import fcntl
import os
from pathlib import Path
import pickle
import sys

HERE = Path(__file__).resolve().parent
BENCH, ROOT = HERE.parent, HERE.parent.parent
sys.path.insert(0, str(BENCH))
sys.path.insert(0, str(BENCH / 'selection_baselines'))
from selection_baselines import worker as reference


class IndexUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        raise pickle.UnpicklingError('Expected only a plain list of image indices')


def prepare(manifest, source):
    dataset = reference.RawDataset(manifest['data_root'], source, True)
    expected = dict(source=source, source_fingerprint=dataset.fingerprint,
                    selection_origin='MKDT_supplied_SAS_indices',
                    upstream_revision=manifest['upstream_revision'],
                    upstream_files=manifest['datasets'][source],
                    selection_label_usage='not_documented_by_MKDT',
                    code_hashes=manifest['code_hashes'])
    folder = HERE / 'selections' / source
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / '.prepare.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        files = {}
        for percentage in (1, 2):
            artifact = manifest['datasets'][source][str(percentage)]
            path = HERE / 'indices' / artifact['filename']
            assert reference.file_hash(path) == artifact['sha256']
            with path.open('rb') as handle:
                indices = IndexUnpickler(handle).load()
            count = int(len(dataset) * percentage / 100)
            assert isinstance(indices, list) and len(indices) == count
            assert all(type(i) is int and 0 <= i < len(dataset) for i in indices)
            assert len(set(indices)) == count
            result = dict(complete=True, expected=expected, method='sas_mkdt',
                          percentage=percentage, count=count, indices=indices,
                          upstream_file=artifact)
            relative = f'sas_mkdt_pt{percentage}.json'
            target = folder / relative
            if target.is_file():
                assert reference.read(target) == result
            else:
                reference.atomic_json(target, result)
            files[relative] = reference.file_hash(target)
            print(f'Validated unchanged MKDT SAS: {source}, {count} unique indices', flush=True)
        reference.atomic_json(folder / 'complete.json', dict(complete=True, expected=expected, files=files))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('source', choices=('CIFAR10', 'CIFAR100'))
    parser.add_argument('percentage', type=int, choices=(1, 2))
    args = parser.parse_args()
    assert os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_JOB_ACCOUNT') == 'aip-yiweilu'
    assert reference.torch.cuda.is_available() and reference.torch.cuda.device_count() == 1
    reference.torch.set_num_threads(1)
    reference.torch.set_num_interop_threads(1)
    reference.torch.backends.cuda.matmul.allow_tf32 = False
    reference.torch.backends.cudnn.allow_tf32 = False
    reference.seed_all(0)
    manifest = reference.read(HERE / 'manifest.json')
    for relative, checksum in manifest['code_hashes'].items():
        assert reference.file_hash(ROOT / relative) == checksum, 'Code changed: ' + relative
    prepare(manifest, args.source)
    reference.HERE = HERE
    reference.evaluate(manifest, args.source, 'sas_mkdt', args.percentage)


if __name__ == '__main__':
    main()
