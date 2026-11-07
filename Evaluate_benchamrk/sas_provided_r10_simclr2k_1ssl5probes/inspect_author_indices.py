"""Read-only inspection of the supplied subset lists and their ordering."""
import collections
import io
import json
import pickle
from pathlib import Path

ROOT = Path('/scratch/mmoslem3/sas_official_standalone/upstream')

class IndexUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        raise ValueError(f'Unexpected executable pickle content: {module}.{name}')

def read(path):
    obj = IndexUnpickler(io.BytesIO(path.read_bytes())).load()
    keys = list(obj) if isinstance(obj, dict) else None
    indices = obj['indices'] if isinstance(obj, dict) else obj
    assert isinstance(indices, list) and all(type(i) is int and 0 <= i < 50000 for i in indices)
    assert len(indices) == len(set(indices))
    return indices, keys

for name in ('cifar10', 'cifar100'):
    full, keys = read(ROOT/'final_subsets'/f'{name}-cl-core-idx.pkl')
    assert sorted(full) == list(range(50000))
    positions = {index: pos for pos, index in enumerate(full)}
    examples = []
    for path in sorted((ROOT/'examples'/name).glob('*.pkl')):
        indices, example_keys = read(path)
        pos = [positions[i] for i in indices]
        spans = []
        start = previous = pos[0]
        for p in pos[1:]:
            if p != previous + 1:
                spans.append([start, previous])
                start = p
            previous = p
        spans.append([start, previous])
        examples.append(dict(file=path.name, count=len(indices), keys=example_keys,
            is_global_prefix=(indices == full[:len(indices)]), contiguous_spans=len(spans),
            first_spans=spans[:12], last_spans=spans[-3:]))
    print(json.dumps(dict(dataset=name, full_count=len(full), full_keys=keys, examples=examples)), flush=True)
