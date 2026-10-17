"""One-core, 6-GB preflight. All short-run outputs are validation-only."""
import copy
import gc
import json
from pathlib import Path
import sys
import time

batch, data_root = map(Path, sys.argv[1:3])
sys.path.insert(0, str(batch / 'code/Evaluate_benchamrk'))
import torch
from common import atomic_json, digest_json
from data import RawDataset, TARGETS, select_subset
from evaluate import linear_probe
from run import validate_protocol
from train_ssl import train

torch.set_num_threads(1)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
assert torch.cuda.is_available() and torch.cuda.device_count() == 1
print('GPU:', torch.cuda.get_device_name(), 'torch:', torch.__version__, flush=True)
protocol = json.loads((batch / 'code/Evaluate_benchamrk/protocol.json').read_text())
validate_protocol(protocol)
started = time.monotonic()
fingerprints = {}
for name in TARGETS:
    fingerprints[name] = []
    for training in (True, False):
        dataset = RawDataset(data_root, name, training)
        image, label = dataset[0]
        assert image.mode == 'RGB' and 0 <= label < dataset.classes
        fingerprints[name].append(dataset.fingerprint)
        print('DATA', name, training, len(dataset), flush=True)
        del dataset
        gc.collect()

smoke = copy.deepcopy(protocol)
smoke['ssl'].update(subset_epochs=2, warmup_epochs=0, checkpoint_every_epochs=2)
smoke['evaluation']['epochs'] = 1
root = batch / 'validation_only'
atomic_json(root / 'protocol.json', {'validation_only': True, 'protocol': smoke})
timings = []
for source in protocol['sources']:
    source_set = RawDataset(data_root, source, True)
    selected, _ = select_subset(source_set, 'random', 5, 0, smoke,
                                root / 'selections', 0, 'cuda')
    for model in protocol['models']:
        for method in protocol['methods']:
            folder = root / source / model / method
            identity = {'validation_only': True, 'source': source, 'model': model,
                        'method': method, 'signature': digest_json(smoke), 'seed': 0}
            torch.cuda.reset_peak_memory_stats()
            begin = time.monotonic()
            # Exercise a genuine intermediate save and optimizer/RNG resume.
            pending, _ = train(selected, smoke, source, model, method, 0,
                               folder, identity, 0, 'cuda', stop_after_epoch=1)
            assert pending is None and not (folder / 'encoder.pt').exists()
            checkpoint, metadata = train(selected, smoke, source, model, method, 0,
                                          folder, identity, 0, 'cuda', stop_after_epoch=2)
            epochs = [json.loads(line) for line in (folder / 'ssl.jsonl').read_text().splitlines()]
            timing = {'source': source, 'model': model, 'ssl': method, 'subset': 5,
                      'two_epoch_wall_seconds': time.monotonic() - begin,
                      'epoch_seconds': [row['seconds'] for row in epochs],
                      'peak_gpu_bytes': torch.cuda.max_memory_allocated()}
            timings.append(timing)
            print('SMOKE_SSL', json.dumps(timing), flush=True)
            targets = TARGETS if model == 'ResNet18' and method == 'simclr' else [source]
            for target in targets:
                train_set = source_set if target == source else RawDataset(data_root, target, True)
                test_set = RawDataset(data_root, target, False)
                for label in protocol['evaluation']['label_percentages']:
                    result = linear_probe(checkpoint, identity, smoke, source, target,
                                          train_set, test_set, label, 0,
                                          folder / 'probes' / f'{target}_lbl{label}.json', 0, 'cuda')
                    assert result['encoder_unchanged'] and result['test_count'] == len(test_set)
                    assert result['encoder_sha256'] == metadata['encoder_sha256']
                del train_set, test_set
                gc.collect()
            (folder / 'training.pt').unlink(missing_ok=True)
            checkpoint.unlink()
            gc.collect()
            torch.cuda.empty_cache()
    del selected, source_set
    gc.collect()
atomic_json(batch / 'validation.json', {'complete': True, 'validation_only': True,
            'production_protocol_sha256': digest_json(protocol), 'cases': timings,
            'datasets': fingerprints, 'workers': 0, 'threads': 1,
            'seconds': time.monotonic() - started})
print('VALIDATION PASSED; short-run scores are not production results.', flush=True)

