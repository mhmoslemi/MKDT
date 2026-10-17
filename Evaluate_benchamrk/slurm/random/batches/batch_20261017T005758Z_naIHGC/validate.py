"""GPU/data smoke tests only; these short-run scores are NOT benchmark results."""
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

torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
assert torch.cuda.is_available(), 'CUDA is unavailable'
assert torch.cuda.device_count() == 1, 'Expected one assigned GPU'
print('GPU:', torch.cuda.get_device_name(), 'torch:', torch.__version__, flush=True)
protocol = json.loads((batch / 'code/Evaluate_benchamrk/protocol.json').read_text())
validate_protocol(protocol)
started = time.monotonic()
datasets = {}
for name in TARGETS:
    datasets[name] = (RawDataset(data_root, name, True), RawDataset(data_root, name, False))
    for dataset in datasets[name]:
        image, label = dataset[0]
        assert image.mode == 'RGB' and 0 <= label < dataset.classes
    print('DATA', name, *(len(d) for d in datasets[name]), flush=True)

smoke = copy.deepcopy(protocol)
smoke['ssl'].update(subset_epochs=2, warmup_epochs=0, checkpoint_every_epochs=2)
smoke['evaluation']['epochs'] = 1
root = batch / 'validation_only'
atomic_json(root / 'protocol.json', {'validation_only': True, 'protocol': smoke})
timings = []
for source in protocol['sources']:
    selected, _ = select_subset(datasets[source][0], 'random', 5, 0, smoke,
                                root / 'selections', 4, 'cuda')
    for model in protocol['models']:
        for method in protocol['methods']:
            folder = root / source / model / method
            identity = {'validation_only': True, 'source': source, 'model': model,
                        'method': method, 'signature': digest_json(smoke), 'seed': 0}
            torch.cuda.reset_peak_memory_stats()
            begin = time.monotonic()
            checkpoint, metadata = train(selected, smoke, source, model, method, 0,
                                          folder, identity, 4, 'cuda')
            elapsed = time.monotonic() - begin
            epochs = [json.loads(line) for line in (folder / 'ssl.jsonl').read_text().splitlines()]
            timing = {'source': source, 'model': model, 'ssl': method, 'subset': 5,
                      'two_epoch_wall_seconds': elapsed,
                      'epoch_seconds': [row['seconds'] for row in epochs],
                      'peak_gpu_bytes': torch.cuda.max_memory_allocated()}
            timings.append(timing)
            print('SMOKE_SSL', json.dumps(timing), flush=True)
            # Check every transfer split and the two label budgets, at both
            # native source resolutions, using the SAME saved encoder.
            targets = TARGETS if model == 'ResNet18' and method == 'simclr' else [source]
            for target in targets:
                for label in protocol['evaluation']['label_percentages']:
                    result = linear_probe(checkpoint, identity, smoke, source, target,
                                          *datasets[target], label, 0,
                                          folder / 'probes' / f'{target}_lbl{label}.json', 4, 'cuda')
                    assert result['encoder_unchanged'] and result['test_count'] == len(datasets[target][1])
                    assert result['encoder_sha256'] == metadata['encoder_sha256']
            # These are disposable smoke-test checkpoints, not production data.
            (folder / 'training.pt').unlink(missing_ok=True)
            checkpoint.unlink()
            gc.collect()
            torch.cuda.empty_cache()
atomic_json(batch / 'validation.json', {'complete': True, 'validation_only': True,
            'production_protocol_sha256': digest_json(protocol), 'cases': timings,
            'seconds': time.monotonic() - started})
print('VALIDATION PASSED; no smoke-test scores enter the production tables.', flush=True)

