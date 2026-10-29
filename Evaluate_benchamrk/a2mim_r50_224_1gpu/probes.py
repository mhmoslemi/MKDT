"""Five frozen heads per cell; keep the upstream ResNet50/224 preprocessing explicit."""
import copy
import json
import math
import statistics
import sys
import time
from pathlib import Path

from contracts import BENCH, HERE, allocated, atomic_json, exclusive_run, file_hash, folder, load_unit, read
import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Subset
from torchvision import transforms as T
import common
import evaluate
from data import ImageView, RawDataset, label_indices, loader

NORMALIZATION = [[0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.201]]
original_builder = common.build_encoder


def build_encoder(model, size):
    if model != 'ResNet50ImageNet':
        return original_builder(model, size)
    assert size == 224
    encoder = torchvision.models.resnet50(weights=None, zero_init_residual=True)
    encoder.fc = nn.Identity()
    return encoder, 2048


def transform(protocol, source, training):
    operations = [T.RandomResizedCrop(224, scale=(0.2, 1), interpolation=T.InterpolationMode.BICUBIC),
                  T.RandomHorizontalFlip()] if training else [T.Resize(256, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(224)]
    return T.Compose(operations + [T.ToTensor(), T.Normalize(*NORMALIZATION)])


common.build_encoder = build_encoder
evaluate.probe_transform = transform


def protocol_for(source, target):
    protocol = copy.deepcopy(read(BENCH / 'protocol.json'))
    protocol['normalization'][source] = NORMALIZATION
    protocol['input_size'][source] = 224
    cfg = protocol['evaluation']
    cfg.update(epochs=100, lr=1.0 if target == 'CIFAR100' else 0.1,
               batch_size=256, momentum=0.9, weight_decay=0.0,
               crop_scale=[0.2, 1.0], interpolation='bicubic', test_resize=256,
               input_size=224, normalization=NORMALIZATION, architecture='ResNet50ImageNet')
    return protocol


def run(unit_file, data_root, bundle):
    unit = load_unit(unit_file)
    output = folder(unit)
    marker = read(output / 'ssl_complete.json')
    checkpoint = output / 'encoder.pt'
    if (not marker['complete'] or marker['identity'] != unit['identity'] or marker['epochs'] != unit['epochs']
            or file_hash(checkpoint) != marker['checkpoint_file_sha256'] or read(output / 'config.json') != unit):
        raise ValueError('Wrong or incomplete upstream encoder')
    source = unit['source']
    targets = [source, 'CIFAR100' if source == 'CIFAR10' else 'CIFAR10', 'TinyImageNet']
    cells = [(target, labels) for target in targets for labels in (1, 5)]
    assert bundle in (0, 1)
    for target, labels in cells[bundle * 4:(bundle + 1) * 4]:
        with exclusive_run(output / ('probe_locks/%s_lbl%d' % (target, labels))):
            train_set, test_set = RawDataset(data_root, target, True), RawDataset(data_root, target, False)
            protocol = protocol_for(source, target)
            samples = []
            for seed in range(5):
                path = output / 'evaluations' / ('%s_lbl%d_probe%d.json' % (target, labels, seed))
                result = evaluate.linear_probe(checkpoint, unit['identity'], protocol, source, target,
                    train_set, test_set, labels, seed, path, workers=0, device='cuda')
                score = result['accuracy_percent']
                if (not result['complete'] or not result['encoder_unchanged']
                        or result['encoder_sha256'] != marker['encoder_sha256'] or not math.isfinite(score)):
                    raise ValueError('Invalid frozen probe')
                samples.append(dict(ssl_seed=0, probe_seed=seed, accuracy_percent=score, result_file=str(path)))
            scores = [sample['accuracy_percent'] for sample in samples]
            summary = {key: unit[key] for key in ('source', 'kind', 'subset', 'epochs')}
            summary.update(complete=True, target=target, labels=labels, method='a2mim',
                model='ResNet50ImageNet', input_size=224, one_gpu_adaptation=True,
                ssl_seed=0, linear_epochs=100, expected_probes=5, completed_probes=5,
                std_ddof=1, samples=samples, mean_percent=statistics.mean(scores),
                std_percent=statistics.stdev(scores), identity=unit['identity'], evaluation=protocol['evaluation'])
            summary['tex_value'] = '\\dsacc{%.2f}{%.2f}' % (summary['mean_percent'], summary['std_percent'])
            atomic_json(output / ('summary_%s_lbl%d.json' % (target, labels)), summary)
            print('CELL COMPLETE:', json.dumps(summary), flush=True)
            del train_set, test_set


def calibrate(data_root):
    """Measure this architecture/preprocessing, not the older native-32px probe times."""
    encoder, dimension = build_encoder('ResNet50ImageNet', 224)
    encoder = encoder.cuda().eval().requires_grad_(False)
    observations = {}
    for target in ('CIFAR10', 'CIFAR100', 'TinyImageNet'):
        train_set, test_set = RawDataset(data_root, target, True), RawDataset(data_root, target, False)
        protocol = protocol_for('CIFAR10', target)
        for labels in (1, 5):
            indices = label_indices(train_set, labels, 0)
            batches = loader(ImageView(Subset(train_set, indices), transform(protocol, 'CIFAR10', True)),
                             min(256, len(indices)), 0, 0, shuffle=True)
            head = nn.Linear(dimension, train_set.classes).cuda()
            optimizer = torch.optim.SGD(head.parameters(), lr=protocol['evaluation']['lr'], momentum=0.9)
            times = []
            for epoch in range(3):
                torch.cuda.synchronize()
                started = time.monotonic()
                for images, truth in batches:
                    with torch.no_grad():
                        features = encoder(images.cuda())
                    loss = F.cross_entropy(head(features), truth.cuda())
                    assert torch.isfinite(loss)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                torch.cuda.synchronize()
                times.append(time.monotonic() - started)
            observations['%s|%d' % (target, labels)] = dict(epoch_seconds=times[1:], labeled_count=len(indices))
        batches = loader(ImageView(test_set, transform(protocol, 'CIFAR10', False)), 256, 0, 0)
        torch.cuda.synchronize()
        started = time.monotonic()
        with torch.no_grad():
            for images, _ in batches:
                assert torch.isfinite(encoder(images.cuda())).all()
        torch.cuda.synchronize()
        elapsed = time.monotonic() - started
        for labels in (1, 5):
            observations['%s|%d' % (target, labels)]['test_seconds'] = elapsed
        del train_set, test_set
    atomic_json(HERE / 'probe_calibration.json', dict(complete=True, gpu=torch.cuda.get_device_name(), observations=observations))
    print('PROBE TIMING COMPLETE:', json.dumps(observations), flush=True)


if __name__ == '__main__':
    allocated()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if sys.argv[1] == '--calibrate':
        calibrate(Path(sys.argv[2]).resolve())
    else:
        run(sys.argv[1], Path(sys.argv[2]).resolve(), int(sys.argv[3]))
