"""Short GPU preflight and timing; no experiment checkpoint is trained here."""
import ast
import gc
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent
sys.path.insert(0, str(BENCH))
import torch
from torch import nn
import common
import data
import train_ssl
from models import install


def main():
    assert os.environ.get('SLURM_JOB_ID') and torch.cuda.is_available()
    for path in HERE.glob('*.py'):
        ast.parse(path.read_text(), filename=str(path))
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    install()
    protocol = json.loads((BENCH / 'protocol.json').read_text())
    root = Path(sys.argv[1]).resolve()
    results = {}
    for name in ('ResNet18', 'ResNet34', 'ResNet50'):
        common.seed_all(123)
        model = train_ssl.SSLModel(name, 32, protocol['methods']['simclr']).cuda()
        assert sum(isinstance(m, nn.BatchNorm2d) for m in model.encoder.modules()) > 0
        optimizer = train_ssl.LARS(model.parameters(), 0.4, protocol['ssl'])
        batches = {}
        for count in (100, 256):
            first, second = torch.randn(count, 3, 32, 32, device='cuda'), torch.randn(count, 3, 32, 32, device='cuda')
            times = []
            for step in range(7):
                torch.cuda.synchronize()
                start = time.monotonic()
                optimizer.zero_grad(set_to_none=True)
                a, b = model(first, second)
                loss = train_ssl.simclr_loss(a, b, protocol['methods']['simclr']['temperature'])
                loss.backward()
                optimizer.step()
                torch.cuda.synchronize()
                assert torch.isfinite(loss) and torch.isfinite(a).all()
                if step >= 2:
                    times.append(time.monotonic() - start)
            batches[str(count)] = max(times)
            del first, second, a, b, loss
        encoder = model.encoder.eval().requires_grad_(False)
        frozen_hash = common.state_hash(encoder)
        head = nn.Linear(model.dimension, 10).cuda()
        images = torch.randn(256, 3, 32, 32, device='cuda')
        times = []
        for step in range(7):
            torch.cuda.synchronize()
            start = time.monotonic()
            head.zero_grad(set_to_none=True)
            with torch.no_grad():
                features = encoder(images)
            head(features).square().mean().backward()
            torch.cuda.synchronize()
            if step >= 2:
                times.append(time.monotonic() - start)
        assert frozen_hash == common.state_hash(encoder)
        test_path = Path(os.environ['SLURM_TMPDIR']) / (name + '_cross_arch_check.pt')
        identity = {'preflight': name}
        common.atomic_save(test_path, {'kind': 'ssl_reference_encoder', 'complete': True, 'identity': identity,
            'model': name, 'input_size': 32, 'encoder': encoder.cpu().state_dict(), 'encoder_sha256': frozen_hash})
        loaded, dim, _ = common.load_encoder(test_path, identity, device='cpu')
        assert dim == model.dimension and common.state_hash(loaded) == frozen_hash
        results[name] = {'ssl_batch_seconds': batches, 'probe_batch_seconds': max(times),
                         'feature_dimension': dim, 'bn_only': True, 'strict_reload': True, 'frozen_bn': True}
        print(name, results[name], flush=True)
        del model, optimizer, encoder, head, images, features, loaded
        gc.collect()
        torch.cuda.empty_cache()
    sources = {}
    for source in ('CIFAR10', 'CIFAR100'):
        for size in (0.2, 1):
            path = BENCH.parent / 'OT-SSL-DD/ours_syntehtic' / source / 'ResNet18BN' / ('pt%g' % size) / (
                'res_IIC_%s_ResNet18BN_%gpercent_final.pt' % (source, size))
            saved = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
            assert saved['complete'] and saved['iteration'] == (10000 if size == 0.2 else 8000)
            assert saved['dataset'] == source and saved['model'] == 'ResNet18BN' and saved['percentage'] == size
            assert saved['teacher_source'] == 'pretrained' and saved['data'].shape == (int(50000 * size / 100), 3, 32, 32)
            assert saved['dsa'] and saved['run_args']['disp_weight'] == saved['run_args']['pair_weight'] == 0
            assert torch.isfinite(saved['data']).all()
            sources[str(path.relative_to(BENCH.parent))] = common.file_hash(path)
            del saved
    fingerprints = {}
    for target in ('CIFAR10', 'CIFAR100', 'TinyImageNet'):
        for training in (True, False):
            dataset = data.RawDataset(root, target, training)
            fingerprints['%s_%s' % (target, 'train' if training else 'test')] = dataset.fingerprint
            del dataset
    hashes = common.code_hashes()
    hashes.update({str(p.relative_to(BENCH.parent)): common.file_hash(p)
                   for p in (HERE / 'models.py', HERE / 'calibrate.py', BENCH / 'protocol.json')})
    common.atomic_json(HERE / 'calibration.json', {'complete': True, 'models': results,
        'synthetics': sources, 'fingerprints': fingerprints, 'code_hashes': hashes,
        'gpu': torch.cuda.get_device_name(0), 'job': os.environ['SLURM_JOB_ID']})
    print('PASS: BN architectures, finite SSL gradients, frozen BN, strict checkpoint reload and all four synthetic sets.', flush=True)


if __name__ == '__main__':
    main()
