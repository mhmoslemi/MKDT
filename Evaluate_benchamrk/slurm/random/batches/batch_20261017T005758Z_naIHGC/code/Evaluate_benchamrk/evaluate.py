"""Train only a fresh linear head; keep encoder parameters AND BN buffers frozen."""

import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Subset

from common import atomic_json, code_hashes, digest_json, load_encoder, seed_all, state_hash
from data import ImageView, label_indices, loader, probe_transform
from train_ssl import cosine_lr


def linear_probe(checkpoint, identity, protocol, source, target, train_set, test_set,
                 percentage, seed, output, workers, device):
    output = Path(output)
    indices = label_indices(train_set, percentage, seed)
    cfg = protocol['evaluation']
    expected = {'identity': identity, 'source': source, 'target': target, 'seed': seed,
                'label_percentage': percentage, 'labeled_indices': indices,
                'train_fingerprint': train_set.fingerprint, 'test_fingerprint': test_set.fingerprint,
                'evaluation': cfg, 'evaluation_code_hashes': code_hashes()}
    if output.exists():
        result = json.loads(output.read_text())
        if result.get('expected') != expected or not result.get('complete'):
            raise ValueError(f'Incompatible or incomplete evaluation file: {output}')
        return result
    encoder, dimension, saved = load_encoder(checkpoint, identity, device)
    initial_hash = state_hash(encoder)
    seed_all(seed + 6000)
    head = nn.Linear(dimension, train_set.classes).to(device)
    nn.init.normal_(head.weight, std=cfg['head_init_std'])
    nn.init.zeros_(head.bias)
    optimizer = torch.optim.SGD(head.parameters(), lr=cfg['lr'], momentum=cfg['momentum'],
                                weight_decay=cfg['weight_decay'])
    selected = Subset(train_set, indices)
    train_loader = loader(ImageView(selected, probe_transform(protocol, source, True)),
                          min(cfg['batch_size'], len(indices)), workers, seed, shuffle=True)
    test_loader = loader(ImageView(test_set, probe_transform(protocol, source, False)),
                         cfg['batch_size'], workers, seed)
    total_steps = len(train_loader) * cfg['epochs']
    step = 0
    started = time.monotonic()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_suffix('.jsonl').open('a', buffering=1) as log:
        for epoch in range(cfg['epochs']):
            epoch_seed = seed * 1000003 + epoch + 60000
            seed_all(epoch_seed)
            train_loader.generator.manual_seed(epoch_seed)
            # Never call train() on a container that contains the backbone.
            encoder.eval()
            head.train()
            loss_sum = seen = correct = 0
            for images, labels in train_loader:
                lr = cosine_lr(step, total_steps, cfg['lr'])
                optimizer.param_groups[0]['lr'] = lr
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                with torch.no_grad():
                    features = encoder(images)
                logits = head(features)
                loss = F.cross_entropy(logits, labels)
                if not torch.isfinite(loss):
                    raise RuntimeError('Non-finite linear probe loss')
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                seen += len(labels)
                loss_sum += float(loss.detach()) * len(labels)
                correct += int((logits.argmax(1) == labels).sum())
                step += 1
            stats = {'epoch': epoch + 1, 'lr': lr, 'loss': loss_sum / seen,
                     'train_accuracy_percent': 100 * correct / seen}
            print(json.dumps(stats, allow_nan=False), file=log)
            if (epoch + 1) % 10 == 0 or epoch + 1 == cfg['epochs']:
                print(f'PROBE {target} lbl={percentage} seed={seed}: {stats}', flush=True)
    # Evaluate once, at the prespecified final epoch. Never pick the best test epoch.
    head.eval()
    correct = total = 0
    per_class_correct = np.zeros(train_set.classes, dtype=np.int64)
    per_class_count = np.zeros_like(per_class_correct)
    feature_sum = feature_square_sum = None
    with torch.no_grad():
        for images, labels in test_loader:
            features = encoder(images.to(device, non_blocking=True))
            if not torch.isfinite(features).all():
                raise RuntimeError('Non-finite encoder features')
            predicted = head(features).argmax(1).cpu()
            hits = (predicted == labels).numpy()
            correct += int(hits.sum())
            total += len(labels)
            np.add.at(per_class_count, labels.numpy(), 1)
            np.add.at(per_class_correct, labels.numpy(), hits.astype(np.int64))
            sums, squares = features.double().sum(0), features.double().square().sum(0)
            feature_sum = sums if feature_sum is None else feature_sum + sums
            feature_square_sum = squares if feature_square_sum is None else feature_square_sum + squares
    final_hash = state_hash(encoder)
    if final_hash != initial_hash or final_hash != saved['encoder_sha256']:
        raise RuntimeError('Encoder parameters or BatchNorm buffers changed during probing')
    present = per_class_count > 0
    label_counts = np.bincount(train_set.targets[indices], minlength=train_set.classes)
    variance = (feature_square_sum / total - (feature_sum / total).square()).clamp_min(0)
    result = {'complete': True, 'expected': expected, 'encoder_sha256': initial_hash,
              'encoder_unchanged': True, 'accuracy_percent': 100 * correct / total,
              'macro_accuracy_percent': float(100 * (per_class_correct[present] / per_class_count[present]).mean()),
              'labeled_count': len(indices), 'labeled_class_counts': label_counts.tolist(),
              'labeled_classes_present': int((label_counts > 0).sum()),
              'total_classes': train_set.classes, 'test_count': total,
              'mean_feature_std': float(variance.sqrt().mean()),
              'near_constant_feature_fraction': float((variance < 1e-10).float().mean()),
              'seconds': time.monotonic() - started}
    atomic_json(output, result)
    print(f'RESULT {target} lbl={percentage} seed={seed}: {result["accuracy_percent"]:.4f}', flush=True)
    return result


def main():
    """Re-probe a retained encoder without re-running SSL."""
    import argparse
    from common import HERE, ROOT
    from data import RawDataset, TARGETS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--target', choices=TARGETS, required=True)
    parser.add_argument('--labels', type=int, choices=(1, 5, 100), required=True)
    parser.add_argument('--data-root', type=Path, default=ROOT.parent / 'data')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to(HERE):
        parser.error('Place evaluation output inside Evaluate_benchamrk')
    config = json.loads((args.checkpoint.parent / 'config.json').read_text())
    signature = config['signature']
    if digest_json(signature) != config['identity']['signature']:
        raise ValueError('Checkpoint config content does not match its saved signature')
    if args.workers < 0 or args.threads < 1:
        parser.error('Invalid worker/thread count')
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    train_set, test_set = RawDataset(args.data_root, args.target, True), RawDataset(args.data_root, args.target, False)
    linear_probe(args.checkpoint, config['identity'], signature['protocol'], signature['source'], args.target,
                 train_set, test_set, args.labels, config['identity']['seed'], args.output, args.workers, args.device)


if __name__ == '__main__':
    main()
