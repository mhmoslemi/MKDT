"""One full-data, from-scratch A²MIM ResNet18BN teacher; automatic resume."""
import argparse
import json
import os
from pathlib import Path
import signal
import sys
import time

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent
sys.path.insert(0, str(BENCH))

import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms as T

from common import (atomic_json, atomic_save, digest_json, exclusive_run,
                    file_hash, load_encoder, restore_rng, rng_state, seed_all, state_hash)
from data import ImageView, RawDataset, SOURCE_COUNTS
from a2mim_teacher.model import A2MIM, block_mask, learning_rate, optimizer_for, reconstruction_loss


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=list(SOURCE_COUNTS), required=True)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--output-root', default=str(HERE / 'checkpoints'))
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise RuntimeError('Run in the supplied single-GPU Slurm job')
    if args.epochs <= 10 or args.batch_size < 2:
        parser.error('epochs must exceed warmup=10; batch size must be >=2')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    seed_all(args.seed)
    protocol = json.loads((BENCH / 'protocol.json').read_text())
    size = protocol['input_size'][args.dataset]
    normalization = protocol['normalization'][args.dataset]
    dataset = RawDataset(args.data_root, args.dataset, True)
    if len(dataset) != SOURCE_COUNTS[args.dataset]:
        raise ValueError('Expected the entire source training split')
    transform = T.Compose([
        T.RandomResizedCrop(size, scale=(0.8, 1.0), ratio=(3 / 4, 4 / 3),
                            interpolation=T.InterpolationMode.BILINEAR),
        T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize(*normalization)])
    generator = torch.Generator()
    stream = DataLoader(ImageView(dataset, transform), batch_size=args.batch_size,
                        shuffle=True, drop_last=False, num_workers=0, pin_memory=True,
                        generator=generator)
    patch_size = 8  # Native backbone stride=8; 4x4 CIFAR / 8x8 Tiny masking blocks.
    hashes = {str(p.relative_to(BENCH.parent)): file_hash(p) for p in (
        HERE / 'model.py', HERE / 'train.py', BENCH / 'common.py', BENCH / 'data.py',
        BENCH.parent / 'networks.py', BENCH.parent / 'benchmarks/downstream_data.py')}
    signature = {'method': 'a2mim', 'model': 'ResNet18', 'source': args.dataset,
        'training_split_fingerprint': dataset.fingerprint, 'count': len(dataset),
        'seed': args.seed, 'epochs': args.epochs, 'batch_size': args.batch_size,
        'input_size': size, 'normalization': normalization, 'pretrained': False,
        'encoder_stride': 8, 'mask_patch_size': patch_size, 'mask_ratio': 0.6,
        'feature_mask_stage': 'before_layer4', 'feature_mask_channels': 256,
        'mask_token_init_std': 0.02, 'mask_gamma_init': 1e-6, 'mask_token_lr_mult': 0.1,
        'pixel_loss': 'official_focal_l1_alpha0.2_gamma1_sigmoid',
        'frequency_loss': 'official_FocalFrequencyLoss_ave_spectrum_log_matrix_batch_matrix',
        'frequency_weight': 0.5, 'fft_unmask_replace': 'mixed', 'fft_unmask_weight': 0.001,
        'optimizer': 'adamw', 'weight_decay': 0.05, 'betas': [0.9, 0.999], 'eps': 1e-8,
        'peak_lr': 1.2e-3 * args.batch_size / 2048, 'minimum_lr': 1e-5,
        'schedule': 'official_StepFixCosineAnnealing', 'warmup_epochs': 10,
        'precision': 'float32', 'checkpoint_every_epochs': 10,
        'augmentation': 'random_resized_crop_0.8_1.0_bilinear_and_flip',
        'code_hashes': hashes, 'reference': json.loads((HERE / 'sources.json').read_text())}
    identity = {'signature': digest_json(signature), 'seed': args.seed,
                'selection': {'scope': 'full'}, 'selection_method': 'full'}
    folder = (Path(args.output_root).resolve() / args.dataset / 'ResNet18/full/pt100/a2mim'
              / ('seed_%03d' % args.seed))
    with exclusive_run(folder):
        config = {'identity': identity, 'signature': signature}
        config_path = folder / 'config.json'
        if config_path.exists() and json.loads(config_path.read_text()) != config:
            raise ValueError('Saved run configuration differs; use a new output directory')
        atomic_json(config_path, config)
        if (folder / 'encoder.pt').exists():
            encoder, _, metadata = load_encoder(folder / 'encoder.pt', identity, 'cpu')
            if metadata['method'] != 'a2mim' or metadata['epochs'] != args.epochs:
                raise ValueError('Existing final checkpoint is not this A²MIM run')
            print('Already complete:', folder / 'encoder.pt', flush=True)
            return
        atomic_json(folder / 'environment.json', {'torch': str(torch.__version__),
            'torchvision': str(torchvision.__version__), 'python': sys.version,
            'gpu': torch.cuda.get_device_name(), 'job_id': os.environ['SLURM_JOB_ID'],
            'account': os.environ.get('SLURM_JOB_ACCOUNT'), 'workers': 0})
        model = A2MIM(size).cuda()
        optimizer = optimizer_for(model, args.batch_size)
        checkpoint = folder / 'training.pt'
        start_epoch = start_batch = steps = 0
        elapsed = 0.0
        resumed_rng = None
        totals = [0.0, 0.0, 0.0, 0]
        if checkpoint.exists():
            saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
            if saved['identity'] != identity or saved['steps_per_epoch'] != len(stream):
                raise ValueError('Resume checkpoint belongs to a different recipe/data split')
            model.load_state_dict(saved['model'], strict=True)
            optimizer.load_state_dict(saved['optimizer'])
            start_epoch, start_batch, steps = saved['epoch'], saved['batch'], saved['steps']
            elapsed, totals, resumed_rng = saved['training_wall_seconds'], saved['totals'], saved['rng']
            print('RESUME: epoch=%d, next batch=%d, total epochs=%d' % (
                start_epoch, start_batch, args.epochs), flush=True)
        stop = False

        def request_stop(signum, frame):
            nonlocal stop
            stop = True

        signal.signal(signal.SIGUSR1, request_stop)
        signal.signal(signal.SIGTERM, request_stop)
        signal.signal(signal.SIGINT, request_stop)
        started = time.monotonic()

        def save(epoch, batch, totals):
            atomic_save(checkpoint, {'identity': identity, 'model': model.state_dict(),
                'optimizer': optimizer.state_dict(), 'epoch': epoch, 'batch': batch,
                'steps': steps, 'steps_per_epoch': len(stream), 'rng': rng_state(),
                'totals': totals, 'training_wall_seconds': elapsed + time.monotonic() - started})

        print('A2MIM %s: %d images, %d epochs, ResNet18BN, batch=%d, mask=0.6, patch=%d' % (
            args.dataset, len(dataset), args.epochs, args.batch_size, patch_size), flush=True)
        with (folder / 'ssl.jsonl').open('a', buffering=1) as log:
            for epoch in range(start_epoch, args.epochs):
                epoch_seed = args.seed * 1000003 + epoch + 10000
                seed_all(epoch_seed)
                generator.manual_seed(epoch_seed)
                model.train()
                epoch_started = time.monotonic()
                if epoch != start_epoch or not start_batch:
                    totals = [0.0, 0.0, 0.0, 0]
                for batch, (images, _unused_labels) in enumerate(stream):
                    # Regenerate the same shuffle/augmentations up to a partial save.
                    if epoch == start_epoch and batch < start_batch:
                        if batch + 1 == start_batch:
                            restore_rng(resumed_rng)
                        continue
                    images = images.cuda(non_blocking=True)
                    mask = block_mask(len(images), size, patch_size, images.device)
                    lr = learning_rate(steps, len(stream), args.epochs, signature['peak_lr'])
                    for group in optimizer.param_groups:
                        # MMCV applies its absolute minimum LR to every param group.
                        group['lr'] = learning_rate(steps, len(stream), args.epochs,
                                                   signature['peak_lr'] * group['lr_mult'])
                    optimizer.zero_grad(set_to_none=True)
                    prediction = model(images, mask)
                    loss, spatial, frequency = reconstruction_loss(prediction, images, mask)
                    if not bool(torch.isfinite(loss)):
                        raise FloatingPointError('Nonfinite A²MIM loss at step %d' % steps)
                    loss.backward()
                    optimizer.step()
                    steps += 1
                    values = [float(v.detach()) for v in (loss, spatial, frequency)]
                    for i, value in enumerate(values):
                        totals[i] += value * len(images)
                    totals[3] += len(images)
                    if steps % 200 == 0:
                        print('STEP %d: loss=%.6f spatial=%.6f frequency=%.6f lr=%.3e' % (
                            steps, *values, lr), flush=True)
                    if stop:
                        if batch + 1 == len(stream):
                            save(epoch + 1, 0, [0.0, 0.0, 0.0, 0])
                        else:
                            save(epoch, batch + 1, totals)
                        print('Saved resumable epoch %d, batch %d: %s' % (
                            epoch, batch + 1, checkpoint), flush=True)
                        raise SystemExit(75)
                row = {'epoch': epoch + 1, 'steps': steps, 'lr': lr,
                    'loss': totals[0] / totals[3], 'spatial': totals[1] / totals[3],
                    'frequency': totals[2] / totals[3], 'examples': totals[3],
                    'seconds': time.monotonic() - epoch_started}
                log.write(json.dumps(row) + '\n')
                if (epoch + 1) % 10 == 0 or epoch + 1 == args.epochs:
                    save(epoch + 1, 0, [0.0, 0.0, 0.0, 0])
                    print('EPOCH %d/%d: loss=%.6f; saved %s' % (
                        epoch + 1, args.epochs, row['loss'], checkpoint), flush=True)
                start_batch = 0
        encoder = model.encoder.cpu().eval()
        metadata = {'kind': 'ssl_reference_encoder', 'complete': True,
            'identity': identity, 'source': args.dataset, 'model': 'ResNet18',
            'method': 'a2mim', 'seed': args.seed, 'input_size': size,
            'normalization': normalization, 'epochs': args.epochs, 'steps': steps,
            'effective_batch': args.batch_size, 'encoder_sha256': state_hash(encoder)}
        atomic_save(folder / 'encoder.pt', {**metadata, 'encoder': encoder.state_dict()})
        verified, _, _ = load_encoder(folder / 'encoder.pt', identity, 'cpu')
        atomic_json(folder / 'ssl_complete.json', {**metadata,
            'checkpoint_file_sha256': file_hash(folder / 'encoder.pt'),
            'training_wall_seconds': elapsed + time.monotonic() - started})
        print('COMPLETE: verified frozen teacher:', folder / 'encoder.pt', flush=True)


if __name__ == '__main__':
    main()
