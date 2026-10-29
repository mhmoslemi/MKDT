"""Subset training with the existing A²MIM model, loss, optimizer and schedule."""
import json
import signal
import time

import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T

from common import atomic_json, atomic_save, file_hash, load_encoder, restore_rng, rng_state, seed_all, state_hash
from data import ImageView
from a2mim_teacher.model import A2MIM, block_mask, learning_rate, optimizer_for, reconstruction_loss


def train(dataset, config, folder):
    signature, identity = config['signature'], config['identity']
    recipe = signature['recipe']
    size, batch_size, epochs = recipe['input_size'], recipe['batch_size'], recipe['epochs']
    destination = folder / 'encoder.pt'
    if destination.exists():
        encoder, _, saved = load_encoder(destination, identity, 'cpu')
        del encoder
        if saved['method'] != 'a2mim' or saved['epochs'] != 2000:
            raise ValueError('Not the completed 2k A²MIM encoder')
        if not (folder / 'ssl_complete.json').exists():
            atomic_json(folder / 'ssl_complete.json', {**{k: v for k, v in saved.items() if k != 'encoder'},
                'checkpoint_file_sha256': file_hash(destination)})
        return
    seed_all(0)
    transform = T.Compose([T.RandomResizedCrop(size, scale=(0.8, 1.0), ratio=(3 / 4, 4 / 3),
        interpolation=T.InterpolationMode.BILINEAR), T.RandomHorizontalFlip(),
        T.ToTensor(), T.Normalize(*recipe['normalization'])])
    generator = torch.Generator()
    stream = DataLoader(ImageView(dataset, transform), batch_size=batch_size, shuffle=True,
                        drop_last=False, num_workers=0, pin_memory=True, generator=generator)
    model = A2MIM(size).cuda()
    optimizer = optimizer_for(model, batch_size)
    checkpoint = folder / 'training.pt'
    start_epoch = start_batch = steps = 0
    elapsed = 0.0
    resumed_rng = None
    totals = [0.0, 0.0, 0.0, 0]
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if saved['identity'] != identity or saved['steps_per_epoch'] != len(stream):
            raise ValueError('Resume checkpoint differs from this subset and recipe')
        model.load_state_dict(saved['model'], strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        start_epoch, start_batch, steps = saved['epoch'], saved['batch'], saved['steps']
        elapsed, totals, resumed_rng = saved['training_wall_seconds'], saved['totals'], saved['rng']
        print('RESUME A2MIM: epoch=%d batch=%d / %d epochs' % (start_epoch, start_batch, epochs), flush=True)
    stop = False

    def request_stop(signum, frame):
        nonlocal stop
        stop = True

    for sig in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, request_stop)
    started = time.monotonic()

    def save(epoch, batch, running):
        atomic_save(checkpoint, dict(identity=identity, model=model.state_dict(),
            optimizer=optimizer.state_dict(), epoch=epoch, batch=batch, steps=steps,
            steps_per_epoch=len(stream), rng=rng_state(), totals=running,
            training_wall_seconds=elapsed + time.monotonic() - started))

    with (folder / 'ssl.jsonl').open('a', buffering=1) as log:
        for epoch in range(start_epoch, epochs):
            epoch_seed = epoch + 10000
            seed_all(epoch_seed)
            generator.manual_seed(epoch_seed)
            model.train()
            epoch_started = time.monotonic()
            if epoch != start_epoch or not start_batch:
                totals = [0.0, 0.0, 0.0, 0]
            for batch, (images, _labels) in enumerate(stream):
                if epoch == start_epoch and batch < start_batch:
                    if batch + 1 == start_batch:
                        restore_rng(resumed_rng)
                    continue
                images = images.cuda(non_blocking=True)
                mask = block_mask(len(images), size, 8, images.device)
                lr = learning_rate(steps, len(stream), epochs, recipe['peak_lr'])
                for group in optimizer.param_groups:
                    group['lr'] = learning_rate(steps, len(stream), epochs,
                                               recipe['peak_lr'] * group['lr_mult'])
                optimizer.zero_grad(set_to_none=True)
                prediction = model(images, mask)
                loss, spatial, frequency = reconstruction_loss(prediction, images, mask)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError('Nonfinite A²MIM loss at step %d' % steps)
                loss.backward()
                finite = torch.stack([torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None]).all()
                if not bool(finite):
                    raise FloatingPointError('Nonfinite A²MIM gradient')
                optimizer.step()
                steps += 1
                for i, value in enumerate((loss, spatial, frequency)):
                    totals[i] += float(value.detach()) * len(images)
                totals[3] += len(images)
                if steps % 200 == 0:
                    print('A2MIM step=%d loss=%.6f spatial=%.6f frequency=%.6f lr=%.3e' %
                          (steps, float(loss.detach()), float(spatial.detach()), float(frequency.detach()), lr), flush=True)
                if stop:
                    if batch + 1 == len(stream):
                        save(epoch + 1, 0, [0.0, 0.0, 0.0, 0])
                    else:
                        save(epoch, batch + 1, totals)
                    raise SystemExit(75)
            log.write(json.dumps(dict(epoch=epoch + 1, steps=steps, lr=lr,
                loss=totals[0] / totals[3], spatial=totals[1] / totals[3],
                frequency=totals[2] / totals[3], examples=totals[3],
                seconds=time.monotonic() - epoch_started)) + '\n')
            if (epoch + 1) % 500 == 0 or epoch + 1 == epochs:
                save(epoch + 1, 0, [0.0, 0.0, 0.0, 0])
            start_batch = 0
    encoder = model.encoder.cpu().eval()
    metadata = dict(kind='ssl_reference_encoder', complete=True, identity=identity,
        source=signature['source'], model='ResNet18', method='a2mim', seed=0,
        input_size=size, normalization=recipe['normalization'], epochs=epochs,
        steps=steps, effective_batch=batch_size, encoder_sha256=state_hash(encoder))
    atomic_save(destination, {**metadata, 'encoder': encoder.state_dict()})
    verified, _, _ = load_encoder(destination, identity, 'cpu')
    del verified
    atomic_json(folder / 'ssl_complete.json', {**metadata,
        'checkpoint_file_sha256': file_hash(destination),
        'training_wall_seconds': elapsed + time.monotonic() - started})
    print('COMPLETE: A2MIM subset encoder:', destination, flush=True)
