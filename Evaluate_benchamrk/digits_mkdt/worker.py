"""MKDT digit prerequisites: original Barlow teacher, feature targets, KD experts."""
import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
BENCH, ROOT = HERE.parent, HERE.parent.parent
sys.path[:0] = [str(ROOT), str(BENCH)]

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from common import (atomic_json, atomic_save, code_hashes, digest_json, exclusive_run,
                    file_hash, rng_state, restore_rng, seed_all, state_hash)
from data import ImageView
import train_teacher as original_teacher
import buffer as original_buffer
import sort_dset as original_sort


def module_at(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


digit_data = module_at('mkdt_digit_data', BENCH / 'digits_simclr/datasets.py')
previous_mkdt = module_at('previous_mkdt', BENCH / 'mkdt_02_jobs/worker.py')


def read(path):
    return json.loads(Path(path).read_text())


def hashes():
    files = [Path(__file__), BENCH / 'digits_simclr/datasets.py',
             BENCH / 'mkdt_02_jobs/worker.py']
    files += [ROOT / name for name in ('train_teacher.py', 'buffer.py', 'sort_dset.py',
                                      'utils.py', 'reparam_module.py', 'more_dataset/__init__.py')]
    return {**code_hashes(), **{str(path.relative_to(ROOT)): file_hash(path) for path in files}}


def recipe():
    saved = read(HERE / 'recipe.json')
    if saved['code_hashes'] != hashes():
        raise ValueError('MKDT source changed; refusing to mix old and new prerequisites')
    return saved


def raw_data(source, data_root):
    saved = recipe()
    if str(data_root.resolve()) != saved['data_root']:
        raise ValueError('Different MKDT data root')
    raw = digit_data.DigitDataset(data_root, source, True)
    if raw.fingerprint != saved['datasets'][source]['train_fingerprint']:
        raise ValueError('Training images/order changed')
    return raw, saved['normalization'][source]


class NormalizedDigits(Dataset):
    def __init__(self, raw, normalization):
        self.raw = raw
        self.mean, self.std = (torch.tensor(x).view(3, 1, 1) for x in normalization)

    def __len__(self):
        return len(self.raw)

    def __getitem__(self, index):
        image = self.raw.images[index].float().div(255)
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)
        return (image - self.mean) / self.std, int(self.raw.targets[index])


def source_identity(source):
    return dict(recipe_sha256=digest_json(recipe()), source=source)


def teacher(source, data_root, stop=1000):
    identity = source_identity(source)
    raw, normalization = raw_data(source, data_root)
    output = HERE / 'teachers' / source
    with exclusive_run(output):
        config = dict(identity=identity, recipe=recipe()['teacher'])
        if (output / 'config.json').exists() and read(output / 'config.json') != config:
            raise ValueError('Different saved MKDT teacher settings')
        atomic_json(output / 'config.json', config)
        final = output / ('barlow_twins_resnet18_%s.pt' % source.lower())
        if (output / 'complete.json').exists():
            marker = read(output / 'complete.json')
            if marker['identity'] != identity or marker['file_sha256'] != file_hash(final):
                raise ValueError('Teacher completion/checksum mismatch')
            return
        cfg = recipe()['teacher']
        seed_all(0)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        # Reuse the historical MKDT model, projector, objective and CIFAR view recipe.
        model = original_teacher.BarlowTwins(source, cfg['proj_dim'], cfg['lambd']).cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
        views = ImageView(raw, original_teacher.TwoViewTransform(*normalization))
        stream = DataLoader(original_teacher.ViewsOnlyDataset(views), batch_size=256,
                            shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
        start = 0
        checkpoint = output / 'training.pt'
        if checkpoint.exists():
            saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
            if saved['identity'] != identity or saved['steps_per_epoch'] != len(stream):
                raise ValueError('MKDT teacher resume identity/step count differs')
            model.load_state_dict(saved['model'], strict=True)
            optimizer.load_state_dict(saved['optimizer'])
            start = saved['epoch']
            restore_rng(saved['rng'])
            print('RESUME Barlow teacher %s: %d/1000 epochs' % (source, start), flush=True)
        assert 0 <= start <= 1000 and stop in (1, 1000)
        with (output / 'ssl.jsonl').open('a', buffering=1) as log:
            for epoch in range(start, stop):
                model.train()
                began = time.monotonic()
                total_loss = seen = 0
                for batch, (first, second) in enumerate(stream):
                    step = epoch * len(stream) + batch + 1
                    lr = original_teacher.learning_rate_at_step(
                        step, 1000 * len(stream), 10 * len(stream), cfg['lr'])
                    optimizer.param_groups[0]['lr'] = lr
                    loss = model(first.cuda(non_blocking=True), second.cuda(non_blocking=True))
                    if not torch.isfinite(loss):
                        raise RuntimeError('Non-finite original-recipe Barlow loss')
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
                        raise RuntimeError('Non-finite Barlow gradient')
                    optimizer.step()
                    total_loss += float(loss.detach()) * len(first)
                    seen += len(first)
                    if step == 1 or step % 200 == 0:
                        print('BARLOW %s epoch=%d/1000 step=%d loss=%.6f lr=%.6g' % (
                            source, epoch + 1, step, total_loss / seen, lr), flush=True)
                row = dict(epoch=epoch + 1, steps=(epoch + 1) * len(stream),
                           loss=total_loss / seen, seconds=time.monotonic() - began, lr=lr)
                print(json.dumps(row), file=log)
                if (epoch + 1) % 50 == 0 or epoch + 1 == stop:
                    atomic_save(checkpoint, dict(identity=identity, epoch=epoch + 1,
                        steps_per_epoch=len(stream), model=model.state_dict(),
                        optimizer=optimizer.state_dict(), rng=rng_state()))
        if stop == 1000:
            backbone = model.backbone.cpu().eval()
            atomic_save(final, backbone.state_dict())
            atomic_json(output / 'complete.json', dict(complete=True, identity=identity, epochs=1000,
                encoder_sha256=state_hash(backbone), file_sha256=file_hash(final)))
            print('SAVED MKDT BARLOW TEACHER:', final, flush=True)


def target(source):
    return HERE / 'target_rep' / 'barlow_twins' / (source + '_target_rep_train.pt')


def validate_target(source):
    path = target(source)
    marker = read(path.with_suffix('.json'))
    teacher_marker = read(HERE / 'teachers' / source / 'complete.json')
    if (marker['identity'] != source_identity(source)
            or marker['teacher_sha256'] != teacher_marker['encoder_sha256']
            or marker['file_sha256'] != file_hash(path)):
        raise ValueError('Wrong or changed MKDT feature targets')
    return marker


def features(source, data_root):
    raw, normalization = raw_data(source, data_root)
    path = target(source)
    with exclusive_run(path.parent / source):
        if path.with_suffix('.json').exists():
            validate_target(source)
            return
        output = HERE / 'teachers' / source
        marker = read(output / 'complete.json')
        final = output / ('barlow_twins_resnet18_%s.pt' % source.lower())
        if (not marker['complete'] or marker['epochs'] != 1000
                or marker['identity'] != source_identity(source) or marker['file_sha256'] != file_hash(final)):
            raise ValueError('Need the finalized matching Barlow teacher')
        model = original_teacher.build_backbone(source)
        model.load_state_dict(torch.load(final, map_location='cpu', weights_only=True), strict=True)
        if state_hash(model) != marker['encoder_sha256']:
            raise ValueError('Teacher weights differ')
        model.cuda().eval().requires_grad_(False)
        stream = DataLoader(NormalizedDigits(raw, normalization), batch_size=256,
                            shuffle=False, num_workers=0, pin_memory=True)
        labels = torch.empty(len(raw), 512)
        position = 0
        with torch.no_grad():
            for images, _ in stream:
                values = model(images.cuda(non_blocking=True)).cpu()
                assert values.shape == (len(images), 512) and torch.isfinite(values).all()
                labels[position:position + len(values)] = values
                position += len(values)
        assert position == len(raw) and torch.isfinite(labels).all()
        assert state_hash(model) == marker['encoder_sha256']
        atomic_save(path, labels)
        atomic_json(path.with_suffix('.json'), dict(complete=True, identity=source_identity(source),
            teacher_sha256=marker['encoder_sha256'], file_sha256=file_hash(path),
            shape=list(labels.shape), training_fingerprint=raw.fingerprint, normalization=normalization))
        print('SAVED MKDT REAL FEATURE TARGETS:', path, flush=True)


def experts(source, start, data_root):
    if start not in range(0, 100, 5):
        raise ValueError('Expert shards must start at 0,5,...,95')
    feature_info = validate_target(source)
    output = HERE / 'work' / source
    with exclusive_run(output / ('_shard_%03d' % start)):
        marker = output / ('shard_%03d.json' % start)
        if marker.exists():
            previous_mkdt.validate_shard(output, start)
            if read(marker)['target_file_sha256'] != feature_info['file_sha256']:
                raise ValueError('Expert shard used different feature targets')
            return
        raw, normalization = raw_data(source, data_root)
        normalized = NormalizedDigits(raw, normalization)

        def dataset_for(dataset, data_path):
            if dataset != source or Path(data_path).resolve() != data_root.resolve():
                raise ValueError('Wrong dataset passed to original MKDT functions')
            return 3, (32, 32), 10, normalized, None

        bank = output / 'buffers' / source / 'ConvNet'
        for index in range(start, start + 5):
            path = bank / ('replay_buffer_%d.pt' % index)
            if path.exists():
                previous_mkdt.validate_buffer(path)
        # Only the dataset reader changes. The original expert and scoring functions run unchanged.
        original_buffer.get_dataset = dataset_for
        original_buffer.main(argparse.Namespace(dataset=source, model='ConvNet', num_experts=5,
            lr_teacher=0.1, batch_train=256, data_path=str(data_root), buffer_path=str(output / 'buffers'),
            train_epochs=20, mom=0.9, l2=1e-4, train_labels_path=str(target(source)), criterion='mse',
            device=0, start_index=start, seed=0, stream_data=True, workers=0, progress_percent_step=0))
        for index in range(start, start + 5):
            previous_mkdt.validate_buffer(bank / ('replay_buffer_%d.pt' % index))
        original_sort.get_dataset = dataset_for
        losses = output / ('losses_%03d.pt' % start)
        original_sort.main(argparse.Namespace(dataset=source, model='ConvNet', num_buffers=5,
            buffer_start=start, loss_output=str(losses), stream_data=True, workers=0,
            ssl_algo='barlow_twins', batch_train=256, data_path=str(data_root), device='cuda:0',
            train_labels_path=str(target(source)), buffer_dir=str(bank),
            buffer_path=str(output / 'buffers'), output_dir=None, result_dir=str(HERE / 'target_rep')))
        values = torch.load(losses, map_location='cpu', weights_only=True)
        assert values.shape == (len(raw),) and torch.isfinite(values).all()
        paths = [bank / ('replay_buffer_%d.pt' % i) for i in range(start, start + 5)] + [losses]
        atomic_json(marker, dict(complete=True, identity=source_identity(source),
            target_file_sha256=feature_info['file_sha256'],
            files={str(path.relative_to(output)): file_hash(path) for path in paths}))


def initialize(source):
    output = HERE / 'work' / source
    with exclusive_run(output / '_initialization'):
        feature_info = validate_target(source)
        count = digit_data.COUNTS[source][0]
        losses = torch.zeros(count, dtype=torch.float64)
        for start in range(0, 100, 5):
            previous_mkdt.validate_shard(output, start)
            shard = read(output / ('shard_%03d.json' % start))
            if (shard['identity'] != source_identity(source)
                    or shard['target_file_sha256'] != feature_info['file_sha256']):
                raise ValueError('Expert shard recipe differs')
            values = torch.load(output / ('losses_%03d.pt' % start), map_location='cpu', weights_only=True)
            assert values.shape == (count,) and torch.isfinite(values).all()
            losses += values / 20
        order = np.argsort(losses.numpy())[::-1].copy()
        files = {}
        for size in (0.2, 1, 2):
            indices = order[:int(count * size / 100)].copy()
            assert len(indices) == len(np.unique(indices)) and len(indices) >= 2
            path = output / ('%s_barlow_twins_%g_high_loss_indices.pkl' % (source, size))
            temporary = path.with_suffix('.tmp.pkl')
            with temporary.open('wb') as stream:
                pickle.dump(indices, stream)
            temporary.replace(path)
            files[path.name] = file_hash(path)
        atomic_json(output / 'init_complete.json', dict(complete=True, identity=source_identity(source),
            expert_count=100, expert_epochs=20, sizes=[0.2, 1, 2], files=files))
        print('MKDT PREREQUISITES READY:', source, flush=True)


def prepare(args):
    prepared = read(BENCH / 'digits_simclr/prepared.json')
    cfg = dict(data_root=str(args.data_root.resolve()), code_hashes=hashes(),
        datasets=prepared['datasets'], normalization={s: prepared['protocol']['normalization'][s]
            for s in digit_data.COUNTS}, teacher=dict(original_teacher.PAPER_DEFAULTS),
        experts=dict(model='ConvNet', normalization='original_MKDT_instancenorm', count=100,
            epochs=20, batch_size=256, optimizer='sgd', lr=0.1, momentum=0.9, weight_decay=1e-4,
            loss='mse', snapshots='initial_plus_each_epoch', seeds=list(range(100))))
    with exclusive_run(HERE / 'preparation'):
        if (HERE / 'recipe.json').exists() and read(HERE / 'recipe.json') != cfg:
            raise ValueError('Existing MKDT prerequisite recipe differs')
        atomic_json(HERE / 'recipe.json', cfg)
        timings = {}
        for source, (count, _) in digit_data.COUNTS.items():
            # This is the first PRODUCTION epoch, with the full 1000-epoch LR schedule.
            # Its checkpoint is resumed by the teacher job: no discarded calibration run.
            teacher(source, args.data_root, stop=1)
            log = HERE / 'teachers' / source / 'ssl.jsonl'
            first = json.loads(log.read_text().splitlines()[0])
            state = torch.load(HERE / 'teachers' / source / 'training.pt', map_location='cpu', weights_only=True)
            timings[source] = dict(teacher=math.ceil((1000 - state['epoch']) * first['seconds'] + 180),
                features=180, experts=math.ceil(5 * 164 * count / 50000 + 60), init=240,
                teacher_epoch_seconds=first['seconds'], expert_timing='old CIFAR10 job 1939145 (one new expert + scoring), scaled conservatively')
            del state
        atomic_json(HERE / 'timing.json', timings)
    subprocess.run(['bash', str(args.launcher), '--submit-prepared', str(args.venv), str(args.data_root)], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'teacher', 'features', 'experts', 'init'))
    parser.add_argument('--source', choices=tuple(digit_data.COUNTS))
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--launcher', type=Path)
    parser.add_argument('--venv', type=Path)
    args = parser.parse_args()
    if (not os.environ.get('SLURM_JOB_ID')
            or os.environ.get('SLURM_CLUSTER_NAME', os.environ.get('CC_CLUSTER')) != 'vulcan'):
        raise RuntimeError('Use the supplied Vulcan Slurm launcher')
    if args.stage != 'init' and (not torch.cuda.is_available() or torch.cuda.device_count() != 1):
        raise RuntimeError('Exactly one allocated GPU is required')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # Match train_teacher.py, rather than change the current benchmark's settings.
    torch.backends.cudnn.benchmark = True
    if args.stage == 'prepare':
        if args.launcher is None or args.venv is None:
            parser.error('prepare requires launcher and venv')
        prepare(args)
    else:
        if args.source is None:
            parser.error('A dataset is required')
        if args.stage == 'teacher':
            teacher(args.source, args.data_root)
        elif args.stage == 'features':
            features(args.source, args.data_root)
        elif args.stage == 'experts':
            experts(args.source, args.start, args.data_root)
        else:
            initialize(args.source)


if __name__ == '__main__':
    main()
