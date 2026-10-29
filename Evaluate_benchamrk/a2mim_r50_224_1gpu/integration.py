"""Original upstream A2MIM training; adapt only input source, one-GPU execution and outputs."""
import hashlib
import math
import os
from pathlib import Path
import signal

import numpy as np
from PIL import Image
import torch
import torchvision
from mmcv import Config
from mmcv.runner import Hook
from openmixup.core.hooks import HOOKS
from openmixup.datasets.registry import DATASOURCES

from contracts import HERE, ROOT, UPSTREAM, atomic_json, atomic_save, digest_json, file_hash, folder, load_unit, read, state_hash

stop_requested = False


def stop(signum, frame):
    global stop_requested
    stop_requested = True


@DATASOURCES.register_module()
class SavedBenchmarkSource:
    def __init__(self, unit_file, data_root):
        unit = load_unit(unit_file)
        self.return_label = False
        self.unit = unit
        source, kind = unit['source'], unit['kind']
        raw = getattr(torchvision.datasets, source)(data_root, train=True, download=False)
        self.CLASSES = raw.classes
        raw_fingerprint = digest_json(dict(name=source, training=True, labels=list(raw.targets),
            pixels_sha256=hashlib.sha256(raw.data.tobytes()).hexdigest()))
        if raw_fingerprint != unit['input']['source_fingerprint']:
            raise ValueError('Training split differs from the existing benchmark')
        self.images = None
        self.raw = raw
        self.indices = None
        count = 50000 if kind == 'full' else int(50000 * unit['subset'] / 100)
        if kind == 'random':
            self.indices = unit['input']['indices']
            if (len(self.indices) != count or len(set(self.indices)) != count
                    or any(not 0 <= i < 50000 for i in self.indices)):
                raise ValueError('Invalid saved random subset')
        elif kind in ('mkdt', 'ours'):
            spec = unit['input']
            path = ROOT / spec['path']
            if file_hash(path) != spec['file_sha256']:
                raise ValueError('Synthetic data changed after submission')
            saved = torch.load(path, map_location='cpu', weights_only=True)
            if kind == 'ours':
                if (not saved['complete'] or saved['iteration'] != {0.2: 10000, 1: 8000}[unit['subset']]
                        or saved['dataset'] != source or saved['model'] != 'ResNet18BN'
                        or saved['percentage'] != unit['subset'] or saved['normalization'] != spec['normalization']):
                    raise ValueError('Wrong original MIPA synthetic set')
                images = saved['data'].detach().cpu().contiguous()
                if hashlib.sha256(images.numpy().tobytes()).hexdigest() != spec['images_sha256']:
                    raise ValueError('MIPA image values differ from the saved benchmark')
            else:
                images = saved.detach().cpu().contiguous()
            if images.shape != (count, 3, 32, 32) or not images.is_floating_point() or not torch.isfinite(images).all():
                raise ValueError('Invalid synthetic tensor')
            mean, std = (torch.tensor(v).view(1, 3, 1, 1) for v in spec['normalization'])
            pixels = images * std + mean
            self.images = pixels.clamp(0, 1).mul(255).to(torch.uint8).permute(0, 2, 3, 1).contiguous().numpy()
            self.raw = None
        self.count = count
        self.labels = [-1] * count  # Required dataset metadata; SSL never consumes labels.

    def get_length(self):
        return self.count

    def get_sample(self, index):
        if self.images is not None:
            return Image.fromarray(self.images[index])
        return self.raw[index if self.indices is None else self.indices[index]][0]


def config_for(unit_file, data_root):
    unit = load_unit(unit_file)
    cfg = Config.fromfile(str(UPSTREAM / 'configs/selfsup/a2mim/cifar100/r50_l3_sz224_init_4xb256_accu2_cos_ep1000.py'))
    assert cfg.model.backbone.depth == 50 and cfg.optimizer.lr == 0.0012
    assert cfg.data.imgs_per_gpu == 64 and cfg.optimizer_config.update_interval == 2
    assert cfg.use_fp16 and cfg.model.head.fft_weight == 0.5
    cfg.model.backbone.norm_cfg = dict(type='BN', requires_grad=True)
    cfg.dist_params.backend = 'gloo'  # One rank only; avoid this legacy NCCL build's native startup crash.
    cfg.optimizer_config.update_interval = 8
    cfg.data.workers_per_gpu = 0
    cfg.data.drop_last = False
    cfg.persistent_workers = False
    cfg.runner.max_epochs = unit['epochs']
    cfg.data.train.data_source = dict(type='SavedBenchmarkSource', unit_file=str(Path(unit_file).resolve()), data_root=str(data_root))
    cfg.custom_hooks = [dict(type='StopExportA2MIMHook', unit_file=str(Path(unit_file).resolve()))]
    cfg.checkpoint_config = dict(interval=10 if unit['kind'] == 'full' else 500,
                                 max_keep_ckpts=1, save_optimizer=True)
    cfg.log_config.interval = 200
    cfg.seed = 0
    assert cfg.data.train.pipeline[0]['size'] == 224 and cfg.data.train.pipeline[0]['scale'] == (0.67, 1.0)
    mask = cfg.data.train.mask_pipeline[0]
    assert mask['mask_patch_size'] == 32 and mask['model_patch_size'] == 16 and mask['mask_ratio'] == 0.6
    return cfg, unit


@HOOKS.register_module()
class StopExportA2MIMHook(Hook):
    def __init__(self, unit_file, dist_mode=True):
        self.unit = load_unit(unit_file)
        for sig in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, stop)

    def after_train_iter(self, runner):
        if not np.isfinite(runner.outputs['log_vars']['loss']):
            raise FloatingPointError('Nonfinite upstream A2MIM loss')

    def after_train_epoch(self, runner):
        unit = self.unit
        epoch = runner.epoch + 1
        if epoch == unit['epochs']:
            backbone = runner.model.module.backbone
            state = {k: v.detach().cpu() for k, v in backbone.state_dict().items()
                     if k not in ('mask_token', 'mask_gamma')}
            encoder = torchvision.models.resnet50(weights=None, zero_init_residual=True)
            encoder.fc = torch.nn.Identity()
            encoder.load_state_dict(state, strict=True)
            encoder.eval().requires_grad_(False)
            count = 50000 if unit['kind'] == 'full' else int(50000 * unit['subset'] / 100)
            metadata = dict(kind='ssl_reference_encoder', complete=True, identity=unit['identity'],
                model='ResNet50ImageNet', input_size=224, source=unit['source'], method='a2mim', seed=0,
                epochs=epoch, steps=runner.iter, effective_batch=64,
                normalization=[[0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.201]],
                encoder_sha256=state_hash(encoder), one_gpu_adaptation=True, accumulation=8,
                physical_batch=64, nominal_accumulated_batch=512, microbatches_per_epoch=math.ceil(count / 64),
                last_microbatch=count % 64 or 64, mean_images_per_eight_microbatches=8 * count / math.ceil(count / 64))
            destination = folder(unit) / 'encoder.pt'
            atomic_save(destination, {**metadata, 'encoder': encoder.state_dict()})
            atomic_json(destination.parent / 'ssl_complete.json', {**metadata, 'checkpoint_file_sha256': file_hash(destination)})
            print('COMPLETE upstream A2MIM encoder:', destination, flush=True)
        elif stop_requested:
            runner.save_checkpoint(runner.work_dir, filename_tmpl='epoch_{}.pth', save_optimizer=True)
            print('Stopped after saving epoch %d/%d; rerun the same launcher to resume.' % (epoch, unit['epochs']), flush=True)
            raise SystemExit(75)
