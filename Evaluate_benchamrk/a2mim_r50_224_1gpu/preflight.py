"""Allocated-GPU environment/model check; no published accuracy or teacher is produced."""
import json
import os
from pathlib import Path
import time

import torch
import torchvision
from mmcv import Config
from openmixup.models import build_model
from openmixup.datasets import build_dataset

HERE = Path(__file__).resolve().parent


def main():
    assert os.environ.get('SLURM_JOB_ID') and torch.cuda.is_available()
    assert torch.cuda.device_count() == 1
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = True
    cfg = Config.fromfile(str(HERE / 'upstream/configs/selfsup/a2mim/cifar100/r50_l3_sz224_init_4xb256_accu2_cos_ep1000.py'))
    assert cfg.runner.max_epochs == 1000 and cfg.data.imgs_per_gpu == 64
    assert cfg.optimizer_config.update_interval == 2 and cfg.use_fp16
    assert cfg.model.backbone.depth == 50
    assert cfg.data.train.pipeline[0]['size'] == 224
    assert cfg.data.train.pipeline[0]['scale'] == (0.67, 1.0)
    assert cfg.data.train.mask_pipeline[0]['mask_patch_size'] == 32
    cfg.model.backbone.norm_cfg = dict(type='BN', requires_grad=True)
    cfg.data.train.data_source = dict(type='CIFAR10', root=str(HERE.parents[2] / 'data'), split='train', return_label=False)
    dataset = build_dataset(cfg.data.train)
    batches = iter(torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0))
    model = build_model(cfg.model).cuda().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.optimizer.lr, weight_decay=0.05)
    scaler = torch.cuda.amp.GradScaler()
    original = model.backbone.conv1.weight.detach().clone()
    times = []
    optimizer.zero_grad(set_to_none=True)
    for step in range(12):
        torch.cuda.synchronize()
        started = time.monotonic()
        batch = next(batches)
        masked = batch['img'].cuda()
        mask, images = (value.cuda() for value in batch['mask'])
        with torch.cuda.amp.autocast():
            loss = model(masked, mask=[mask, images], mode='train')['loss']
        assert torch.isfinite(loss), 'Nonfinite official A2MIM loss'
        scaler.scale(loss / 8).backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        if (step + 1) % 8 == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        times.append(time.monotonic() - started)
        print('PREFLIGHT step=%d loss=%.6f seconds=%.4f' % (step, float(loss), times[-1]), flush=True)
    assert not torch.equal(original, model.backbone.conv1.weight)
    encoder = torchvision.models.resnet50(weights=None, zero_init_residual=True)
    encoder.fc = torch.nn.Identity()
    state = {k: v.detach().cpu() for k, v in model.backbone.state_dict().items()
             if k not in ('mask_token', 'mask_gamma')}
    encoder.load_state_dict(state, strict=True)
    encoder.eval()
    assert encoder(torch.randn(2, 3, 224, 224)).shape == (2, 2048)
    report = dict(complete=True, upstream_commit='37e1372446ab35c03fcf77302539bd4a5bee47eb',
        torch=str(torch.__version__), torchvision=str(torchvision.__version__), gpu=torch.cuda.get_device_name(),
        official_model_and_loss=True, finite_gradients=True, encoder_updated=True, strict_torchvision_export=True,
        train_step_seconds=times[3:], peak_gpu_bytes=torch.cuda.max_memory_allocated(),
        physical_batch=64, intended_single_gpu_accumulation=8, input_size=224, epochs_full=1000, epochs_subset=10000)
    temporary = HERE / ('environment_ready.%d.tmp' % os.getpid())
    temporary.write_text(json.dumps(report, indent=2) + '\n')
    temporary.replace(HERE / 'environment_ready.json')
    print('PREFLIGHT PASSED: official ResNet50/224 A2MIM, finite optimizer updates, strict encoder export.', flush=True)


if __name__ == '__main__':
    main()
