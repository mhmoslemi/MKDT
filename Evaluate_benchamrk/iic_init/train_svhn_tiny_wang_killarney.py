"""IIC clustering heads on a frozen full-data SimCLR teacher; no encoder updates."""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

import torch
from torch import nn

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent
sys.path.insert(0, str(BENCH))
from common import (ROOT, atomic_json, atomic_save, digest_json, exclusive_run,
                    file_hash, load_encoder, seed_all, state_hash)
from data import ImageView, RawDataset, loader, probe_transform, ssl_transform
from digits_simclr.datasets import DigitDataset


def iic_losses(first, second):
    """One negative MI per subhead, using IIC's symmetrized batch joint."""
    assert first.shape == second.shape and first.ndim == 3 and len(first) > 1
    joint = torch.einsum('bhc,bhd->hcd', first, second)
    joint = (joint + joint.transpose(1, 2)) / 2
    joint = joint / joint.sum((1, 2), keepdim=True)
    return joint_losses(joint)


def joint_losses(joint):
    tiny = torch.finfo(joint.dtype).tiny
    rows, columns = joint.sum(2, keepdim=True), joint.sum(1, keepdim=True)
    return -(joint * (joint.clamp_min(tiny).log() - rows.clamp_min(tiny).log()
                     - columns.clamp_min(tiny).log())).sum((1, 2))


def probabilities(heads, features):
    return torch.stack([head(features).softmax(1) for head in heads], dim=1)


def check_loss():
    uniform = torch.full((12, 2, 3), 1 / 3)
    perfect = torch.eye(3).repeat(4, 1)[:, None].expand(-1, 2, -1)
    assert torch.allclose(iic_losses(uniform, uniform), torch.zeros(2), atol=1e-6)
    assert torch.allclose(iic_losses(perfect, perfect), torch.full((2,), -math.log(3)), atol=1e-6)
    logits = torch.randn(12, 2, 3, requires_grad=True)
    iic_losses(logits.softmax(2), logits.flip(0).softmax(2)).mean().backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def run(output):
    config = json.loads((output / 'config.json').read_text())
    assert output.is_relative_to(HERE / 'runs')
    assert os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_JOB_ACCOUNT') == config['account']
    assert config['account'] == 'aip-boyuwang'
    cluster = os.environ.get('SLURM_CLUSTER_NAME', os.environ.get('CC_CLUSTER'))
    assert cluster == 'killarney'
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    assert 'L40S' in torch.cuda.get_device_name(0)
    source = config['dataset']
    assert source in ('SVHN', 'TinyImageNet') and config['epochs'] == 200
    assert config['cluster_counts'] == ({'main': 10, 'overcluster': 70} if source == 'SVHN' else {'main': 200, 'overcluster': 1400})
    assert config['batch_size'] == 256 and config['lr'] == 0.0001 and config['subheads'] == 5
    for file, sha in config['code_hashes'].items():
        assert file_hash(ROOT / file) == sha, 'Code changed: ' + file
    teacher = Path(config['teacher']['path'])
    teacher_sha = config['teacher']['file_sha256']
    marker = teacher.parent / 'ssl_complete.json'
    assert file_hash(teacher) == teacher_sha
    assert file_hash(marker) == config['teacher']['marker_sha256']
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    seed_all(config['seed'])
    check_loss()
    encoder, dimension, saved = load_encoder(teacher)
    assert (saved['source'], saved['model'], saved['method'], saved['seed']) == (source, 'ResNet18', 'simclr', config['teacher']['seed'])
    assert saved['encoder_sha256'] == config['teacher']['encoder_sha256']
    assert not encoder.training and all(not p.requires_grad for p in encoder.parameters())

    def unchanged():
        assert not encoder.training and all(p.grad is None for p in encoder.parameters())
        assert state_hash(encoder) == saved['encoder_sha256'], 'Encoder or BatchNorm buffers changed'

    completed = output / 'complete.json'
    if completed.exists():
        done = json.loads(completed.read_text())
        assert done['config_sha256'] == digest_json(config) and done['complete']
        for name, sha in done['outputs'].items():
            assert file_hash(output / name) == sha
        unchanged()
        print('SKIP completed: ' + str(output), flush=True)
        return

    protocol = config['protocol']
    assert saved['normalization'] == protocol['normalization'][source]
    dataset_class = DigitDataset if source == 'SVHN' else RawDataset
    dataset = dataset_class(config['data_root'], source, True)
    assert len(dataset) == config['train_count'] and dataset.fingerprint == config['source_fingerprint']
    stream = loader(ImageView(dataset, ssl_transform(protocol, source, 'simclr')),
                    config['batch_size'], 0, config['seed'], shuffle=True)
    heads = nn.ModuleDict({name: nn.ModuleList([nn.Linear(dimension, count) for _ in range(config['subheads'])])
                           for name, count in config['cluster_counts'].items()}).cuda()
    assert not ({id(p) for p in heads.parameters()} & {id(p) for p in encoder.parameters()})
    optimizer = torch.optim.Adam(heads.parameters(), lr=config['lr'])
    checkpoint = output / 'training.pt'
    start = 0
    if checkpoint.exists():
        previous = torch.load(checkpoint, map_location='cpu', weights_only=True)
        assert previous['config_sha256'] == digest_json(config)
        heads.load_state_dict(previous['heads'], strict=True)
        optimizer.load_state_dict(previous['optimizer'])
        start = previous['epoch']
        assert 0 <= start <= config['epochs']
    print('Frozen teacher: ' + str(teacher), flush=True)
    print('Teacher file SHA256: ' + teacher_sha, flush=True)
    print('%s head-only IIC: %d -> %d epochs; main=%d, overcluster=%d, 5 subheads each; SimCLR views.' %
          (source, start, config['epochs'], config['cluster_counts']['main'], config['cluster_counts']['overcluster']), flush=True)
    # Both independent head families can share the same frozen-feature pass.
    with (output / 'training.jsonl').open('a', buffering=1) as log:
        for epoch in range(start, config['epochs']):
            epoch_seed = config['seed'] * 1000003 + 80000 + epoch
            seed_all(epoch_seed)
            stream.generator.manual_seed(epoch_seed)
            heads.train()
            started, seen = time.monotonic(), 0
            totals = {name: torch.zeros(config['subheads'], device='cuda') for name in heads}
            for (first, second), _unused_labels in stream:
                with torch.no_grad():
                    features = encoder(torch.cat((first, second)).cuda(non_blocking=True))
                    real_first, real_second = features.chunk(2)
                losses = {name: iic_losses(probabilities(group, real_first), probabilities(group, real_second))
                          for name, group in heads.items()}
                loss = torch.cat(list(losses.values())).mean()
                assert torch.isfinite(loss), 'Non-finite IIC loss'
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in heads.parameters())
                optimizer.step()
                for name in heads:
                    totals[name] += losses[name].detach() * len(first)
                seen += len(first)
            assert seen == len(dataset)
            unchanged()
            stats = dict(epoch=epoch + 1, seconds=time.monotonic() - started,
                         subhead_mi={name: (-value / seen).tolist() for name, value in totals.items()})
            print(json.dumps(stats, allow_nan=False), file=log)
            print('IIC ' + json.dumps(stats, allow_nan=False), flush=True)
            atomic_save(checkpoint, dict(config_sha256=digest_json(config), epoch=epoch + 1,
                        heads=heads.state_dict(), optimizer=optimizer.state_dict()))

    heads.eval()
    seed_all(config['seed'] + 99000)
    stream = loader(ImageView(dataset, ssl_transform(protocol, source, 'simclr')),
                    config['batch_size'], 0, config['seed'] + 99000)
    joints = {name: torch.zeros(config['subheads'], count, count, device='cuda', dtype=torch.float64)
              for name, count in config['cluster_counts'].items()}
    with torch.no_grad():
        for (first, second), _unused_labels in stream:
            first, second = encoder(torch.cat((first, second)).cuda(non_blocking=True)).chunk(2)
            for name, group in heads.items():
                q1, q2 = probabilities(group, first).double(), probabilities(group, second).double()
                joints[name] += torch.einsum('bhc,bhd->hcd', q1, q2)
    selected, scores = {}, {}
    for name, joint in joints.items():
        joint = (joint + joint.transpose(1, 2)) / 2
        joint = joint / joint.sum((1, 2), keepdim=True)
        scores[name] = -joint_losses(joint).cpu()
        selected[name] = int(scores[name].argmax())

    stream = loader(ImageView(dataset, probe_transform(protocol, source, False)),
                    config['batch_size'], 0, config['seed'])
    predictions = {name: [] for name in heads}
    with torch.no_grad():
        for images, _unused_labels in stream:
            features = encoder(images.cuda(non_blocking=True))
            for name, group in heads.items():
                predictions[name].append(group[selected[name]](features).softmax(1).cpu())
    clusters = {}
    for name, chunks in predictions.items():
        probs = torch.cat(chunks)
        assert probs.shape == (len(dataset), config['cluster_counts'][name]) and torch.isfinite(probs).all()
        confidence, assignments = probs.max(1)
        members = [torch.where(assignments == index)[0] for index in range(probs.shape[1])]
        representatives = [int(indices[confidence[indices].argmax()]) if len(indices) else -1 for indices in members]
        clusters[name] = dict(subhead=selected[name], subhead_mi=scores[name], probabilities=probs,
                              assignments=assignments, confidence=confidence, members=members,
                              representative_indices=torch.tensor(representatives), counts=torch.bincount(assignments, minlength=probs.shape[1]))
        assert sum(len(indices) for indices in members) == len(dataset)
        print('%s cluster counts: %s' % (name, clusters[name]['counts'].tolist()), flush=True)
    unchanged()
    assert file_hash(teacher) == teacher_sha, 'Original teacher file changed'
    assert file_hash(marker) == config['teacher']['marker_sha256']
    atomic_save(output / 'heads.pt', dict(kind='frozen_teacher_iic_heads', complete=True, epoch=config['epochs'],
                config=config, heads={key: value.cpu() for key, value in heads.state_dict().items()},
                selected_subheads=selected, subhead_mi=scores, teacher_unchanged=True))
    atomic_save(output / 'clusters.pt', dict(kind='frozen_teacher_iic_clusters', complete=True, config=config,
                image_indices=torch.arange(len(dataset)), clusters=clusters, teacher_unchanged=True))
    atomic_json(completed, dict(complete=True, epochs=config['epochs'], config_sha256=digest_json(config),
                teacher_file_sha256_before=teacher_sha, teacher_file_sha256_after=file_hash(teacher),
                teacher_encoder_sha256=saved['encoder_sha256'], teacher_unchanged=True,
                selected_subheads=selected, subhead_mi={name: values.tolist() for name, values in scores.items()},
                cluster_sizes={name: value['counts'].tolist() for name, value in clusters.items()},
                outputs={name: file_hash(output / name) for name in ('heads.pt', 'clusters.pt')}))
    print('DONE; original teacher weights and BatchNorm buffers unchanged. Saved: ' + str(output), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    destination = args.output.resolve()
    assert destination.is_relative_to(HERE / 'runs')
    with exclusive_run(destination):
        run(destination)
