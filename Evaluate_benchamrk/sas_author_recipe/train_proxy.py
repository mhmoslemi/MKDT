"""Run the authors' ResNet50, projection head and loss, with resumable checkpoints."""
from functools import partial
import json
import os
from pathlib import Path
import random
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE.parent))
import numpy as np
import torch
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchvision import transforms as T
from common import atomic_json, atomic_save, digest_json, file_hash, seed_all
from data import RawDataset

sys.path.insert(0, str(HERE / 'upstream/sas'))
from resnet import ResNet50, StemCIFAR
from projection_heads.critic import LinearCritic
from data_proc.augmentation import ColourDistortion
import trainer as upstream_trainer


class ProxyViews(Dataset):
    def __init__(self, base, normalization):
        self.base = base
        self.transform = T.Compose([T.RandomResizedCrop(64, interpolation=T.InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(), ColourDistortion(s=0.5), T.ToTensor(), T.Normalize(*normalization)])

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        image, _ = self.base[index]
        return [self.transform(image), self.transform(image)]


def build():
    net = ResNet50(stem=StemCIFAR)
    return net, LinearCritic(net.representation_dim, temperature=0.5)


def rng_state():
    state = np.random.get_state()
    return dict(python=random.getstate(), numpy=(state[0], state[1].tolist(), *state[2:]),
                torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state())


def restore_rng(state):
    random.setstate(state['python'])
    value = state['numpy']
    np.random.set_state((value[0], np.asarray(value[1], dtype=np.uint32), *value[2:]))
    torch.set_rng_state(state['torch'])
    torch.cuda.set_rng_state(state['cuda'])


def main():
    assert os.environ.get('SLURM_JOB_ACCOUNT') == 'aip-yiweilu'
    rank, world = int(os.environ['LOCAL_RANK']), int(os.environ['WORLD_SIZE'])
    assert world == 4 and torch.cuda.device_count() == 4
    torch.cuda.set_device(rank)
    torch.set_num_threads(1)
    dist.init_process_group('nccl', device_id=torch.device('cuda', rank))
    seed_all(0)
    manifest = json.loads((HERE / 'manifest.json').read_text())
    for relative, checksum in manifest['code_hashes'].items():
        assert file_hash(ROOT / relative) == checksum, 'Code changed: ' + relative
    base = RawDataset(manifest['data_root'], 'TinyImageNet', True)
    cfg = manifest['tiny_recipe']
    identity = digest_json(dict(recipe=cfg, source_fingerprint=base.fingerprint,
                                code_hashes=manifest['code_hashes'], world_size=world))
    folder = HERE / 'proxy/TinyImageNet'
    folder.mkdir(parents=True, exist_ok=True)
    checkpoint = folder / 'training.pt'
    net, critic = build()
    net, critic = net.cuda(rank), critic.cuda(rank)
    optimizer = torch.optim.Adam(list(net.parameters()) + list(critic.parameters()),
                                 lr=0.001, weight_decay=1e-6)
    start, restored = 0, None
    if checkpoint.is_file():
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        assert saved['identity'] == identity
        net.load_state_dict(saved['encoder'])
        critic.load_state_dict(saved['critic'])
        optimizer.load_state_dict(saved['optimizer'])
        start, restored = saved['epoch'], saved['rng'][rank]
        del saved
    data = ProxyViews(base, cfg['normalization'])
    sampler = DistributedSampler(data, num_replicas=world, rank=rank, seed=0, shuffle=True, drop_last=True)
    stream = DataLoader(data, batch_size=512 // world, sampler=sampler, num_workers=1, pin_memory=True)
    net = DistributedDataParallel(net, device_ids=[rank])
    upstream_trainer.tqdm = partial(upstream_trainer.tqdm, disable=True)
    trainer = upstream_trainer.Trainer(net=net, critic=critic, trainloader=stream,
        clftrainloader=None, testloader=None, num_classes=200, optimizer=optimizer,
        device=rank, distributed=True, rank=rank, world_size=world)
    if restored is not None:
        restore_rng(restored)
    if rank == 0:
        print(f'Official SAS ResNet50 proxy: resume {start}/400, global batch 512, Adam 0.001, tau 0.5, 4 GPUs', flush=True)
    for epoch in range(start, 400):
        began = time.monotonic()
        loss = trainer.train()
        assert np.isfinite(loss) and all(torch.isfinite(p).all() for p in net.parameters())
        if rank == 0:
            record = dict(epoch=epoch + 1, loss=loss, seconds=time.monotonic() - began)
            print(json.dumps(record), flush=True)
            with (folder / 'training.jsonl').open('a') as log:
                log.write(json.dumps(record) + '\n')
        if epoch == 0 or (epoch + 1) % 5 == 0:
            states = [None] * world if rank == 0 else None
            dist.gather_object(rng_state(), states, dst=0)
            if rank == 0:
                atomic_save(checkpoint, dict(identity=identity, epoch=epoch + 1,
                    encoder=net.module.state_dict(), critic=critic.state_dict(),
                    optimizer=optimizer.state_dict(), rng=states))
            dist.barrier()
    if rank == 0:
        atomic_json(folder / 'complete.json', dict(complete=True, identity=identity, epochs=400,
                    source_fingerprint=base.fingerprint, checkpoint_sha256=file_hash(checkpoint)))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
