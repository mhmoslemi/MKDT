"""Small numerical regression check, run inside the first allocated expert job."""
import argparse
from contextlib import redirect_stdout, redirect_stderr
import io
import os
from pathlib import Path
import pickle
import tempfile

import torch
from torch import nn

import distill


def main():
    if 'SLURM_JOB_ID' not in os.environ or not torch.cuda.is_available():
        raise RuntimeError('Run this check inside an allocated GPU job')
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory(prefix='mkdt_resume_', dir=os.environ['SLURM_TMPDIR']) as temporary:
        root = Path(temporary)
        bank = root / 'experts'
        bank.mkdir()
        torch.manual_seed(7)
        images = torch.randn(8, 3, 2, 2)
        torch.save(torch.randn(8, 4), root / 'features.pt')
        with (root / 'indices.pkl').open('wb') as stream:
            pickle.dump([0, 1, 2, 3], stream)
        for expert in range(2):
            net = nn.Sequential(nn.Flatten(), nn.Linear(12, 4))
            snapshots = [[p.detach().clone() + epoch * (0.01 + expert * 0.005)
                          for p in net.parameters()] for epoch in range(4)]
            torch.save([snapshots], bank / f'replay_buffer_{expert}.pt')
        distill.get_dataset = lambda **kwargs: (3, (2, 2), 1, [(im, 0) for im in images], None)
        distill.get_network = lambda *args, **kwargs: nn.Sequential(nn.Flatten(), nn.Linear(12, 4))
        settings = dict(dataset='toy', model='toy', seed=0, data_path=str(root),
            iters=4, eval_it=2, checkpoint_every=1, train_labels_path=str(root / 'features.pt'),
            image_init_idx_path=str(root / 'indices.pkl'), expert_dir=str(bank),
            lr_teacher=0.05, lr_img=0.01, lr_labels=0.01, lr_lr=1e-4,
            batch_syn=4, criterion='mse', expert_epochs=1, syn_steps=2,
            min_start_epoch=0, max_start_epoch=2, run_id='check')
        original_load = torch.load
        observed = set()

        def traced_load(filename, *args, **kwargs):
            if Path(filename).name.startswith('replay_buffer_'):
                observed.add(Path(filename).name)
            return original_load(filename, *args, **kwargs)

        torch.load = traced_load
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            distill.main(argparse.Namespace(**settings, save_dir=str(root / 'full'), resume=False, stop_at=4))
            assert observed == {'replay_buffer_0.pt', 'replay_buffer_1.pt'}, 'Expert file cursor did not load both banks'
            distill.main(argparse.Namespace(**settings, save_dir=str(root / 'split'), resume=False, stop_at=2))
            distill.main(argparse.Namespace(**settings, save_dir=str(root / 'split'), resume=True, stop_at=4))
        torch.load = original_load
        full = torch.load(root / 'full/training.pt', map_location='cpu', weights_only=False)
        split = torch.load(root / 'split/training.pt', map_location='cpu', weights_only=False)
        for name in ('images', 'labels', 'synlr', 'torch_rng'):
            torch.testing.assert_close(full[name], split[name], rtol=0, atol=0)
        for expected, actual in zip(full['optimizers'], split['optimizers']):
            for parameter, values in expected['state'].items():
                for name, value in values.items():
                    torch.testing.assert_close(value, actual['state'][parameter][name], rtol=0, atol=0)
        assert full['iteration'] == split['iteration'] == 4
        assert full['expert_files'] == split['expert_files']
        assert full['file_idx'] == split['file_idx'] and full['expert_idx'] == split['expert_idx']
        assert full['python_rng'] == split['python_rng']
        assert np_equal(full['numpy_rng'], split['numpy_rng'])
        print('PASS: uninterrupted versus resumed distillation, including momentum and RNG.', flush=True)


def np_equal(first, second):
    return first[0] == second[0] and (first[1] == second[1]).all() and first[2:] == second[2:]


if __name__ == '__main__':
    main()
