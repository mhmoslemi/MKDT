"""Run the unmodified upstream trainer with the requested single-GPU configuration."""
import os
from pathlib import Path
import runpy
import sys
import torch

from contracts import HERE, UPSTREAM, allocated, atomic_json, exclusive_run, file_hash, folder, read
from integration import SavedBenchmarkSource, config_for


def main():
    allocated()
    checking = sys.argv[1] == '--check'
    arguments = sys.argv[2:] if checking else sys.argv[1:]
    cfg, unit = config_for(arguments[0], Path(arguments[1]).resolve())
    output = folder(unit)
    if checking:
        for path in sorted((HERE / 'units').glob('*.json')):
            dataset = SavedBenchmarkSource(str(path), arguments[1])
            assert dataset.get_length() in (100, 500, 50000) and dataset.get_sample(0).size == (32, 32)
            print('INPUT CHECK PASSED:', path.name, dataset.get_length(), flush=True)
            del dataset
        output = Path(os.environ['SLURM_TMPDIR']) / 'a2mim_upstream_integration_check'
        cfg.runner.max_epochs = 8
        cfg.custom_hooks = []
        cfg.checkpoint_config.interval = 8
        cfg.log_config.interval = 1
    with exclusive_run(output):
        identity_file = output / 'config.json'
        if identity_file.exists() and read(identity_file) != unit:
            raise ValueError('Saved recipe differs; do not mix experiments')
        atomic_json(identity_file, unit)
        if not checking and (output / 'ssl_complete.json').exists():
            marker = read(output / 'ssl_complete.json')
            if (not marker['complete'] or marker['identity'] != unit['identity'] or marker['epochs'] != unit['epochs']
                    or file_hash(output / 'encoder.pt') != marker['checkpoint_file_sha256']):
                raise ValueError('Wrong completed encoder')
            print('SKIP completed encoder:', output, flush=True)
            return
        config = output / 'upstream_config.py'
        cfg.dump(str(config))
        os.environ.update(RANK='0', LOCAL_RANK='0', WORLD_SIZE='1', MASTER_ADDR='127.0.0.1',
                          MASTER_PORT=str(20000 + int(os.environ['SLURM_JOB_ID']) % 30000))
        sys.argv = [str(UPSTREAM / 'tools/train.py'), str(config), '--launcher', 'pytorch',
                    '--seed', '0', '--auto_resume', '--work_dir', str(output)]
        runpy.run_path(str(UPSTREAM / 'tools/train.py'), run_name='__main__')
        if checking:
            saved = torch.load(output / 'epoch_8.pth', map_location='cpu', weights_only=False)
            assert saved['meta']['epoch'] == 8 and saved['meta']['iter'] == 16
            assert saved['optimizer']['state'] and 'fp16' in saved['meta']
            assert all(torch.isfinite(value).all() for value in saved['state_dict'].values())
            atomic_json(HERE / 'integration_ready.json', dict(complete=True, identity=unit['identity'],
                original_trainer=True, original_optimizer_hook=True, epochs=8, microbatches=16))
            print('UPSTREAM TRAINER / DATALOADER / FP16 ACCUMULATION / CHECKPOINT CHECK PASSED', flush=True)
            return
        if not (output / 'ssl_complete.json').exists():
            raise RuntimeError('Upstream training returned without its final encoder export')


if __name__ == '__main__':
    main()
