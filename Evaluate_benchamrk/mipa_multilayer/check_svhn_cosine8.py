"""Measure the three SVHN K=8 generation sizes on a GPU, without SSL training."""
import json
import os
from pathlib import Path
import sys
import time

import torch

HERE = Path(__file__).resolve().parent
BENCH, ROOT = HERE.parent, HERE.parent.parent
sys.path.insert(0, str(ROOT / 'OT-SSL-DD'))
import method1


def main():
    assert os.environ.get('SLURM_JOB_ID') and torch.cuda.is_available()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    common = method1.reference_modules()['common']
    original_step = torch.optim.Adam.step
    results = {}
    for subset in (0.2, 1, 2):
        folder = BENCH / ('mipa_similarity_ablation/results/SVHN/ResNet18BN/pt%g/mmd/simclr/ssl_seed_000' % subset)
        old = json.loads((folder / 'config.json').read_text())['signature']
        args = dict(old['generation_args'], Iteration=7, lr_img=0.001, cosine_views=8)
        assert not args.get('assignment_filter', False)
        output = Path(os.environ['SLURM_TMPDIR']) / ('svhn_cos8_%g_%s' % (subset, os.environ['SLURM_JOB_ID']))
        assert not output.exists()
        argv = [str(ROOT / 'OT-SSL-DD/method1.py')]
        for key, value in args.items():
            if isinstance(value, bool):
                argv.append('--' + (key if value else 'no-' + key))
            else:
                argv += ['--' + key] + [str(v) for v in (value if isinstance(value, list) else [value])]
        sys.argv = argv + ['--device', 'cuda', '--data_path', str(ROOT.parent / 'data'), '--save_path', str(output),
                          '--init', 'random', '--similarity', 'cosine', '--feature_layers', 'last', '--distill_aug_backend', 'dsa', '--num_eval', '0']
        points = []
        started = time.monotonic()

        def step(optimizer, *values, **kwargs):
            value = original_step(optimizer, *values, **kwargs)
            torch.cuda.synchronize()
            points.append(time.monotonic())
            return value

        torch.optim.Adam.step = step
        try:
            method1.main()
        finally:
            torch.optim.Adam.step = original_step
        assert len(points) == 7
        intervals = [second - first for first, second in zip(points[2:], points[3:])]
        saved = torch.load(output / ('res_IIC_SVHN_ResNet18BN_%gpercent_final.pt' % subset), map_location='cpu', weights_only=True)
        assert saved['complete'] and saved['iteration'] == 7 and torch.isfinite(saved['data']).all()
        assert saved['iic_params']['cosine_views'] == 8 and not saved['iic_params']['assignment_filter']
        assert saved['data'].shape == (len(old['selection']['indices']), 3, 32, 32)
        results['%g' % subset] = dict(iteration_seconds_p95=max(intervals), measured_intervals=intervals,
            startup_seconds=points[0] - started, teacher_sha256=old['teacher']['file_sha256'])
        print('SVHN K=8 TIMING:', subset, json.dumps(results['%g' % subset]), flush=True)
    common.atomic_json(HERE / 'svhn_cosine8_timing.json', dict(complete=True, method_sha256=common.file_hash(ROOT / 'OT-SSL-DD/method1.py'),
        check_sha256=common.file_hash(__file__), sizes=results))


if __name__ == '__main__':
    main()
