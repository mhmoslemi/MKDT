"""Check paired K=10 cosine views and measure full-loop GPU time using temporary 75-step runs, without SSL training."""
import json
import math
import os
from pathlib import Path
import re
import sys
import time

import torch

HERE = Path(__file__).resolve().parent
BENCH, ROOT = HERE.parent, HERE.parent.parent
sys.path.insert(0, str(ROOT / 'OT-SSL-DD'))
import method1


def main():
    assert os.environ.get('SLURM_JOB_ID') and torch.cuda.is_available() and torch.cuda.device_count() == 1
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    common = method1.reference_modules()['common']
    results = {}
    original_augmenter = method1.make_diff_augmenter
    for source in ('CIFAR10', 'FashionMNIST'):
        config = BENCH / ('mipa_similarity_ablation/results/%s/ResNet18BN/pt0.2/mmd/simclr/ssl_seed_000/config.json' % source)
        old = json.loads(config.read_text())['signature']
        args = dict(old['generation_args'], Iteration=75, lr_img=0.002, cosine_views=10)
        if source == 'FashionMNIST':
            args['single_channel'] = True
        output = Path(os.environ['SLURM_TMPDIR']) / ('mipa_cosine10_timing_' + source + '_' + os.environ['SLURM_JOB_ID'])
        assert not output.exists()
        argv = [str(ROOT / 'OT-SSL-DD/method1.py')]
        for key, value in args.items():
            if isinstance(value, bool):
                argv.append('--' + (key if value else 'no-' + key))
            else:
                argv += ['--' + key] + [str(v) for v in (value if isinstance(value, list) else [value])]
        sys.argv = argv + ['--device', 'cuda', '--data_path', str(ROOT.parent / 'data'), '--save_path', str(output),
                          '--init', 'random', '--similarity', 'cosine', '--distill_aug_backend', 'dsa', '--save_every', '500']
        points = []
        started = time.monotonic()
        pre_loop = None

        def progress(*values, **kwargs):
            nonlocal pre_loop
            if values and isinstance(values[0], str):
                if 'distillation begins' in values[0]:
                    torch.cuda.synchronize()
                    pre_loop = time.monotonic() - started
                match = re.search(r'iter = (\d+), neg_MI = ([\d.eE+-]+)', values[0])
                if match:
                    assert math.isfinite(float(match[2]))
                    torch.cuda.synchronize()
                    points.append((int(match[1]), time.monotonic()))
            print(*values, **kwargs)

        method1.print = progress
        augmentation_seeds = []

        def paired_augmenter(*values, **kwargs):
            augment = original_augmenter(*values, **kwargs)

            def tracked(images, seed):
                augmentation_seeds.append(seed)
                return augment(images, seed)

            return tracked

        method1.make_diff_augmenter = paired_augmenter
        method1.main()
        assert pre_loop is not None and [point[0] for point in points] == [25, 50, 75]
        rates = [(second[1] - first[1]) / (second[0] - first[0]) for first, second in zip(points, points[1:])]
        path = output / ('res_IIC_%s_ResNet18BN_0.2percent_final.pt' % source)
        saved = torch.load(path, map_location='cpu', weights_only=True)
        assert saved['complete'] and saved['iteration'] == 75 and torch.isfinite(saved['data']).all()
        assert saved['iic_params']['similarity'] == 'cosine' and saved['iic_params']['cosine_views'] == 10
        assert saved['iic_params']['feature_layers'] == 'last'
        assert len(augmentation_seeds) == 75 * 10 * 4
        assert all(augmentation_seeds[i:i+2] == augmentation_seeds[i+2:i+4] for i in range(0, len(augmentation_seeds), 4))
        if source == 'FashionMNIST':
            assert torch.equal(saved['data'], saved['data'][:, :1].expand_as(saved['data']))
        results[source] = dict(iteration_seconds_p95=max(rates), pre_loop_seconds=pre_loop,
            measured_intervals=rates, teacher_sha256=common.file_hash(ROOT / old['teacher']['path']))
        print('COSINE K=10 GPU TIMING:', source, json.dumps(results[source]), flush=True)
    common.atomic_json(HERE / 'cosine10_timing.json', dict(complete=True, method_sha256=common.file_hash(ROOT / 'OT-SSL-DD/method1.py'),
        check_sha256=common.file_hash(__file__), datasets=results))


if __name__ == '__main__':
    main()
