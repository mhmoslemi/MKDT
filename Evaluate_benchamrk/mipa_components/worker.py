"""Eq. 7 component controls using the frozen CIFAR10 main-table implementation."""
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import sys

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path('/scratch/mmoslem3/MKDT')
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'OT-SSL-DD'))
import eval as reference_eval


def read(path):
    return json.loads(Path(path).read_text())


def agreement(q_u, q_v):
    return 0.5 * (q_u - q_v).square().sum(dim=1).mean()


def agreement_loss(real_u, real_v, synthetic_u, synthetic_v, tau, beta, pool_size):
    assert beta == 1 and pool_size == 0
    temperature = tau * math.log(100.0) / math.log(max(len(synthetic_u), 100))
    q_u = F.softmax(F.normalize(real_u, dim=1) @ F.normalize(synthetic_u, dim=1).T / temperature, dim=1)
    q_v = F.softmax(F.normalize(real_v, dim=1) @ F.normalize(synthetic_v, dim=1).T / temperature, dim=1)
    return agreement(q_u, q_v)


def main():
    assert len(sys.argv) in (2, 3)
    variant = sys.argv[1]
    percentage = float(sys.argv[2]) if len(sys.argv) == 3 else 0.2
    assert variant in ('independent_agree', 'shared_agree')
    assert percentage in (0.2, 1)
    assert os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_JOB_ACCOUNT') in ('aip-yiweilu', 'aip-boyuwang')
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    reference = reference_eval.reference_modules()
    common, data = reference['common'], reference['data']
    baseline = ROOT / 'Evaluate_benchamrk/mipa_cvpr_color_crop'
    unit = ('CIFAR10_pt%g' % percentage).replace('.', 'p')
    job = next(value for value in read(baseline / 'manifest.json')['jobs'] if value['name'] == unit)
    for path, expected in job['code_hashes'].items():
        assert common.file_hash(path) == expected, 'Baseline code changed: ' + path
    assert common.file_hash(job['teacher']) == job['teacher_sha256']
    component = dict(objective='Eq7: mean_i(sum_j((q1_ij-q2_ij)^2))/2',
                     real_synthetic_augmentations='independent' if variant == 'independent_agree' else 'shared')
    run = HERE / 'runs' / variant if percentage == 0.2 else HERE / 'runs' / ('pt%g' % percentage) / variant
    if os.environ['SLURM_JOB_ACCOUNT'] == 'aip-boyuwang':
        run = run.with_name(run.name + '_wang')
    synthetic_dir, folder = run / 'synthetic', run / 'evaluation'
    checkpoint = synthetic_dir / ('res_IIC_CIFAR10_ResNet18BN_%gpercent_final.pt' % percentage)
    num_syn = int(50000 * percentage / 100)
    implementation_hash = common.file_hash(__file__)
    # Check the sum-over-prototypes scaling of Eq. 7, not mean pixel/entry MSE.
    example = torch.eye(2, device='cuda')
    assert agreement(example, example).item() == 0 and agreement(example, example.flip(0)).item() == 1
    with common.exclusive_run(run):
        # Completed results keep their original implementation identity after this size extension.
        if (folder / 'complete.json').exists():
            config, marker = read(folder / 'config.json'), read(folder / 'complete.json')
            signature = config['signature']
            assert marker['complete'] and marker['component'] == signature['component'] == component
            assert marker['identity'] == config['identity'] and common.digest_json(signature) == config['identity']['signature']
            assert signature['generation_args'] == job['generation_args'] and common.file_hash(checkpoint) == signature['synthetic_sha256']
            for labels in (1, 5):
                summary = read(folder / ('summary_CIFAR10_lbl%d.json' % labels))
                assert summary['complete'] and summary['completed_probes'] == 5
            print('SKIP completed: ' + str(folder), flush=True)
            return
        extra_rng = np.random.default_rng(30000)
        if checkpoint.exists():
            previous = torch.load(checkpoint, map_location='cpu', weights_only=True)
            assert previous['component'] == component and previous['component_implementation_sha256'] == implementation_hash
            extra_rng.bit_generator.state = previous['synthetic_augmentation_rng']
            del previous

        spec = importlib.util.spec_from_file_location('mipa_component_base', baseline / 'snapshot/method1_cvpr.py')
        method = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(method)
        method.bipartite_view_transition_loss = agreement_loss
        original_augmenter, original_save = method.make_diff_augmenter, common.atomic_save

        def make_augmenter(strategy, param):
            augment = original_augmenter(strategy, param)
            calls = 0

            def paired(images, seed):
                nonlocal calls
                synthetic_call = calls % 4 >= 2
                calls += 1
                assert len(images) == (num_syn if synthetic_call else job['generation_args']['batch_real'])
                if synthetic_call and variant == 'independent_agree':
                    seed = int(extra_rng.integers(0, 2**31 - 1))
                return augment(images, seed)

            return paired

        def save_component(path, payload):
            if Path(path) == checkpoint:
                payload.update(method='MIPA_Eq7_Agreement', component=component, component_implementation_sha256=implementation_hash,
                               synthetic_augmentation_rng=extra_rng.bit_generator.state)
            original_save(path, payload)

        method.make_diff_augmenter = make_augmenter

        def print_component(*values, **kwargs):
            if values and isinstance(values[0], str) and values[0].startswith('Objective:'):
                values = ('Objective: Eq. 7 assignment disagreement only; ' + component['real_synthetic_augmentations'] + ' real/synthetic augmentations.',)
            print(*values, **kwargs)

        method.print = print_component
        common.atomic_save = save_component
        sys.argv = [str(baseline / 'snapshot/method1_cvpr.py')]
        for key, value in job['generation_args'].items():
            if isinstance(value, bool):
                if value:
                    sys.argv.append('--' + key)
            else:
                sys.argv += ['--' + key, str(value)]
        sys.argv += ['--device', 'cuda', '--data_path', job['data_root'], '--save_path', str(synthetic_dir), '--resume', '--save_every', '500']
        print('COMPONENT: ' + json.dumps(component), flush=True)
        try:
            method.main()
        finally:
            common.atomic_save = original_save

        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        assert saved['complete'] and saved['iteration'] == job['generation_args']['Iteration'] and saved['run_args'] == job['generation_args']
        assert saved['component'] == component and saved['teacher_seed'] == 0
        full_checkpoint = Path(job['synthetic_dir']) / checkpoint.name
        full = torch.load(full_checkpoint, map_location='cpu', weights_only=True)
        assert torch.equal(saved['initial_data'], full['initial_data']), 'Initialization differs from the full-MIPA control'
        del full
        config = read(Path(job['evaluation_dir']) / 'config.json')
        signature = dict(config['signature'], experiment='mipa_components', component=component,
                         component_implementation_sha256=implementation_hash, synthetic_path=str(checkpoint),
                         synthetic_sha256=common.file_hash(checkpoint))
        protocol = signature['protocol']
        assert protocol['ssl']['subset_epochs'] == 2000 and protocol['evaluation']['epochs'] == 100
        identity = dict(signature=common.digest_json(signature), seed=0, selection_method='ours')
        config = dict(signature=signature, identity=identity)
        if (folder / 'config.json').exists():
            assert read(folder / 'config.json') == config, 'Evaluation inputs changed'
        common.atomic_json(folder / 'config.json', config)
        train_set, test_set = (data.RawDataset(job['data_root'], 'CIFAR10', training) for training in (True, False))
        assert train_set.fingerprint == signature['source_fingerprint']
        mean, std = (torch.tensor(values).view(1, 3, 1, 1) for values in saved['normalization'])
        synthetic = reference_eval.SyntheticImages((saved['data'] * std + mean).clamp(0, 1).contiguous())
        del saved
        progress_spec = importlib.util.spec_from_file_location('mipa_component_progress', baseline / 'snapshot/worker.py')
        baseline_worker = importlib.util.module_from_spec(progress_spec)
        progress_spec.loader.exec_module(baseline_worker)
        reference['train_ssl'].print = baseline_worker.progress
        with reference_eval.ssl_progress(reference['train_ssl'], 200, 2000):
            encoder, trained = reference['train_ssl'].train(synthetic, protocol, 'CIFAR10', 'ResNet18', 'simclr', 0,
                                                          folder, identity, workers=0, device='cuda')
        assert trained['epochs'] == 2000
        common.atomic_json(folder / 'ssl_complete.json', dict(complete=True, identity=identity, epochs=2000,
                           encoder_sha256=trained['encoder_sha256'], checkpoint_file_sha256=common.file_hash(encoder)))
        for labels in (1, 5):
            scores = []
            for seed in range(5):
                result = reference['evaluate'].linear_probe(encoder, identity, protocol, 'CIFAR10', 'CIFAR10', train_set,
                    test_set, labels, seed, folder / 'evaluations' / ('CIFAR10_lbl%d_probe%d.json' % (labels, seed)), workers=0, device='cuda')
                assert result['complete'] and result['encoder_unchanged'] and math.isfinite(result['accuracy_percent'])
                scores.append(result['accuracy_percent'])
            summary = dict(complete=True, component=component, labels=labels, ssl_epochs=2000, linear_epochs=100,
                           completed_probes=5, expected_probes=5, std_ddof=1, accuracies=scores,
                           mean_percent=statistics.mean(scores), std_percent=statistics.stdev(scores))
            common.atomic_json(folder / ('summary_CIFAR10_lbl%d.json' % labels), summary)
            print('COMPLETE %s lbl=%d: %.2f +/- %.2f' % (variant, labels, summary['mean_percent'], summary['std_percent']), flush=True)
        assert common.file_hash(job['teacher']) == job['teacher_sha256']
        common.atomic_json(folder / 'complete.json', dict(complete=True, component=component, identity=identity))


if __name__ == '__main__':
    main()
