"""Isolated CIFAR10 0.2% nearest-prototype repulsion ablation; baseline unchanged."""
import argparse
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

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
BASE = ROOT / 'Evaluate_benchamrk/mipa_cvpr_color_crop'
sys.path.insert(0, str(ROOT / 'OT-SSL-DD'))
import eval as reference_eval


def read(path):
    return json.loads(Path(path).read_text())


def nearest_cosines(features):
    z = F.normalize(features, dim=1)
    similarity = z @ z.T
    return similarity.masked_fill(torch.eye(len(z), dtype=torch.bool, device=z.device), -torch.inf).max(dim=1).values


def duplicate_loss(features, rho):
    return (nearest_cosines(features) - rho).clamp_min(0).square().mean()


def self_check():
    assert duplicate_loss(torch.eye(3), .95).item() == 0, 'Self-similarity must be excluded'
    expected = torch.tensor((1 - .95) ** 2)
    assert torch.allclose(duplicate_loss(torch.tensor([[1., 0.], [1., 0.]]), .95), expected)
    close = torch.tensor([[1., 0.], [1., .1]], requires_grad=True)
    loss = duplicate_loss(close, .95)
    loss.backward()
    assert loss.item() > 0 and torch.isfinite(close.grad).all() and close.grad.norm() > 0
    assert torch.allclose(loss.detach(), duplicate_loss(close.detach().flip(0), .95))
    print('PASS: diagonal exclusion, exact penalty, permutation invariance, nonzero gradients', flush=True)


@torch.no_grad()
def diagnostics(saved, checkpoint, job, common, run, rho):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cache = ROOT / 'OT-SSL-DD/tsne_mipa_random/CIFAR10'
    provenance = read(cache / 'pt0.2/inputs.json')
    baseline_checkpoint = Path(job['synthetic_dir']) / checkpoint.name
    assert provenance['teacher_sha256'] == job['teacher_sha256']
    assert common.file_hash(baseline_checkpoint) == provenance['checkpoint_sha256']
    assert common.file_hash(cache / 'real_features.npy') == provenance['real_features_sha256']
    real = torch.from_numpy(np.load(cache / 'real_features.npy', allow_pickle=False)).cuda()
    teacher, _, weights = common.load_encoder(job['teacher'])
    mean, std = [torch.tensor(v, device='cuda')[None, :, None, None] for v in saved['normalization']]
    tm, ts = [torch.tensor(v, device='cuda')[None, :, None, None] for v in weights['normalization']]
    initial = (saved['initial_data'].cuda() * std + mean).clamp(0, 1)
    final = (saved['data'].cuda() * std + mean).clamp(0, 1)
    with np.load(cache / 'pt0.2/features.npz', allow_pickle=False) as cached:
        random = torch.from_numpy(cached['random']).cuda()
        baseline = torch.from_numpy(cached['mipa']).cuda()
    assert torch.allclose(F.normalize(teacher((initial - tm) / ts), dim=1), random, atol=2e-4)
    current = F.normalize(teacher((final - tm) / ts), dim=1)
    report = dict(feature_space=provenance['feature_space'], rho=rho, labels_used=False, metrics={})
    for name, features in [('random', random), ('baseline_mipa', baseline), ('penalized_mipa', current)]:
        nearest = nearest_cosines(features)
        distances = torch.cat([1 - (block @ features.T).max(dim=1).values for block in real.split(1024)])
        report['metrics'][name] = dict(nearest_prototype_cosine_mean=nearest.mean().item(),
                                      fraction_nearest_above_rho=(nearest > rho).float().mean().item(),
                                      real_to_nearest_prototype_cosine_distance_mean=distances.mean().item())
    assert common.state_hash(teacher) == weights['encoder_sha256']
    common.atomic_json(run / 'feature_diagnostics.json', report)
    print('FEATURE DIAGNOSTICS: ' + json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--duplicate_weight', type=float, choices=(1., 10., 100., 1000.), required=True)
    parser.add_argument('--duplicate_threshold', type=float, default=.95)
    args = parser.parse_args()
    assert math.isfinite(args.duplicate_threshold) and -1 < args.duplicate_threshold < 1
    assert os.environ.get('SLURM_JOB_ID') and torch.cuda.is_available() and torch.cuda.device_count() == 1
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    self_check()
    reference = reference_eval.reference_modules()
    common, data = reference['common'], reference['data']
    job = read(BASE / 'snapshot/CIFAR10_pt0p2.json')
    for path, expected in job['code_hashes'].items():
        assert common.file_hash(path) == expected, 'Baseline code changed: ' + path
    assert common.file_hash(job['teacher']) == job['teacher_sha256']
    implementation_hash = common.file_hash(__file__)
    ablation = dict(duplicate_weight=args.duplicate_weight, duplicate_threshold=args.duplicate_threshold,
                    penalty='mean_j relu(max_{k!=j} cosine(z_j,z_k)-rho)^2',
                    views='Mean penalty across the two existing augmented synthetic views', coverage_weight=0.)
    run = HERE / 'runs' / ('lambda%g_rho%g' % (args.duplicate_weight, args.duplicate_threshold))
    synthetic_dir, folder = run / 'synthetic', run / 'evaluation'
    checkpoint = synthetic_dir / 'res_IIC_CIFAR10_ResNet18BN_0.2percent_final.pt'
    baseline_checkpoint = Path(job['synthetic_dir']) / checkpoint.name
    with common.exclusive_run(run):
        samples, iteration = [], 0
        if checkpoint.exists():
            previous = torch.load(checkpoint, map_location='cpu', weights_only=True)
            assert previous['ablation'] == ablation and previous['ablation_implementation_sha256'] == implementation_hash
            samples, iteration = previous['ablation_loss_samples'], previous['iteration']
            del previous
        spec = importlib.util.spec_from_file_location('mipa_duplicate_base', BASE / 'snapshot/method1_cvpr.py')
        method = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(method)
        original_loss, original_save = method.bipartite_view_transition_loss, common.atomic_save

        def objective(real_u, real_v, syn_u, syn_v, tau, beta, pool_size):
            nonlocal iteration
            assert beta == 1 and pool_size == 0
            mi = original_loss(real_u, real_v, syn_u, syn_v, tau, beta, pool_size)
            penalty = .5 * (duplicate_loss(syn_u, args.duplicate_threshold) + duplicate_loss(syn_v, args.duplicate_threshold))
            total = mi + args.duplicate_weight * penalty
            iteration += 1
            if iteration == 1 or iteration % 200 == 0:
                with torch.no_grad():
                    nearest = torch.cat((nearest_cosines(syn_u), nearest_cosines(syn_v)))
                    sample = dict(iteration=iteration, negative_mi=mi.item(), duplicate=penalty.item(),
                                  weighted_duplicate=(args.duplicate_weight * penalty).item(), total=total.item(),
                                  fraction_nn_above_rho=(nearest > args.duplicate_threshold).float().mean().item())
                samples.append(sample)
                print('ABLATION LOSS: ' + json.dumps(sample), flush=True)
            return total

        def save_ablation(path, payload):
            if Path(path) == checkpoint:
                payload.update(method='MIPA_nearest_duplicate_ablation', ablation=ablation,
                               ablation_implementation_sha256=implementation_hash, ablation_loss_samples=samples)
            original_save(path, payload)

        method.bipartite_view_transition_loss = objective
        common.atomic_save = save_ablation
        sys.argv = [str(BASE / 'snapshot/method1_cvpr.py')]
        for key, value in job['generation_args'].items():
            if isinstance(value, bool):
                if value:
                    sys.argv.append('--' + key)
            else:
                sys.argv += ['--' + key, str(value)]
        sys.argv += ['--device', 'cuda', '--data_path', job['data_root'], '--save_path', str(synthetic_dir), '--resume', '--save_every', '500']
        print('ABLATION: ' + json.dumps(ablation), flush=True)
        try:
            method.main()
        finally:
            common.atomic_save = original_save
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        assert saved['complete'] and saved['iteration'] == 10000 and saved['run_args'] == job['generation_args']
        baseline = torch.load(baseline_checkpoint, map_location='cpu', weights_only=True)
        assert torch.equal(saved['initial_data'], baseline['initial_data']), 'Initialization differs from the control'
        del baseline
        diagnostics(saved, checkpoint, job, common, run, args.duplicate_threshold)

        signature = dict(read(Path(job['evaluation_dir']) / 'config.json')['signature'], experiment='mipa_duplicate_ablation',
                         ablation=ablation, ablation_implementation_sha256=implementation_hash,
                         synthetic_path=str(checkpoint), synthetic_sha256=common.file_hash(checkpoint))
        protocol = signature['protocol']
        assert protocol['ssl']['subset_epochs'] == 2000 and protocol['evaluation']['epochs'] == 100
        identity = dict(signature=common.digest_json(signature), seed=0, selection_method='ours')
        config = dict(signature=signature, identity=identity)
        if (folder / 'config.json').exists():
            assert read(folder / 'config.json') == config, 'Evaluation inputs changed'
        common.atomic_json(folder / 'config.json', config)
        train_set, test_set = (data.RawDataset(job['data_root'], 'CIFAR10', training) for training in (True, False))
        assert train_set.fingerprint == signature['source_fingerprint']
        mean, std = [torch.tensor(v)[None, :, None, None] for v in saved['normalization']]
        synthetic = reference_eval.SyntheticImages((saved['data'] * std + mean).clamp(0, 1).contiguous())
        del saved
        progress_spec = importlib.util.spec_from_file_location('duplicate_eval_progress', BASE / 'snapshot/worker.py')
        baseline_worker = importlib.util.module_from_spec(progress_spec)
        progress_spec.loader.exec_module(baseline_worker)
        reference['train_ssl'].print = baseline_worker.progress
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
                assert result['complete'] and result['encoder_unchanged'] and result['encoder_sha256'] == trained['encoder_sha256']
                assert math.isfinite(result['accuracy_percent'])
                scores.append(result['accuracy_percent'])
            control_path = Path(job['evaluation_dir']) / ('summary_CIFAR10_lbl%d.json' % labels)
            control = read(control_path)
            assert control['complete'] and control['completed_probes'] == 5
            summary = dict(complete=True, ablation=ablation, labels=labels, ssl_epochs=2000, linear_epochs=100,
                           completed_probes=5, expected_probes=5, std_ddof=1, accuracies=scores,
                           mean_percent=statistics.mean(scores), std_percent=statistics.stdev(scores),
                           baseline_summary=str(control_path), baseline_mean_percent=control['mean_percent'],
                           baseline_std_percent=control['std_percent'], difference_pp=statistics.mean(scores) - control['mean_percent'])
            common.atomic_json(folder / ('summary_CIFAR10_lbl%d.json' % labels), summary)
            print('COMPLETE: ' + json.dumps(summary), flush=True)
        assert common.file_hash(job['teacher']) == job['teacher_sha256']
        common.atomic_json(folder / 'complete.json', dict(complete=True, ablation=ablation, identity=identity))


if __name__ == '__main__':
    main()
