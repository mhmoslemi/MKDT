"""CIFAR10 0.2% table recipe plus symmetric contrast of assignment columns."""
import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import sys

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
BASE = ROOT / 'Evaluate_benchamrk/mipa_cvpr_color_crop'
sys.path.insert(0, str(ROOT / 'OT-SSL-DD'))
import eval as reference_eval


def read(path):
    return json.loads(Path(path).read_text())


def module_at(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def column_loss(q_u, q_v, temperature):
    # Center over real images, then compare each prototype's response pattern.
    u = F.normalize(q_u - q_u.mean(dim=0, keepdim=True), dim=0, eps=1e-6)
    v = F.normalize(q_v - q_v.mean(dim=0, keepdim=True), dim=0, eps=1e-6)
    correlation = u.T @ v
    targets = torch.arange(q_u.shape[1], device=q_u.device)
    logits = correlation / temperature
    loss = .5 * (F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets))
    return loss, correlation


def assignment_columns(real_u, real_v, syn_u, syn_v, tau, temperature):
    effective_tau = tau * math.log(100.) / math.log(max(len(syn_u), 100))
    q_u = F.softmax((F.normalize(real_u, dim=1) @ F.normalize(syn_u, dim=1).T) / effective_tau, dim=1)
    q_v = F.softmax((F.normalize(real_v, dim=1) @ F.normalize(syn_v, dim=1).T) / effective_tau, dim=1)
    loss, correlation = column_loss(q_u, q_v, temperature)
    return loss, correlation, q_u, q_v


def self_check(original_loss):
    q = torch.eye(4, dtype=torch.float64).repeat(2, 1)
    matched, correlation = column_loss(q, q, .2)
    expected = math.log1p(3 * math.exp((-1 / 3 - 1) / .2))
    assert torch.allclose(matched, matched.new_tensor(expected), atol=1e-10)
    assert torch.allclose(correlation.diag(), torch.ones(4, dtype=q.dtype))
    assert column_loss(q, q.roll(1, dims=1), .2)[0] > matched
    uniform = torch.full((8, 4), .25, dtype=torch.float64, requires_grad=True)
    flat, _ = column_loss(uniform, uniform, .2)
    assert torch.allclose(flat, flat.new_tensor(math.log(4)))
    assert torch.isfinite(torch.autograd.grad(flat, uniform)[0]).all()
    generator = torch.Generator().manual_seed(923)
    a, b = [torch.randn(8, 4, generator=generator, dtype=torch.float64).softmax(1) for _ in range(2)]
    assert torch.allclose(column_loss(a, b, .2)[0], column_loss(b, a, .2)[0])
    permutation = torch.tensor([2, 0, 3, 1])
    assert torch.allclose(column_loss(a, b, .2)[0], column_loss(a.flip(0)[:, permutation], b.flip(0)[:, permutation], .2)[0])
    assert torch.autograd.gradcheck(lambda x, y: column_loss(x, y, .2)[0], (a.requires_grad_(), b.requires_grad_()))
    real_u, real_v = [torch.randn(8, 6, generator=generator, dtype=torch.float64) for _ in range(2)]
    syn_u, syn_v = [torch.randn(4, 6, generator=generator, dtype=torch.float64, requires_grad=True) for _ in range(2)]
    extra, _, qu, qv = assignment_columns(real_u, real_v, syn_u, syn_v, .2, .2)
    joint = qu.T @ qv / len(qu)
    joint = .5 * (joint + joint.T)
    mi = (joint * (joint.sum(1, keepdim=True).log() + joint.sum(0, keepdim=True).log() - joint.log())).sum()
    assert torch.allclose(mi, original_loss(real_u, real_v, syn_u, syn_v, .2, 1, 0), atol=1e-12)
    gradients = torch.autograd.grad(extra, (syn_u, syn_v))
    assert all(torch.isfinite(g).all() and g.norm() > 0 for g in gradients)
    print('PASS: analytic column loss, matching identity, view/permutation symmetry, flat columns, gradcheck, unchanged assignments/MI, synthetic gradients', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--column_weight', type=float, default=.1)
    parser.add_argument('--column_temperature', type=float, default=.2)
    args = parser.parse_args()
    assert all(math.isfinite(x) and x > 0 for x in (args.column_weight, args.column_temperature))
    assert os.environ.get('SLURM_JOB_ID') and torch.cuda.is_available() and torch.cuda.device_count() == 1
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    reference = reference_eval.reference_modules()
    common, data = reference['common'], reference['data']
    job = read(BASE / 'snapshot/CIFAR10_pt0p2.json')
    for path, expected in job['code_hashes'].items():
        assert common.file_hash(path) == expected, 'Baseline code changed: ' + path
    assert common.file_hash(job['teacher']) == job['teacher_sha256']
    control_config = read(Path(job['evaluation_dir']) / 'config.json')
    assert common.digest_json(control_config['signature']) == control_config['identity']['signature']
    assert control_config['signature']['generation_args'] == job['generation_args']
    baseline_checkpoint = Path(control_config['signature']['synthetic_path'])
    assert common.file_hash(baseline_checkpoint) == control_config['signature']['synthetic_sha256']
    implementation_hash = common.file_hash(__file__)
    ablation = dict(column_weight=args.column_weight, column_temperature=args.column_temperature,
                    objective='negative_MI + column_weight * symmetric_column_cross_entropy',
                    columns='assignment probabilities, centered over real images, L2 normalization with eps=1e-6',
                    positive='same prototype across the two existing views', adaptive_temperature=False, duplicate_penalty=False)
    run = HERE / 'runs' / ('lambda%g_tc%g' % (args.column_weight, args.column_temperature))
    synthetic_dir, folder = run / 'synthetic', run / 'evaluation'
    checkpoint = synthetic_dir / baseline_checkpoint.name
    method = module_at('mipa_column_base', BASE / 'snapshot/method1_cvpr.py')
    original_loss, original_save = method.bipartite_view_transition_loss, common.atomic_save
    self_check(original_loss)
    with common.exclusive_run(run):
        samples, iteration = [], 0
        if checkpoint.exists():
            previous = torch.load(checkpoint, map_location='cpu', weights_only=True)
            assert previous['ablation'] == ablation and previous['ablation_implementation_sha256'] == implementation_hash
            samples, iteration = previous['ablation_loss_samples'], previous['iteration']
            del previous

        def objective(real_u, real_v, syn_u, syn_v, tau, beta, pool_size):
            nonlocal iteration
            assert beta == 1 and pool_size == 0 and tau == .2 and len(syn_u) == 100
            mi = original_loss(real_u, real_v, syn_u, syn_v, tau, beta, pool_size)
            extra, correlation, qu, qv = assignment_columns(real_u, real_v, syn_u, syn_v, tau, args.column_temperature)
            total = mi + args.column_weight * extra
            if not torch.isfinite(total):
                raise RuntimeError('Non-finite column-ablation objective')
            iteration += 1
            if iteration == 1 or iteration % 200 == 0:
                with torch.no_grad():
                    diagonal = torch.eye(len(syn_u), dtype=torch.bool, device=correlation.device)
                    norms = torch.cat([(q - q.mean(0)).norm(dim=0) for q in (qu, qv)])
                    sample = dict(iteration=iteration, negative_mi=mi.item(), column_loss=extra.item(),
                                  weighted_column_loss=(args.column_weight * extra).item(), total=total.item(),
                                  matching_correlation_mean=correlation.diag().mean().item(),
                                  other_correlation_mean=correlation[~diagonal].mean().item(),
                                  column_identification_accuracy=.5 * ((correlation.argmax(1) == torch.arange(len(syn_u), device=correlation.device)).float().mean().item() + (correlation.argmax(0) == torch.arange(len(syn_u), device=correlation.device)).float().mean().item()),
                                  near_constant_column_fraction=(norms < 1e-6).float().mean().item())
                samples.append(sample)
                print('ABLATION LOSS: ' + json.dumps(sample, allow_nan=False), flush=True)
            return total

        def save_ablation(path, payload):
            if Path(path) == checkpoint:
                payload.update(method='MIPA_assignment_column_ablation', ablation=ablation,
                               ablation_implementation_sha256=implementation_hash, ablation_loss_samples=samples)
            original_save(path, payload)

        def generation_progress(*values, **kwargs):
            if values and isinstance(values[0], str) and ' iter = ' in values[0] and iteration % 200:
                return
            print(*values, **kwargs)

        method.bipartite_view_transition_loss = objective
        method.print = generation_progress
        common.atomic_save = save_ablation
        sys.argv = [str(BASE / 'snapshot/method1_cvpr.py')]
        for key, value in job['generation_args'].items():
            if isinstance(value, bool):
                if value:
                    sys.argv.append('--' + key)
            else:
                sys.argv += ['--' + key, str(value)]
        sys.argv += ['--device', 'cuda', '--data_path', job['data_root'], '--save_path', str(synthetic_dir), '--resume', '--save_every', '500']
        common.atomic_json(run / 'ablation.json', dict(ablation=ablation, generation_args=job['generation_args'],
                           implementation_sha256=implementation_hash, baseline_config=str(Path(job['evaluation_dir']) / 'config.json')))
        print('ABLATION: ' + json.dumps(ablation), flush=True)
        try:
            method.main()
        finally:
            common.atomic_save = original_save
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        assert saved['complete'] and saved['iteration'] == 10000 and saved['run_args'] == job['generation_args']
        assert saved['teacher_source'] == 'pretrained' and saved['teacher_seed'] == 0
        baseline = torch.load(baseline_checkpoint, map_location='cpu', weights_only=True)
        assert torch.equal(saved['initial_data'], baseline['initial_data']), 'Initialization differs from the table control'
        del baseline

        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        signature = dict(control_config['signature'], experiment='mipa_column_ablation', ablation=ablation,
                         ablation_implementation_sha256=implementation_hash,
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
        assert saved['data'].shape == (100, 3, 32, 32) and torch.isfinite(saved['data']).all()
        assert saved['normalization'] == signature['normalization']
        mean, std = [torch.tensor(v)[None, :, None, None] for v in saved['normalization']]
        synthetic = reference_eval.SyntheticImages((saved['data'] * std + mean).clamp(0, 1).contiguous())
        del saved
        baseline_worker = module_at('column_eval_progress', BASE / 'snapshot/worker.py')
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
            summary = dict(complete=True, identity=identity, ablation=ablation, labels=labels, ssl_epochs=2000, linear_epochs=100,
                           completed_probes=5, expected_probes=5, std_ddof=1, accuracies=scores,
                           mean_percent=statistics.mean(scores), std_percent=statistics.stdev(scores),
                           baseline_summary=str(control_path), baseline_mean_percent=control['mean_percent'],
                           baseline_std_percent=control['std_percent'], difference_pp=statistics.mean(scores) - control['mean_percent'])
            common.atomic_json(folder / ('summary_CIFAR10_lbl%d.json' % labels), summary)
            print('COMPLETE: ' + json.dumps(summary, allow_nan=False), flush=True)
        assert common.file_hash(job['teacher']) == job['teacher_sha256']
        common.atomic_json(folder / 'complete.json', dict(complete=True, ablation=ablation, identity=identity))


if __name__ == '__main__':
    main()
