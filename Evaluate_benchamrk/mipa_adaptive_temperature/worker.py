"""CIFAR10 0.2%: entropy-calibrated assignments, unchanged MIPA MI and evaluation."""
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
TAU_MIN, TAU_MAX = 0.001, 1.0
SEARCH_STEPS = 24
sys.path.insert(0, str(ROOT / 'OT-SSL-DD'))
import eval as reference_eval


def read(path):
    return json.loads(Path(path).read_text())


def row_entropy(scores, temperature):
    log_q = F.log_softmax(scores / temperature, dim=1)
    return -(log_q.exp() * log_q).sum(dim=1, keepdim=True)


class EntropyTemperature(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores, perplexity):
        assert scores.ndim == 2 and 1 < perplexity < scores.shape[1]
        target = math.log(perplexity)
        lower = scores.new_full((len(scores), 1), math.log(TAU_MIN))
        upper = scores.new_full((len(scores), 1), math.log(TAU_MAX))
        entropy_low = row_entropy(scores, TAU_MIN)
        entropy_high = row_entropy(scores, TAU_MAX)
        # Higher precision is needed only by the numerical-gradient self-check.
        for _ in range(48 if scores.dtype == torch.float64 else SEARCH_STEPS):
            middle = (lower + upper) * 0.5
            too_flat = row_entropy(scores, middle.exp()) > target
            lower = torch.where(too_flat, lower, middle)
            upper = torch.where(too_flat, middle, upper)
        temperature = ((lower + upper) * 0.5).exp()
        temperature = torch.where(entropy_low >= target, TAU_MIN, temperature)
        temperature = torch.where(entropy_high <= target, TAU_MAX, temperature)
        q = F.softmax(scores / temperature, dim=1)
        centered = scores - (q * scores).sum(dim=1, keepdim=True)
        variance = (q * centered.square()).sum(dim=1, keepdim=True)
        interior = (entropy_low < target) & (target < entropy_high)
        ctx.save_for_backward(q, centered, variance, temperature, interior)
        return temperature

    @staticmethod
    def backward(ctx, grad_temperature):
        q, centered, variance, temperature, interior = ctx.saved_tensors
        # Implicit differentiation of H(softmax(scores/tau)) = log(K).
        derivative = temperature * q * centered / variance.clamp_min(1e-20)
        derivative = torch.where(interior, derivative, 0.)
        return grad_temperature * derivative, None


def assignments(scores, perplexity):
    temperature = EntropyTemperature.apply(scores, perplexity)
    return F.softmax(scores / temperature, dim=1), temperature


def negative_mi(q_u, q_v):
    joint = q_u.T @ q_v / len(q_u)
    joint = 0.5 * (joint + joint.T)
    marginal_i, marginal_j = joint.sum(dim=1, keepdim=True), joint.sum(dim=0, keepdim=True)
    tiny = torch.finfo(joint.dtype).tiny
    entropy_joint = -(joint * joint.clamp_min(tiny).log()).sum()
    entropy_marginals = -(marginal_i * marginal_i.clamp_min(tiny).log()).sum() - (marginal_j * marginal_j.clamp_min(tiny).log()).sum()
    return entropy_joint - entropy_marginals


def self_check(original_loss):
    row = torch.linspace(-0.5, 0.5, 100, dtype=torch.float64)
    scores = torch.stack((row, row * 0.5, row * 2))
    for k in (5, 10):
        q, temperature = assignments(scores, k)
        assert torch.allclose(row_entropy(scores, temperature), torch.full((3, 1), math.log(k), dtype=scores.dtype), atol=1e-9)
        assert torch.allclose(q[0], q[1], atol=1e-9) and torch.allclose(q[0], q[2], atol=1e-9)
        assert torch.allclose(assignments(scores.flip(1), k)[0], q.flip(1), atol=1e-9)
        small = torch.stack((torch.linspace(-0.5, 0.5, 12), torch.linspace(-0.3, 0.7, 12).square())).double().requires_grad_()
        assert torch.autograd.gradcheck(lambda x: assignments(x, k)[0], (small,), fast_mode=True, atol=2e-5, rtol=2e-3)
    flat, temperature = assignments(torch.zeros(2, 12), 5)
    assert torch.allclose(flat, torch.full_like(flat, 1 / 12)) and torch.all(temperature == TAU_MIN)
    tied = torch.tensor([[1.] * 6 + [0.] * 6], requires_grad=True)
    q, temperature = assignments(tied, 5)
    assert temperature.item() < TAU_MIN * 1.001 and torch.isfinite(q).all()
    (q * torch.arange(12)).sum().backward()
    assert torch.isfinite(tied.grad).all()
    wide = torch.tensor([[10.] + [0.] * 11])
    assert assignments(wide, 10)[1].item() == TAU_MAX
    real_u = torch.sin(torch.arange(96).double().reshape(12, 8))
    real_v = torch.cos(torch.arange(96).double().reshape(12, 8) * 0.37)
    syn_u = torch.cos(torch.arange(128).double().reshape(16, 8) * 0.23).requires_grad_()
    syn_v = torch.sin(torch.arange(128).double().reshape(16, 8) * 0.41).requires_grad_()
    su = F.normalize(real_u, dim=1) @ F.normalize(syn_u, dim=1).T
    sv = F.normalize(real_v, dim=1) @ F.normalize(syn_v, dim=1).T
    assert torch.allclose(negative_mi(F.softmax(su / .2, dim=1), F.softmax(sv / .2, dim=1)), original_loss(real_u, real_v, syn_u, syn_v, .2, 1, 0))
    loss = negative_mi(assignments(su, 5)[0], assignments(sv, 5)[0])
    loss.backward()
    assert all(torch.isfinite(x.grad).all() and x.grad.norm() > 0 for x in (syn_u, syn_v))
    print('PASS: entropy targets, scale/permutation invariance, implicit-gradient checks, bounds/ties, unchanged MI, synthetic gradients', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--perplexity', type=int, choices=(5, 10), required=True)
    args = parser.parse_args()
    assert os.environ.get('SLURM_JOB_ID') and torch.cuda.is_available() and torch.cuda.device_count() == 1
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    reference = reference_eval.reference_modules()
    common, data = reference['common'], reference['data']
    job = read(BASE / 'snapshot/CIFAR10_pt0p2.json')
    for path, expected in job['code_hashes'].items():
        assert common.file_hash(path) == expected, 'Baseline code changed: ' + path
    assert common.file_hash(job['teacher']) == job['teacher_sha256']
    spec = importlib.util.spec_from_file_location('adaptive_mipa_base', BASE / 'snapshot/method1_cvpr.py')
    method = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(method)
    self_check(method.bipartite_view_transition_loss)
    ablation = dict(perplexity=args.perplexity, entropy_target_nats=math.log(args.perplexity), temperature_bounds=[TAU_MIN, TAU_MAX],
                    search_steps=SEARCH_STEPS, temperature_gradient='implicit for interior roots; constant at bounds',
                    view_rule='Per-image temperature solved separately in each of the two existing views',
                    objective='Unchanged negative MI; no auxiliary losses', fixed_temperature_ignored=True)
    implementation_hash = common.file_hash(__file__)
    run = HERE / 'runs' / ('K%d' % args.perplexity)
    synthetic_dir, folder = run / 'synthetic', run / 'evaluation'
    checkpoint = synthetic_dir / 'res_IIC_CIFAR10_ResNet18BN_0.2percent_final.pt'
    with common.exclusive_run(run):
        iteration, samples = 0, []
        if checkpoint.exists():
            previous = torch.load(checkpoint, map_location='cpu', weights_only=True)
            assert previous['ablation'] == ablation and previous['ablation_implementation_sha256'] == implementation_hash
            iteration, samples = previous['iteration'], previous['ablation_loss_samples']
            del previous

        def objective(real_u, real_v, syn_u, syn_v, tau, beta, pool_size):
            nonlocal iteration
            assert beta == 1 and pool_size == 0 and len(syn_u) == len(syn_v) == 100
            sim_u = F.normalize(real_u, dim=1) @ F.normalize(syn_u, dim=1).T
            sim_v = F.normalize(real_v, dim=1) @ F.normalize(syn_v, dim=1).T
            q, temperature = assignments(torch.cat((sim_u, sim_v)), args.perplexity)
            q_u, q_v = q.chunk(2)
            loss = negative_mi(q_u, q_v)
            iteration += 1
            if iteration == 1 or iteration % 200 == 0:
                with torch.no_grad():
                    entropy = -(q * q.clamp_min(torch.finfo(q.dtype).tiny).log()).sum(dim=1)
                    marginal = q.mean(dim=0)
                    sample = dict(iteration=iteration, negative_mi=loss.item(), target_perplexity=args.perplexity,
                                  achieved_perplexity_mean=entropy.exp().mean().item(), entropy_error_max=(entropy - math.log(args.perplexity)).abs().max().item(),
                                  target_missed_fraction=((entropy - math.log(args.perplexity)).abs() > 1e-3).float().mean().item(),
                                  tau_min=temperature.min().item(), tau_mean=temperature.mean().item(), tau_max=temperature.max().item(),
                                  lower_bound_fraction=(temperature <= TAU_MIN * 1.0001).float().mean().item(),
                                  upper_bound_fraction=(temperature >= TAU_MAX * .9999).float().mean().item(),
                                  hard_used_prototypes=q.argmax(dim=1).unique().numel(),
                                  effective_used_prototypes=(-(marginal * marginal.clamp_min(1e-30).log()).sum()).exp().item())
                samples.append(sample)
                print('ADAPTIVE LOSS: ' + json.dumps(sample, allow_nan=False), flush=True)
            return loss

        original_save = common.atomic_save

        def save_adaptive(path, payload):
            if Path(path) == checkpoint:
                payload.update(method='MIPA_adaptive_temperature', ablation=ablation,
                               ablation_implementation_sha256=implementation_hash, ablation_loss_samples=samples)
                for key in ('tau', 'effective_tau', 'pool_temperatures'):
                    payload['iic_params'].pop(key, None)
                payload['iic_params'].update(assignment_rule='entropy_calibrated', perplexity=args.perplexity, temperature_bounds=[TAU_MIN, TAU_MAX])
            original_save(path, payload)

        method.bipartite_view_transition_loss = objective
        common.atomic_save = save_adaptive
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
        assert saved['teacher_seed'] == 0 and saved['ablation'] == ablation
        baseline = torch.load(Path(job['synthetic_dir']) / checkpoint.name, map_location='cpu', weights_only=True)
        assert torch.equal(saved['initial_data'], baseline['initial_data']), 'Initialization differs from the control'
        del baseline

        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        signature = dict(read(Path(job['evaluation_dir']) / 'config.json')['signature'], experiment='mipa_adaptive_temperature',
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
        progress_spec = importlib.util.spec_from_file_location('adaptive_eval_progress', BASE / 'snapshot/worker.py')
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
            summary = dict(complete=True, ablation=ablation, identity=identity, labels=labels, ssl_epochs=2000, linear_epochs=100,
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
