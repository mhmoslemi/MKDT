"""Small feature/gradient checks; run on an allocated node, without training."""
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

import method1
from dd_teachers import NormalizedTeacher


class FeatureTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)
        spec = importlib.util.spec_from_file_location('benchmark_networks', method1.ROOT / 'networks.py')
        networks = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(networks)
        network = networks.ResNet18(channel=3, num_classes=1)
        network.classifier = torch.nn.Identity()
        self.teacher = NormalizedTeacher(network, ([0.4] * 3, [0.2] * 3), ([0.5] * 3, [0.3] * 3))

    def test_default_and_equal_weight_similarity(self):
        last = method1.frozen_embedder(self.teacher)
        combined = method1.frozen_embedder(self.teacher, 'all')
        for size in (32, 64):
            images = torch.randn(4, 3, size, size)
            with torch.no_grad():
                self.assertTrue(torch.equal(last(images), self.teacher.embed(images)))
                output = combined(images)
                backbone = self.teacher.network
                x = (images * self.teacher.source_std + self.teacher.source_mean - self.teacher.teacher_mean) / self.teacher.teacher_std
                x = backbone.layer1(F.relu(backbone.bn1(backbone.conv1(x))))
                descriptors = []
                for layer in (backbone.layer2, backbone.layer3):
                    x = layer(x)
                    spatial = x.flatten(2)
                    descriptors.append(torch.cat((spatial.mean(2), spatial.std(2, unbiased=False)), dim=1))
                descriptors.append(last(images))
                expected = sum(F.normalize(value, dim=1)[:2] @ F.normalize(value, dim=1)[2:].t()
                               for value in descriptors) / 3
                actual = method1.feature_similarity(F.normalize(output[:2], dim=1), F.normalize(output[2:], dim=1))
                self.assertEqual(output.shape, (4, 1280))
                torch.testing.assert_close(actual, expected)
            self.assertFalse(backbone.layer2._forward_hooks or backbone.layer3._forward_hooks)

    def test_checkpointed_mi_gradients_leave_teacher_frozen(self):
        embed = method1.frozen_embedder(self.teacher, 'all')
        before = {key: value.clone() for key, value in self.teacher.state_dict().items()}
        real, synthetic = torch.randn(2, 3, 32, 32), torch.randn(3, 3, 32, 32, requires_grad=True)
        with torch.no_grad():
            first, second = embed(real), embed(real.flip(-1))
        views = checkpoint(embed, synthetic, use_reentrant=False), checkpoint(embed, synthetic.flip(-1), use_reentrant=False)
        loss = method1.bipartite_view_transition_loss(first, second, *views, tau=0.2)[0]
        loss.backward()
        self.assertTrue(torch.isfinite(loss) and torch.isfinite(synthetic.grad).all())
        self.assertGreater(synthetic.grad.abs().max().item(), 0)
        self.assertTrue(all(not parameter.requires_grad and parameter.grad is None for parameter in self.teacher.parameters()))
        self.assertTrue(all(torch.equal(before[key], value) for key, value in self.teacher.state_dict().items()))
        self.assertFalse(self.teacher.network.layer2._forward_hooks or self.teacher.network.layer3._forward_hooks)

    def test_resume_rejects_changed_layers_in_both_directions(self):
        for saved_layers, requested in (('all', 'last'), ('last', 'all')):
            with tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / 'res_IIC_CIFAR10_ResNet18BN_0.2percent_final.pt'
                path.touch()
                # Missing metadata represents a legacy final-layer-only checkpoint.
                saved = {} if saved_layers == 'last' else {'iic_params': {'feature_layers': saved_layers}}
                argv = ['method1.py', '--dataset', 'CIFAR10', '--device', 'cpu', '--resume',
                        '--save_path', temporary, '--feature_layers', requested]
                with patch.object(sys, 'argv', argv), patch.object(torch, 'load', return_value=saved):
                    with self.assertRaisesRegex(ValueError, 'Checkpoint feature layers differ'):
                        method1.main()

    def test_gaussian_kl_matches_directional_full_covariance_reference(self):
        real_mean = torch.randn(35, 3, dtype=torch.float64)
        real_factor = torch.randn(35, 3, 3, dtype=torch.float64)
        real_covariance = real_factor @ real_factor.transpose(1, 2) + torch.eye(3)
        syn_mean = torch.randn(4, 3, dtype=torch.float64, requires_grad=True)
        syn_factor = torch.randn(4, 3, 3, dtype=torch.float64, requires_grad=True)
        syn_covariance = syn_factor @ syn_factor.transpose(1, 2) + torch.eye(3)
        ridge = 0.001
        actual = method1.gaussian_kl(real_mean, real_covariance, syn_mean, syn_covariance, ridge)
        real = torch.distributions.MultivariateNormal(real_mean[:, None], covariance_matrix=(real_covariance + ridge * torch.eye(3))[:, None])
        synthetic = torch.distributions.MultivariateNormal(syn_mean[None], covariance_matrix=(syn_covariance + ridge * torch.eye(3))[None])
        expected = torch.distributions.kl_divergence(synthetic, real) / 3
        torch.testing.assert_close(actual, expected)
        actual_grad = torch.autograd.grad(actual.sum(), (syn_mean, syn_factor), retain_graph=True)
        expected_grad = torch.autograd.grad(expected.sum(), (syn_mean, syn_factor))
        for first, second in zip(actual_grad, expected_grad):
            torch.testing.assert_close(first, second)
        identical = method1.gaussian_kl(real_mean[:3], real_covariance[:3], real_mean[:3], real_covariance[:3], ridge)
        torch.testing.assert_close(identical.diagonal(), torch.zeros(3, dtype=torch.float64), atol=1e-12, rtol=0)

    def test_cosine_views_average_scores_before_softmax_with_gradients(self):
        real_u, real_v = [torch.randn(4, 10, 8) for _ in range(2)]
        syn_u, syn_v = [torch.randn(3, 10, 8, requires_grad=True) for _ in range(2)]
        normalized = [F.normalize(value, dim=-1) for value in (real_u, real_v, syn_u, syn_v)]
        ur, vr, us, vs = normalized
        expected_u = torch.stack([ur[:, k] @ us[:, k].t() for k in range(10)]).mean(0)
        expected_v = torch.stack([vr[:, k] @ vs[:, k].t() for k in range(10)]).mean(0)
        torch.testing.assert_close(method1.feature_similarity(ur, us), expected_u)
        pu, pv = (score.div(0.2).softmax(dim=1) for score in (expected_u, expected_v))
        joint = pu.t() @ pv / len(real_u)
        joint = ((joint + joint.t()) / 2).clamp_min(1e-8)
        expected = -(joint * (joint / (joint.sum(1, keepdim=True) * joint.sum(0, keepdim=True))).log()).sum()
        actual = method1.bipartite_view_transition_loss(real_u, real_v, syn_u, syn_v, tau=0.2)[0]
        torch.testing.assert_close(actual, expected)
        actual_grads = torch.autograd.grad(actual, (syn_u, syn_v), retain_graph=True)
        expected_grads = torch.autograd.grad(expected, (syn_u, syn_v))
        for first, second in zip(actual_grads, expected_grads):
            torch.testing.assert_close(first, second)
            self.assertTrue(torch.isfinite(first).all() and (first.square().sum((0, 2)) > 0).all())
        singles = (real_u[:, 0], real_v[:, 0], syn_u[:, 0], syn_v[:, 0])
        baseline = method1.bipartite_view_transition_loss(*singles, tau=0.2)
        one_view = method1.bipartite_view_transition_loss(*(value[:, None] for value in singles), tau=0.2)
        for first, second in zip(baseline, one_view):
            torch.testing.assert_close(first, second)

    def test_resume_rejects_changed_cosine_view_count(self):
        for previous, requested in ((1, 10), (10, 1)):
            with tempfile.TemporaryDirectory() as temporary:
                (Path(temporary) / 'res_IIC_CIFAR10_ResNet18BN_0.2percent_final.pt').touch()
                saved = {'iic_params': dict(similarity='cosine', cosine_views=previous)}
                argv = ['method1.py', '--dataset', 'CIFAR10', '--device', 'cpu', '--resume',
                        '--save_path', temporary, '--cosine_views', str(requested)]
                with patch.object(sys, 'argv', argv), patch.object(torch, 'load', return_value=saved):
                    with self.assertRaisesRegex(ValueError, 'different number of cosine views'):
                        method1.main()

    def test_d3s_calibration_checkpointed_middle_gradients_and_zero_weight(self):
        embed = method1.frozen_embedder(self.teacher, similarity='d3s')
        dataset = torch.utils.data.TensorDataset(torch.randn(6, 3, 32, 32))
        stats = method1.real_d3s_statistics(dataset, self.teacher, 'cpu', 2, 0.001)
        self.assertEqual(stats['indices'].unique().numel(), 6)
        before = {key: value.clone() for key, value in self.teacher.state_dict().items()}
        gray = torch.randn(3, 1, 32, 32, requires_grad=True)
        with torch.no_grad():
            real = embed(torch.randn(2, 3, 32, 32))
            real_second = embed(torch.randn(2, 3, 32, 32))
        synthetic = checkpoint(embed, gray.expand(-1, 3, -1, -1), use_reentrant=False)
        syn_second = checkpoint(embed, gray.flip(-1).expand(-1, 3, -1, -1), use_reentrant=False)
        moments, final = method1.unpack_d3s(synthetic)
        self.assertEqual([cov.shape for _, cov in moments], [torch.Size([3, 128, 128]), torch.Size([3, 256, 256])])
        views = (real, real_second, synthetic, syn_second)
        baseline = method1.bipartite_view_transition_loss(*(method1.unpack_d3s(view)[1] for view in views), tau=0.2)
        disabled = method1.bipartite_view_transition_loss(*views, tau=0.2, similarity='d3s', d3s_weight=0)
        for first, second in zip(baseline, disabled):
            torch.testing.assert_close(first, second)
        buffer = io.BytesIO()
        torch.save(stats, buffer)
        buffer.seek(0)
        restored = torch.load(buffer, weights_only=True)
        torch.testing.assert_close(method1.feature_similarity(real, synthetic, 'd3s', d3s_stats=stats),
                                   method1.feature_similarity(real, synthetic, 'd3s', d3s_stats=restored))
        # Detach final features to verify that the Gaussian middle layers themselves reach pixels.
        views = tuple(torch.cat((view[:, :-512], view[:, -512:].detach()), dim=1) for view in views)
        loss = method1.bipartite_view_transition_loss(*views, tau=0.2, similarity='d3s', d3s_stats=stats)[0]
        loss.backward()
        self.assertTrue(torch.isfinite(loss) and torch.isfinite(gray.grad).all())
        self.assertGreater(gray.grad.abs().max().item(), 0)
        self.assertTrue(all(parameter.grad is None and not parameter.requires_grad for parameter in self.teacher.parameters()))
        self.assertTrue(all(torch.equal(before[key], value) for key, value in self.teacher.state_dict().items()))
        self.assertFalse(self.teacher.network.layer2._forward_hooks or self.teacher.network.layer3._forward_hooks)

    def test_resume_rejects_changed_d3s_parameters(self):
        for flag, value in (('--d3s_reg', '0.01'), ('--d3s_weight', '0.5')):
            with tempfile.TemporaryDirectory() as temporary:
                (Path(temporary) / 'res_IIC_CIFAR10_ResNet18BN_0.2percent_final.pt').touch()
                saved = {'iic_params': dict(similarity='d3s', feature_layers='last', d3s_reg=0.001, d3s_weight=1.0)}
                argv = ['method1.py', '--dataset', 'CIFAR10', '--device', 'cpu', '--resume',
                        '--save_path', temporary, '--similarity', 'd3s', flag, value]
                with patch.object(sys, 'argv', argv), patch.object(torch, 'load', return_value=saved):
                    with self.assertRaisesRegex(ValueError, 'Checkpoint similarity settings differ'):
                        method1.main()


if __name__ == '__main__':
    unittest.main()
