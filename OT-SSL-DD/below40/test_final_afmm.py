"""Small AFMM regression tests; run on an allocated compute node, not the login node."""

import ast
from pathlib import Path
import unittest

import torch
import torch.nn.functional as F


# Load only the loss: no dataset, training, utils imports, or GPU initialization.
source = ast.parse(Path(__file__).with_name('final.py').read_text())
function = next(node for node in source.body
                if isinstance(node, ast.FunctionDef) and node.name == 'augmented_feature_moment_loss')
namespace = {'torch': torch, 'F': F}
exec(compile(ast.Module(body=[function], type_ignores=[]), 'final.py', 'exec'), namespace)
moment_loss = namespace['augmented_feature_moment_loss']


class PairedMomentTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_identical_pairs_have_zero_loss(self):
        u, v = torch.randn(5, 3), torch.randn(5, 3)
        self.assertEqual(moment_loss(u, v, u.clone(), v.clone())[0].item(), 0.0)

    def test_broken_pairing_is_visible(self):
        u = torch.tensor([[-1.0], [1.0]])
        v = u.clone()
        broken_v = v.flip(0)
        # The old pooled-view objective is exactly blind to this permutation.
        old_real, old_syn = torch.cat((u, v)), torch.cat((u, broken_v))
        old_loss = F.mse_loss(old_real.mean(0), old_syn.mean(0))
        old_loss = old_loss + F.mse_loss(torch.cov(old_real.T), torch.cov(old_syn.T))
        self.assertEqual(old_loss.item(), 0.0)
        self.assertEqual(moment_loss(u, v, u, broken_v)[0].item(), 8.0)

    def test_real_targets_detached_and_both_synthetic_views_differentiable(self):
        real_u = torch.randn(7, 3, requires_grad=True)
        real_v = torch.randn(7, 3, requires_grad=True)
        syn_u = torch.randn(4, 3, requires_grad=True)
        syn_v = torch.randn(4, 3, requires_grad=True)
        moment_loss(real_u, real_v, syn_u, syn_v)[0].backward()
        self.assertIsNone(real_u.grad)
        self.assertIsNone(real_v.grad)
        for syn in (syn_u, syn_v):
            self.assertTrue(torch.isfinite(syn.grad).all())
            self.assertGreater(syn.grad.norm().item(), 0)

    def test_numerical_gradient(self):
        real_u, real_v = torch.randn(5, 2, dtype=torch.double), torch.randn(5, 2, dtype=torch.double)
        syn_u = torch.randn(3, 2, dtype=torch.double, requires_grad=True)
        syn_v = torch.randn(3, 2, dtype=torch.double, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(
            lambda u, v: moment_loss(real_u, real_v, u, v)[0], (syn_u, syn_v)))

    def test_pixel_update_through_frozen_encoder(self):
        encoder = torch.nn.Linear(4, 3).double().eval()
        encoder.requires_grad_(False)
        real = torch.randn(8, 4, dtype=torch.double)
        pixels = torch.randn(5, 4, dtype=torch.double, requires_grad=True)
        optimizer = torch.optim.SGD([pixels], lr=0.01)

        def objective():
            return moment_loss(encoder(real), encoder(real.flip(1)),
                               encoder(pixels), encoder(pixels.flip(1)))[0]

        before = objective().item()
        for _ in range(20):
            optimizer.zero_grad()
            objective().backward()
            optimizer.step()
        self.assertLess(objective().item(), before)
        self.assertTrue(all(p.grad is None for p in encoder.parameters()))

    def test_invalid_pair_batches_rejected(self):
        with self.assertRaises(ValueError):
            moment_loss(torch.zeros(2, 3), torch.zeros(3, 3), torch.zeros(2, 3), torch.zeros(2, 3))
        with self.assertRaises(ValueError):
            moment_loss(torch.zeros(2, 3), torch.zeros(2, 3), torch.zeros(1, 3), torch.zeros(1, 3))


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
