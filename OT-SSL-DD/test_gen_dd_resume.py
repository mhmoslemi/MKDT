"""Run on a compute allocation: python -m unittest test_gen_dd_resume.py."""

import ast
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch


source = ast.parse(Path(__file__).with_name('Gen-dd.py').read_text())
function = next(node for node in source.body
                if isinstance(node, ast.FunctionDef) and node.name == 'resume_distillation')
namespace = {'torch': torch, 'os': os}
exec(compile(ast.Module(body=[function], type_ignores=[]), 'Gen-dd.py', 'exec'), namespace)
resume = namespace['resume_distillation']


class ResumeTests(unittest.TestCase):
    def test_optimizer_and_rng_continue_exactly(self):
        torch.manual_seed(9)
        z = torch.randn(3, 2, requires_grad=True)
        optimizer = torch.optim.Adam([z], lr=0.05)

        def step(latents, adam):
            adam.zero_grad()
            (latents - torch.randn_like(latents)).square().sum().backward()
            adam.step()

        for _ in range(3):
            step(z, optimizer)
        expected = {'dataset': 'CIFAR10', 'percentage': 1.0}
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'checkpoint.pt')
            torch.save(dict(expected, latents=z.detach(), iteration=3,
                            loss_history=torch.zeros(3), optimizer_z=optimizer.state_dict(),
                            rng_state=torch.get_rng_state()), path)
            step(z, optimizer)
            continued = z.detach().clone()
            restored = torch.randn(3, 2, requires_grad=True)
            restored_optimizer = torch.optim.Adam([restored], lr=0.05)
            iteration, history = resume(path, restored, restored_optimizer, expected,
                                        SimpleNamespace(restore_rng=torch.set_rng_state))
            self.assertEqual(iteration, 3)
            self.assertEqual(len(history), 3)
            step(restored, restored_optimizer)
            torch.testing.assert_close(restored, continued, rtol=0, atol=0)

    def test_legacy_checkpoint_and_metadata_rejection(self):
        z = torch.zeros(3, 2, requires_grad=True)
        optimizer = torch.optim.Adam([z])
        expected = {'dataset': 'CIFAR10', 'percentage': 1.0}
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'checkpoint.pt')
            self.assertEqual(resume(path, z, optimizer, expected, None), (0, []))
            torch.save(dict(expected, latents=torch.ones(3, 2), iteration=2,
                            loss_history=torch.tensor([2.0, 1.0])), path)
            self.assertEqual(resume(path, z, optimizer, expected, None), (2, [2.0, 1.0]))
            torch.testing.assert_close(z, torch.ones_like(z))
            with self.assertRaisesRegex(ValueError, 'percentage'):
                resume(path, z, optimizer, dict(expected, percentage=2.0), None)
            with self.assertRaisesRegex(ValueError, 'invalid latents'):
                resume(path, torch.zeros(4, 2), optimizer, expected, None)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
