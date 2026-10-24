"""Small input-format fixtures; run on an allocated node, without training."""
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

spec = importlib.util.spec_from_file_location("manual_eval", Path(__file__).with_name("eval.py"))
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


class EvalInputTests(unittest.TestCase):
    def check_input(self, saved, expected_images, expected_normalization):
        mean, std = [0.5071, 0.4865, 0.4409], [0.2009, 0.1984, 0.2023]
        reference = {"data": SimpleNamespace(SPECS={"CIFAR100": (100, mean, std)}),
                     "common": SimpleNamespace(atomic_json=lambda *args: None)}
        with tempfile.TemporaryDirectory() as temporary:
            arguments = ["eval.py", "--dataset", "CIFAR100", "--percentage", "1",
                         "--synthetic-checkpoint", str(Path(temporary) / "images.pt"),
                         "--save-path", str(Path(temporary) / "evaluation")]
            with patch.object(evaluation.sys, "argv", arguments), \
                    patch.object(evaluation.torch, "load", return_value=saved), \
                    patch.object(evaluation.torch, "set_num_threads"), \
                    patch.object(evaluation, "reference_modules", return_value=reference), \
                    patch.object(evaluation, "evaluate_synset_SSL", return_value=(None, {}, 0.4)) as probe:
                evaluation.main()
            self.assertIs(probe.call_args.args[2], expected_images)
            self.assertEqual(probe.call_args.args[5], expected_normalization)

    def test_mkdt_tensor_uses_root_generation_normalization(self):
        images = torch.zeros(2, 3, 32, 32)
        self.check_input(images, images, [[0.5071, 0.4865, 0.4409], [0.2009, 0.1984, 0.2023]])

    def test_existing_dictionary_keeps_images_and_normalization(self):
        images = torch.ones(2, 3, 32, 32)
        normalization = [[0.5, 0.5, 0.5], [0.25, 0.25, 0.25]]
        self.check_input({"data": images, "dataset": "CIFAR100", "percentage": 1,
                          "normalization": normalization}, images, normalization)


if __name__ == "__main__":
    unittest.main()
