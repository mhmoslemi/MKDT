"""Checks for the random-feature cross-covariance formulation."""

import sys

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

import main_idea


@pytest.fixture(autouse=True)
def deterministic_small_tensors():
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(23)
    yield
    torch.set_num_threads(previous_threads)


def identity_augment(images, _seed):
    return images


def test_cross_covariance_is_the_uncentered_statistic_from_the_equation():
    left = torch.tensor([[1.0, 2.0], [3.0, 5.0], [7.0, 11.0]])
    right = torch.tensor([[13.0, 17.0], [19.0, 23.0], [29.0, 31.0]])

    actual = main_idea.cross_covariance(left, right)

    torch.testing.assert_close(actual, sum(
        torch.outer(a, b) for a, b in zip(left, right)
    ) / len(left))
    # A centered covariance would differ for these deliberately nonzero means.
    assert not torch.allclose(actual, (left - left.mean(0)).T @ (right - right.mean(0)) / len(left))


@pytest.mark.parametrize('block_size', [1, 3, 20])
def test_blocked_loss_and_gradient_match_direct_frobenius_objective(block_size):
    synthetic_a = torch.randn(5, 7, dtype=torch.float64, requires_grad=True)
    synthetic_b = torch.randn(5, 7, dtype=torch.float64, requires_grad=True)
    second_a = torch.randn(5, 7, dtype=torch.float64, requires_grad=True)
    second_b = torch.randn(5, 7, dtype=torch.float64, requires_grad=True)
    real = torch.randn(7, 7, dtype=torch.float64)
    pairs = [(synthetic_a, synthetic_b), (second_a, second_b)]

    direct_statistic = sum(main_idea.cross_covariance(a, b) for a, b in pairs) / len(pairs)
    direct_loss = (direct_statistic - real).square().sum()
    direct_gradients = torch.autograd.grad(
        direct_loss, (synthetic_a, synthetic_b, second_a, second_b), retain_graph=True
    )
    blocked_loss = main_idea.cross_covariance_matching_loss(pairs, real, block_size)
    blocked_gradients = torch.autograd.grad(
        blocked_loss, (synthetic_a, synthetic_b, second_a, second_b)
    )

    torch.testing.assert_close(blocked_loss, direct_loss, rtol=1e-12, atol=1e-12)
    for actual, expected in zip(blocked_gradients, direct_gradients):
        torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-11)


@pytest.mark.parametrize('batch_size', [1, 4, 11])
def test_streamed_real_statistic_matches_full_matrix(batch_size):
    features = torch.randn(11, 6, dtype=torch.float64) + 20
    loader = DataLoader(TensorDataset(features), batch_size=batch_size)

    actual = main_idea.estimate_real_cross_covariance(
        torch.nn.Identity(), loader, identity_augment, num_aug_pairs=3,
        rng=np.random.default_rng(4), device='cpu',
    )

    torch.testing.assert_close(
        actual, features.T @ features / len(features), rtol=1e-12, atol=1e-12
    )
    assert not actual.requires_grad


def test_synthetic_feature_extraction_preserves_pixel_gradient():
    images = torch.randn(5, 1, 2, 2, dtype=torch.float64, requires_grad=True)
    encoder = lambda batch: batch.flatten(1)
    pairs = main_idea.synthetic_feature_pairs(
        encoder, images, identity_augment, num_aug_pairs=2, batch_size=2,
        rng=np.random.default_rng(7),
    )
    real = torch.zeros(4, 4, dtype=torch.float64)

    loss = main_idea.cross_covariance_matching_loss(pairs, real, 3)
    loss.backward()

    assert images.grad is not None
    assert torch.isfinite(images.grad).all()
    assert images.grad.abs().sum() > 0


def test_main_uses_only_frozen_random_networks_and_saves_result(monkeypatch, tmp_path):
    images = torch.rand(12, 1, 4, 4)
    labels = torch.arange(12) % 2
    dataset = TensorDataset(images, labels)
    monkeypatch.setattr(main_idea, 'get_dataset', lambda *_: (
        1, (4, 4), 2, ['a', 'b'], [0.0], [1.0], dataset, dataset,
        DataLoader(dataset, batch_size=4),
    ))
    networks = []

    class TinyEncoder(torch.nn.Module):
        def __init__(self, seed):
            super().__init__()
            generator = torch.Generator().manual_seed(seed)
            self.features = torch.nn.Linear(16, 4, bias=False)
            with torch.no_grad():
                self.features.weight.copy_(torch.randn(4, 16, generator=generator))
            self.classifier = torch.nn.Linear(4, 2)

        def embed(self, batch):
            return self.features(batch.flatten(1))

        def forward(self, batch):
            return self.classifier(self.embed(batch))

    def get_network(_model, _channel, _classes, _size, seed=None):
        network = TinyEncoder(seed)
        networks.append((network, [parameter.detach().clone() for parameter in network.parameters()]))
        return network

    monkeypatch.setattr(main_idea, 'get_network', get_network)
    monkeypatch.setattr(sys, 'argv', [
        'main_idea.py', '--device', 'cpu', '--data_path', str(tmp_path / 'data'),
        '--save_path', str(tmp_path), '--percentage', '50', '--Iteration', '2',
        '--num_eval', '0', '--model', 'Tiny', '--random_models', 'Tiny',
        '--batch_real', '5', '--batch_syn', '2', '--num_random_networks', '2',
        '--num_aug_pairs', '2', '--covariance_block_size', '3',
        '--distill_aug_strategy', 'none', '--lr_img', '0.01',
        '--seed', '9',
    ])
    indices = np.random.RandomState(9).permutation(len(images))[:6]
    initial = images[indices].clone()

    main_idea.main()

    result_path = tmp_path / 'res_RFC_CIFAR10_Tiny_50percent.pt'
    saved = torch.load(result_path, weights_only=True)
    assert saved['data'].shape == initial.shape
    assert torch.isfinite(saved['data']).all()
    assert not torch.equal(saved['data'], initial)
    assert saved['iteration'] == 2
    assert len(saved['loss_history']) == 2
    assert saved['random_feature_distribution'] == {
        'models': ['Tiny'], 'samples_per_update': 2, 'trained': False,
    }
    assert saved['cross_covariance']['centered'] is False
    assert len(networks) == 4
    for network, initial_parameters in networks:
        assert not network.training
        assert all(not parameter.requires_grad for parameter in network.parameters())
        for actual, expected in zip(network.parameters(), initial_parameters):
            torch.testing.assert_close(actual, expected)
    assert not (tmp_path / 'vis_RFC_CIFAR10_Tiny_50percent_iter2.png').exists()
