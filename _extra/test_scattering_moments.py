"""Numerical and integration checks: python -m pytest -q test_scattering_moments.py."""

import sys

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from scattering_moments import (
    ScatteringFeatures,
    backward_scattering_moments,
    compute_real_moments,
    moment_loss_and_gradient,
)


@pytest.fixture(autouse=True)
def deterministic_small_tensors():
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(17)
    yield
    torch.set_num_threads(previous_threads)


@pytest.mark.parametrize('batch_size', [1, 4, 11])
def test_streamed_real_moments_match_full_dataset(batch_size):
    features = torch.randn(11, 7, dtype=torch.float64) + 1000
    loader = DataLoader(TensorDataset(features), batch_size=batch_size)
    mean, covariance = compute_real_moments(torch.nn.Identity(), loader, 'cpu')
    torch.testing.assert_close(mean, features.mean(dim=0))
    torch.testing.assert_close(covariance, torch.cov(features.T), rtol=1e-11, atol=1e-11)
    assert not mean.requires_grad and not covariance.requires_grad


@pytest.mark.parametrize('count', [2, 5])
@pytest.mark.parametrize('gamma', [0.0, 2.5])
@pytest.mark.parametrize('block_size', [1, 3, 20])
def test_exact_loss_and_feature_gradient(count, gamma, block_size):
    real = torch.randn(9, 7, dtype=torch.float64)
    syn = torch.randn(count, 7, dtype=torch.float64, requires_grad=True)
    real_mean, real_covariance = real.mean(dim=0), torch.cov(real.T)
    mean_loss = (syn.mean(dim=0) - real_mean).square().sum()
    covariance_loss = (torch.cov(syn.T) - real_covariance).square().sum()
    reference_loss = covariance_loss + gamma * mean_loss
    reference_gradient, = torch.autograd.grad(reference_loss, syn)

    loss, actual_mean, actual_covariance, gradient = moment_loss_and_gradient(
        syn, real_mean, real_covariance, gamma, block_size
    )
    torch.testing.assert_close(loss, reference_loss)
    torch.testing.assert_close(actual_mean, mean_loss)
    torch.testing.assert_close(actual_covariance, covariance_loss)
    torch.testing.assert_close(gradient, reference_gradient, rtol=1e-11, atol=1e-11)


def test_identical_distributions_have_zero_loss_and_gradient():
    features = torch.randn(6, 4, dtype=torch.float64)
    loss, _, _, gradient = moment_loss_and_gradient(
        features, features.mean(dim=0), torch.cov(features.T), covariance_block_size=3
    )
    torch.testing.assert_close(loss, torch.zeros_like(loss), atol=1e-28, rtol=0)
    torch.testing.assert_close(gradient, torch.zeros_like(gradient), atol=1e-14, rtol=0)


@pytest.mark.parametrize('channels', [1, 3])
def test_chunked_scattering_pixel_gradient_matches_full_autograd(channels):
    scattering = ScatteringFeatures((8, 8), J=2, L=2).double()
    assert list(scattering.parameters()) == []
    real = torch.randn(7, channels, 8, 8, dtype=torch.float64)
    real_mean, real_covariance = compute_real_moments(
        scattering, DataLoader(TensorDataset(real), batch_size=3), 'cpu'
    )
    images = torch.randn(5, channels, 8, 8, dtype=torch.float64, requires_grad=True)
    features = scattering(images)
    # 0th + 1st + 2nd order paths, each channel, and every spatial position.
    assert features.shape == (5, channels * (1 + 2 * 2 + 2 ** 2) * 2 * 2)
    reference_loss = (
        (torch.cov(features.T) - real_covariance).square().sum()
        + 1.7 * (features.mean(dim=0) - real_mean).square().sum()
    )
    reference_gradient, = torch.autograd.grad(reference_loss, images)

    loss, _, _ = backward_scattering_moments(
        images, scattering, real_mean, real_covariance,
        gamma=1.7, batch_size=2, covariance_block_size=13,
    )
    torch.testing.assert_close(loss, reference_loss)
    torch.testing.assert_close(images.grad, reference_gradient, rtol=1e-9, atol=1e-11)
    assert images.grad.abs().sum() > 0

    before = images.detach().clone()
    torch.optim.SGD([images], lr=0.001).step()
    after_features = scattering(images)
    after_loss = (
        (torch.cov(after_features.T) - real_covariance).square().sum()
        + 1.7 * (after_features.mean(dim=0) - real_mean).square().sum()
    )
    assert not torch.equal(images, before)
    assert after_loss.item() < loss.item()


@pytest.mark.parametrize('count', [0, 1])
def test_undefined_covariances_are_rejected(count):
    features = torch.zeros(count, 3)
    loader = DataLoader(TensorDataset(features), batch_size=2)
    with pytest.raises(ValueError, match='at least two'):
        compute_real_moments(torch.nn.Identity(), loader, 'cpu')
    with pytest.raises(ValueError, match='at least two'):
        moment_loss_and_gradient(features, torch.zeros(3), torch.eye(3))


@pytest.mark.parametrize('num_eval', [0, 2])
def test_main_saves_distilled_images_and_only_trains_evaluation_networks(monkeypatch, tmp_path, num_eval):
    import main_SSL_new as main

    images = torch.randn(20, 3, 8, 8)
    dataset = TensorDataset(images, torch.arange(20) % 2)
    monkeypatch.setattr(main, 'get_dataset', lambda *_: (
        3, (8, 8), 2, ['a', 'b'], [0, 0, 0], [1, 1, 1],
        dataset, dataset, DataLoader(dataset, batch_size=4),
    ))
    network_calls, evaluation_calls, moment_calls = [], [], []

    def get_network(*_):
        network_calls.append(True)
        return torch.nn.Linear(1, 1)

    def evaluate(it_eval, net, *_):
        evaluation_calls.append(it_eval)
        return net, 0.5, 0.5

    def moments(*args):
        moment_calls.append(True)
        return compute_real_moments(*args)

    monkeypatch.setattr(main, 'get_network', get_network)
    monkeypatch.setattr(main, 'evaluate_synset_SSL', evaluate)
    monkeypatch.setattr(main, 'compute_real_moments', moments)
    monkeypatch.setattr(sys, 'argv', [
        'main_SSL_new.py', '--device', 'cpu', '--data_path', str(tmp_path / 'data'),
        '--save_path', str(tmp_path), '--percentage', '25', '--Iteration', '2',
        '--num_eval', str(num_eval), '--batch_real', '7', '--batch_syn', '2',
        '--scattering_J', '2', '--scattering_L', '2', '--covariance_block_size', '13',
    ])
    np.random.seed(4)
    initial = images[np.random.permutation(len(images))[:5]].clone()
    np.random.seed(4)
    main.main()

    saved = torch.load(tmp_path / 'res_Scattering-SSL_CIFAR10_ConvNet_25percent.pt', weights_only=True)
    assert saved['data'].shape == (5, 3, 8, 8)
    assert torch.isfinite(saved['data']).all()
    assert not torch.equal(saved['data'], initial)
    assert saved['iteration'] == 2
    assert saved['scattering'] == {'J': 2, 'L': 2, 'max_order': 2}
    assert len(moment_calls) == 1
    assert len(network_calls) == 2 * num_eval  # initial and final evaluation only
    assert evaluation_calls == list(range(num_eval)) * 2
    assert (tmp_path / 'vis_Scattering_CIFAR10_ConvNet_25percent_iter2.png').is_file()
