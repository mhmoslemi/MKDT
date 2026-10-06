"""Fixed order-2 scattering and exact, dataset-wide moment matching.

Install the additional dependency with: python -m pip install kymatio==0.3.0
The full flattened coefficients are retained, including channels and spatial
positions. Storing the real covariance therefore requires O(feature_dim ** 2)
memory; batching does not approximate or discard covariance entries.
"""

import math

import torch
from torch import nn


class ScatteringFeatures(nn.Module):
    """Parameter-free Morlet scattering, concatenated through order two."""

    def __init__(self, shape, J=3, L=8):
        super().__init__()
        if J < 2 or 2 ** J > min(shape):
            raise ValueError('Scattering requires 2 <= J <= log2(min(image shape)).')
        if L < 1:
            raise ValueError('Scattering L must be positive.')
        try:
            from kymatio import Scattering2D
        except ImportError as exc:
            raise ImportError(
                'Scattering distillation requires Kymatio. Install it with '
                '`python -m pip install kymatio==0.3.0`.'
            ) from exc
        # Select the public 2D frontend directly, avoiding unrelated 3D imports.
        self.scattering = Scattering2D(J=J, shape=shape, L=L, max_order=2, frontend='torch')

    def forward(self, images):
        # Kymatio applies the same fixed filters independently to each channel.
        return self.scattering(images.contiguous()).flatten(start_dim=1)


@torch.no_grad()
def compute_real_moments(feature_extractor, loader, device):
    """Stream the full real dataset once using a stable batch-Welford merge."""
    count = 0
    mean = None
    m2 = None
    for batch in loader:
        images = batch[0] if isinstance(batch, (tuple, list)) else batch
        features = feature_extractor(images.to(device))
        batch_count = len(features)
        if batch_count == 0:
            continue
        batch_mean = features.mean(dim=0)
        centered = features - batch_mean
        if count == 0:
            mean = batch_mean.clone()
            m2 = centered.T @ centered
        else:
            delta = batch_mean - mean
            total = count + batch_count
            m2.addmm_(centered.T, centered)
            m2.addr_(delta, delta, alpha=count * batch_count / total)
            mean.add_(delta, alpha=batch_count / total)
        count += batch_count
    if count < 2:
        raise ValueError('Real covariance requires at least two images.')
    return mean, m2.div_(count - 1)


@torch.no_grad()
def moment_loss_and_gradient(features, real_mean, real_covariance, gamma=1.0,
                             covariance_block_size=1024, real_features=None):
    """Return the exact loss, its components, and d(loss)/d(features).

    L = ||C_syn - C_real||_F^2 + gamma * ||mu_syn - mu_real||_2^2,
    C_syn = Z.T @ Z / (M - 1), Z = features - mu_syn.

    The covariance gradient is 4 * Z @ (C_syn - C_real) / (M - 1).
    Row blocks bound temporary memory without changing the full objective.
    For fresh augmented views, real_features supplies the real batch instead
    of a cached covariance. Its covariance is computed in the same row blocks,
    avoiding another full feature_dim-by-feature_dim allocation per view.
    """
    if features.ndim != 2 or len(features) < 2:
        raise ValueError('Synthetic covariance requires at least two feature vectors.')
    if not math.isfinite(gamma) or gamma < 0:
        raise ValueError('gamma must be finite and nonnegative.')
    if covariance_block_size < 1:
        raise ValueError('covariance_block_size must be positive.')
    count, feature_dim = features.shape
    if real_mean.shape != (feature_dim,):
        raise ValueError('Real and synthetic feature dimensions must match.')
    real_centered = None
    if real_features is not None:
        if real_covariance is not None:
            raise ValueError('Provide either real_features or real_covariance, not both.')
        if real_features.ndim != 2 or len(real_features) < 2 or real_features.shape[1] != feature_dim:
            raise ValueError('Real features need at least two rows and matching feature dimensions.')
        real_mean = real_features.mean(dim=0)
        real_centered = real_features - real_mean
    elif real_covariance is None or real_covariance.shape != (feature_dim, feature_dim):
        raise ValueError('Real and synthetic covariance dimensions must match.')

    mean = features.mean(dim=0)
    centered = features - mean
    mean_error = mean - real_mean
    mean_loss = mean_error.square().sum()
    covariance_loss = features.new_zeros(())
    feature_gradient = torch.empty_like(features)
    for start in range(0, feature_dim, covariance_block_size):
        end = min(start + covariance_block_size, feature_dim)
        error = centered[:, start:end].T @ centered
        error.div_(count - 1)
        if real_centered is None:
            error.sub_(real_covariance[start:end])
        else:
            error.addmm_(real_centered[:, start:end].T, real_centered,
                         alpha=-1.0 / (len(real_centered) - 1))
        covariance_loss.add_(error.square().sum())
        # Covariance matrices are symmetric; these rows give gradient columns.
        feature_gradient[:, start:end] = (centered @ error.T) * (4.0 / (count - 1))

    # Differentiate centering explicitly (the correction is zero in exact math).
    feature_gradient.sub_(feature_gradient.mean(dim=0, keepdim=True))
    feature_gradient.add_(mean_error.unsqueeze(0), alpha=2.0 * gamma / count)
    loss = covariance_loss + gamma * mean_loss
    return loss, mean_loss, covariance_loss, feature_gradient


def backward_scattering_moments(images, feature_extractor, real_mean,
                                real_covariance, gamma=1.0, batch_size=32,
                                covariance_block_size=1024, real_features=None,
                                accumulate=False):
    """Set image gradients for the full synthetic set with bounded activations.

    First calculate global synthetic moments without a scattering graph. Then
    recompute one image batch at a time and apply the exact feature gradient by
    the chain rule. No pixels change until every batch gradient is available.
    An augmented feature_extractor must replay the same sampled transform on
    both passes. accumulate=True adds this view's pixel gradient to earlier
    views so the caller can take one optimizer step for their summed loss.
    """
    if batch_size < 1:
        raise ValueError('batch_size must be positive.')
    if len(images) < 2:
        raise ValueError('Synthetic covariance requires at least two images.')
    with torch.no_grad():
        features = torch.cat([
            feature_extractor(batch) for batch in images.split(batch_size)
        ], dim=0)
    loss, mean_loss, covariance_loss, feature_gradient = moment_loss_and_gradient(
        features, real_mean, real_covariance, gamma, covariance_block_size,
        real_features=real_features,
    )
    del features
    if not torch.isfinite(loss).item() or not torch.isfinite(feature_gradient).all().item():
        raise FloatingPointError('Non-finite scattering moment loss or feature gradient.')

    image_gradient = torch.empty_like(images)
    for start in range(0, len(images), batch_size):
        end = min(start + batch_size, len(images))
        batch = images[start:end].detach().requires_grad_(True)
        output = feature_extractor(batch)
        image_gradient[start:end] = torch.autograd.grad(
            output, batch, grad_outputs=feature_gradient[start:end]
        )[0]
    if not torch.isfinite(image_gradient).all().item():
        raise FloatingPointError('Non-finite synthetic pixel gradient.')
    if accumulate and images.grad is not None:
        images.grad.add_(image_gradient)
    else:
        images.grad = image_gradient
    return loss, mean_loss, covariance_loss
