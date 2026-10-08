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


@torch.no_grad()
def cross_view_loss_and_gradients(features_a, features_b, real_a, real_b,
                                  covariance_block_size=1024):
    """Match cross-covariance of two views with corresponding image rows.

    C_syn = A.T @ B / (M - 1), C_real = R_a.T @ R_b / (N - 1),
    where each feature matrix is centered by its own column mean.
    For E = C_syn - C_real, L_pair = ||E||_F^2 has gradients
    dL/dA = 2 * B @ E.T / (M - 1), dL/dB = 2 * A @ E / (M - 1).
    E is generally NOT symmetric: both branches must be differentiated.
    All matrix entries are included, using row blocks to bound memory.
    """
    matrices = (features_a, features_b, real_a, real_b)
    if any(matrix.ndim != 2 or len(matrix) < 2 for matrix in matrices):
        raise ValueError('Cross-view covariance requires at least two rows per feature matrix.')
    if len(features_a) != len(features_b) or len(real_a) != len(real_b):
        raise ValueError('The two views must contain the same image rows in the same order.')
    if features_a.shape[1] != real_a.shape[1] or features_b.shape[1] != real_b.shape[1]:
        raise ValueError('Real and synthetic feature dimensions must match for each view.')
    if covariance_block_size < 1:
        raise ValueError('covariance_block_size must be positive.')

    a = features_a - features_a.mean(dim=0)
    b = features_b - features_b.mean(dim=0)
    ra = real_a - real_a.mean(dim=0)
    rb = real_b - real_b.mean(dim=0)
    syn_denominator = len(a) - 1
    real_denominator = len(ra) - 1
    gradient_a = torch.empty_like(a)
    gradient_b = torch.zeros_like(b)
    loss = a.new_zeros(())
    for start in range(0, a.shape[1], covariance_block_size):
        end = min(start + covariance_block_size, a.shape[1])
        error = a[:, start:end].T @ b
        error.div_(syn_denominator)
        error.addmm_(ra[:, start:end].T, rb, alpha=-1.0 / real_denominator)
        loss.add_(error.square().sum())
        gradient_a[:, start:end] = (b @ error.T) * (2.0 / syn_denominator)
        gradient_b.addmm_(a[:, start:end], error, alpha=2.0 / syn_denominator)

    gradient_a.sub_(gradient_a.mean(dim=0, keepdim=True))
    gradient_b.sub_(gradient_b.mean(dim=0, keepdim=True))
    return loss, gradient_a, gradient_b


def _backward_feature_gradient(images, feature_extractor, feature_gradient,
                               batch_size, accumulate):
    """Replay one view in chunks and propagate its feature gradient to pixels."""
    if not torch.isfinite(feature_gradient).all().item():
        raise FloatingPointError('Non-finite scattering feature gradient.')
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


def backward_scattering_views(images, feature_extractors, real_features,
                               gamma=1.0, pair_weight=1.0, batch_size=32,
                               covariance_block_size=1024):
    """Accumulate marginal and paired-view losses into existing pixel gradients.

    L_aug = sum_v L_moments(v) + pair_weight * sum_{a<b} L_pair(a, b).
    Real view matrices share one row order; synthetic views share another.
    Every view is replayed unchanged when propagating its combined gradient.
    The returned pair loss is the unweighted sum over unordered view pairs.
    """
    if batch_size < 1 or len(images) < 2:
        raise ValueError('Use a positive batch size and at least two synthetic images.')
    if not feature_extractors or len(feature_extractors) != len(real_features):
        raise ValueError('Each augmentation view needs a corresponding real feature matrix.')
    if not math.isfinite(pair_weight) or pair_weight < 0:
        raise ValueError('pair_weight must be finite and nonnegative.')
    if pair_weight > 0 and len(feature_extractors) < 2:
        raise ValueError('A positive pair_weight requires at least two augmentation views.')

    with torch.no_grad():
        synthetic_features = [
            torch.cat([extractor(batch) for batch in images.split(batch_size)], dim=0)
            for extractor in feature_extractors
        ]
    view_losses = []
    feature_gradients = []
    mean_loss = images.new_zeros(())
    covariance_loss = images.new_zeros(())
    pair_loss = images.new_zeros(())
    for features, real in zip(synthetic_features, real_features):
        loss, mean, covariance, gradient = moment_loss_and_gradient(
            features, real.mean(dim=0), None, gamma, covariance_block_size,
            real_features=real,
        )
        view_losses.append(loss)
        feature_gradients.append(gradient)
        mean_loss = mean_loss + mean
        covariance_loss = covariance_loss + covariance

    if pair_weight > 0:
        for a in range(len(feature_extractors)):
            for b in range(a + 1, len(feature_extractors)):
                loss, gradient_a, gradient_b = cross_view_loss_and_gradients(
                    synthetic_features[a], synthetic_features[b],
                    real_features[a], real_features[b], covariance_block_size,
                )
                pair_loss = pair_loss + loss
                feature_gradients[a].add_(gradient_a, alpha=pair_weight)
                feature_gradients[b].add_(gradient_b, alpha=pair_weight)
                del gradient_a, gradient_b
    del synthetic_features
    loss = covariance_loss + gamma * mean_loss + pair_weight * pair_loss
    if not torch.isfinite(loss).item():
        raise FloatingPointError('Non-finite augmented scattering loss.')
    for extractor, gradient in zip(feature_extractors, feature_gradients):
        _backward_feature_gradient(images, extractor, gradient, batch_size, accumulate=True)
    return loss, mean_loss, covariance_loss, pair_loss, view_losses


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

    _backward_feature_gradient(images, feature_extractor, feature_gradient, batch_size, accumulate)
    return loss, mean_loss, covariance_loss
