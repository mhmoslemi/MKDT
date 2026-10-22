import os
import math
import copy
import random
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Subset
import torchvision
import torchvision.transforms as T
from torchvision.utils import save_image
from dm_dsa import DMDSA
from eval import ROOT, reference_modules


VISUAL_SAVE_EVERY = 100  # Distillation iterations; always save the final grid too.


# ============================================================================
# 0. Reproducibility & Utilities
# ============================================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2023, 0.1994, 0.2010)


def normalize_cifar(x):
    """Normalizes a [B, C, H, W] tensor in [0, 1] using CIFAR statistics."""
    mean = torch.tensor(CIFAR_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR_STD, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


# ============================================================================
# 1. Backbones: ConvNet, VGG11, ResNet18 (With Multi-Stage Feature Extraction)
# ============================================================================
class ConvNet(nn.Module):
    """Standard 3-stage ConvNet used in Dataset Distillation benchmarks."""
    def __init__(self, width=128, depth=3):
        super().__init__()
        self.stages = nn.ModuleList()
        in_ch = 3
        for _ in range(depth):
            stage = nn.Sequential(
                nn.Conv2d(in_ch, width, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(width),
                nn.ReLU(inplace=True),
                nn.AvgPool2d(kernel_size=2, stride=2),
            )
            self.stages.append(stage)
            in_ch = width
        self.feat_dim = width
        self.multi_dim = width * depth

    def forward_features(self, x, multi_stage=False):
        feats = []
        h = x
        for stage in self.stages:
            h = stage(h)
            pooled = F.adaptive_avg_pool2d(h, 1).flatten(1)
            feats.append(F.normalize(pooled, dim=-1))
        if multi_stage:
            return F.normalize(torch.cat(feats, dim=-1), dim=-1)
        return feats[-1]

    def forward(self, x):
        return self.forward_features(x, multi_stage=False)


class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out, inplace=True)


class ResNet18CIFAR(nn.Module):
    """CIFAR-adapted ResNet-18 (3x3 stem, no maxpool) with multi-stage hooks."""
    def __init__(self):
        super().__init__()
        self.in_planes = 64
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.feat_dim = 512
        self.multi_dim = 64 + 128 + 256 + 512

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward_features(self, x, multi_stage=False):
        h = self.stem(x)
        f1 = self.layer1(h)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        if multi_stage:
            p1 = F.normalize(F.adaptive_avg_pool2d(f1, 1).flatten(1), dim=-1)
            p2 = F.normalize(F.adaptive_avg_pool2d(f2, 1).flatten(1), dim=-1)
            p3 = F.normalize(F.adaptive_avg_pool2d(f3, 1).flatten(1), dim=-1)
            p4 = F.normalize(F.adaptive_avg_pool2d(f4, 1).flatten(1), dim=-1)
            return F.normalize(torch.cat([p1, p2, p3, p4], dim=-1), dim=-1)
        return F.normalize(F.adaptive_avg_pool2d(f4, 1).flatten(1), dim=-1)

    def forward(self, x):
        return self.forward_features(x, multi_stage=False)


class VGG11CIFAR(nn.Module):
    """CIFAR-adapted VGG11 with BatchNorm and multi-stage feature extraction."""
    def __init__(self):
        super().__init__()
        cfg = [
            [64, "M"],
            [128, "M"],
            [256, 256, "M"],
            [512, 512, "M"],
            [512, 512, "M"],
        ]
        self.stages = nn.ModuleList()
        in_ch = 3
        for stage_cfg in cfg:
            layers = []
            for v in stage_cfg:
                if v == "M":
                    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
                else:
                    layers.extend([
                        nn.Conv2d(in_ch, v, kernel_size=3, padding=1, bias=False),
                        nn.BatchNorm2d(v),
                        nn.ReLU(inplace=True),
                    ])
                    in_ch = v
            self.stages.append(nn.Sequential(*layers))
        self.feat_dim = 512
        self.multi_dim = 64 + 128 + 256 + 512 + 512

    def forward_features(self, x, multi_stage=False):
        feats = []
        h = x
        for stage in self.stages:
            h = stage(h)
            pooled = F.adaptive_avg_pool2d(h, 1).flatten(1)
            feats.append(F.normalize(pooled, dim=-1))
        if multi_stage:
            return F.normalize(torch.cat(feats, dim=-1), dim=-1)
        return feats[-1]

    def forward(self, x):
        return self.forward_features(x, multi_stage=False)


class BenchmarkResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder, self.feat_dim = reference_modules()['common'].build_encoder('ResNet18', 32)
        self.multi_dim = 64 + 128 + 256 + 512

    def forward_features(self, images, multi_stage=False):
        features = []
        images = F.relu(self.encoder.bn1(self.encoder.conv1(images)))
        for name in ('layer1', 'layer2', 'layer3', 'layer4'):
            images = getattr(self.encoder, name)(images)
            features.append(F.normalize(F.adaptive_avg_pool2d(images, 1).flatten(1), dim=-1))
        return F.normalize(torch.cat(features, dim=-1), dim=-1) if multi_stage else features[-1]

    def forward(self, images):
        return self.forward_features(images)


def build_backbone(arch="convnet"):
    arch = arch.lower()
    if arch == "convnet":
        return ConvNet(width=128, depth=3)
    elif arch == "resnet18":
        return BenchmarkResNet()
    elif arch == "vgg11":
        return VGG11CIFAR()
    else:
        raise ValueError(f"Unknown architecture: {arch}")


# ============================================================================
# 2. Differentiable Disjoint-Crop GPU Augmentation (Operates on [0, 1] Images)
# ============================================================================
class DiffSSLAugment(nn.Module):
    """
    Pure-PyTorch differentiable SSL augmentation on GPU for [0, 1] images.
    Crucially enforces spatially distinct crops for view u and view v when
    `disjoint_crops=True`, forcing each synthetic image s_k to encode distinct
    semantic parts in different quadrants rather than a uniform stationary ripple.
    """
    def __init__(self, img_size=32, scale=(0.45, 0.85), disjoint_crops=True):
        super().__init__()
        self.img_size = img_size
        self.scale = scale
        self.disjoint_crops = disjoint_crops

    def _sample_crop_and_resize(self, x, quadrant=None):
        B, C, H, W = x.shape
        device = x.device
        out_list = []

        for i in range(B):
            s = random.uniform(self.scale[0], self.scale[1])
            crop_h = max(4, int(round(H * math.sqrt(s))))
            crop_w = max(4, int(round(W * math.sqrt(s))))

            if quadrant == 0:
                # Bias first crop toward top/left regions
                top = random.randint(0, max(0, (H - crop_h) // 2))
                left = random.randint(0, max(0, (W - crop_w) // 2))
            elif quadrant == 1:
                # Bias second crop toward bottom/right regions
                min_t = (H - crop_h) // 2
                min_l = (W - crop_w) // 2
                top = random.randint(min_t, max(min_t, H - crop_h))
                left = random.randint(min_l, max(min_l, W - crop_w))
            else:
                top = random.randint(0, max(0, H - crop_h))
                left = random.randint(0, max(0, W - crop_w))

            patch = x[i : i + 1, :, top : top + crop_h, left : left + crop_w]
            if random.random() < 0.5:
                patch = torch.flip(patch, dims=[3])
            patch = F.interpolate(
                patch, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False
            )
            out_list.append(patch)

        return torch.cat(out_list, dim=0)

    def _color_jitter(self, x):
        B, C, H, W = x.shape
        device = x.device
        if random.random() < 0.8:
            # Differentiable brightness
            b_factor = 1.0 + (torch.rand(B, 1, 1, 1, device=device) - 0.5) * 0.6
            x = x * b_factor
            # Differentiable contrast
            mean = x.mean(dim=(2, 3), keepdim=True)
            c_factor = 1.0 + (torch.rand(B, 1, 1, 1, device=device) - 0.5) * 0.6
            x = (x - mean) * c_factor + mean
            # Differentiable saturation
            gray = x.mean(dim=1, keepdim=True)
            s_factor = 1.0 + (torch.rand(B, 1, 1, 1, device=device) - 0.5) * 0.6
            x = (x - gray) * s_factor + gray

        if random.random() < 0.2:
            gray = (0.2989 * x[:, 0:1] + 0.5870 * x[:, 1:2] + 0.1140 * x[:, 2:3]).repeat(1, 3, 1, 1)
            x = gray

        return torch.clamp(x, 0.0, 1.0)

    def forward(self, x_01):
        if self.disjoint_crops:
            q1, q2 = (0, 1) if random.random() < 0.5 else (1, 0)
            u = self._sample_crop_and_resize(x_01, quadrant=q1)
            v = self._sample_crop_and_resize(x_01, quadrant=q2)
        else:
            u = self._sample_crop_and_resize(x_01, quadrant=None)
            v = self._sample_crop_and_resize(x_01, quadrant=None)

        u = normalize_cifar(self._color_jitter(u))
        v = normalize_cifar(self._color_jitter(v))
        return u, v


class DSAViews(nn.Module):
    def __init__(self, strategy):
        super().__init__()
        self.augmentation = DMDSA(strategy, [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])

    def forward(self, images, seeds=None):
        seeds = DMDSA.seeds(2) if seeds is None else seeds
        return tuple(normalize_cifar(self.augmentation(images, seed)) for seed in seeds)


# ============================================================================
# 3. Section 4.1: Structural Initialization via Untrained Priors (K-Means++)
# ============================================================================
@torch.no_grad()
def initialize_synthetic_via_untrained_priors(real_images_01, num_syn, arch="convnet", device="cuda"):
    """
    Extracts multi-stage features from an ensemble of 3 untrained random encoders
    and runs K-Means++ + Lloyd iterations to pick M real medoid images from T.
    Completely label-free!
    """
    print(f"[Init] Extracting untrained prior features on { len(real_images_01) } unlabeled images...")
    models = [build_backbone(arch).to(device).eval() for _ in range(3)]
    feats_all = []

    batch_size = 512
    for i in range(0, len(real_images_01), batch_size):
        batch = normalize_cifar(real_images_01[i : i + batch_size].to(device))
        f_batch = torch.cat([m.forward_features(batch, multi_stage=True) for m in models], dim=-1)
        feats_all.append(F.normalize(f_batch, dim=-1).cpu())

    feats = torch.cat(feats_all, dim=0).to(device)  # [N, D]
    N, D = feats.shape

    # Fast GPU K-Means++ initialization
    centroids = torch.empty((num_syn, D), device=device)
    first_idx = random.randint(0, N - 1)
    centroids[0] = feats[first_idx]
    min_dists = torch.cdist(feats, centroids[0:1]).squeeze(1).pow(2)

    for k in range(1, num_syn):
        probs = min_dists / (min_dists.sum() + 1e-8)
        next_idx = torch.multinomial(probs, 1).item()
        centroids[k] = feats[next_idx]
        new_dists = torch.cdist(feats, centroids[k : k + 1]).squeeze(1).pow(2)
        min_dists = torch.minimum(min_dists, new_dists)

    # 15 Lloyd refinement iterations
    for _ in range(15):
        dists = torch.cdist(feats, centroids)  # [N, M]
        assign = dists.argmin(dim=1)
        for k in range(num_syn):
            mask = assign == k
            if mask.any():
                centroids[k] = F.normalize(feats[mask].mean(dim=0), dim=-1)

    # Select closest real image (medoid) to each centroid
    dists = torch.cdist(feats, centroids)
    medoid_indices = dists.argmin(dim=0).cpu()
    init_syn = real_images_01[medoid_indices].clone()
    print(f"[Init] Selected {num_syn} diverse structural medoids via untrained priors.")
    return init_syn, medoid_indices


# ============================================================================
# 4. Method 3 Core Components: Student Wrapper, Online SSL, Bipartite View MI
# ============================================================================
class StudentModel(nn.Module):
    """Wraps a backbone with an SSL MLP projector for online bootstrapping on S."""
    def __init__(self, arch="convnet", proj_dim=128):
        super().__init__()
        self.backbone = build_backbone(arch)
        feat_dim = self.backbone.feat_dim
        self.projector = nn.Sequential(
            nn.Linear(feat_dim, feat_dim, bias=False),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, proj_dim, bias=True),
        )

    def forward_multi_features(self, x):
        return self.backbone.forward_features(x, multi_stage=True)

    def forward_ssl(self, x):
        h = self.backbone.forward_features(x, multi_stage=False)
        z = F.normalize(self.projector(h), dim=-1)
        return h, z


def simclr_nt_xent_loss(z1, z2, temperature=0.2):
    """Contrastive NT-Xent loss used to update students on S and for evaluation."""
    B = z1.size(0)
    z = torch.cat([z1, z2], dim=0)  # [2B, D]
    sim = torch.mm(z, z.t()) / temperature
    mask = torch.eye(2 * B, device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(mask, -1e9)
    targets = torch.cat([
        torch.arange(B, 2 * B, device=z.device),
        torch.arange(0, B, device=z.device),
    ], dim=0)
    return F.cross_entropy(sim, targets)


def barlow_twins_loss(z1, z2, lambd=5e-3):
    """Barlow Twins cross-correlation loss for evaluation pre-training."""
    B, D = z1.shape
    z1_norm = (z1 - z1.mean(dim=0)) / (z1.std(dim=0) + 1e-5)
    z2_norm = (z2 - z2.mean(dim=0)) / (z2.std(dim=0) + 1e-5)
    c = torch.mm(z1_norm.t(), z2_norm) / float(B)  # [D, D]
    on_diag = torch.diagonal(c).add_(-1.0).pow_(2).sum()
    off_diag = (c.pow(2).sum() - torch.diagonal(c).pow(2).sum())
    return on_diag + lambd * off_diag


class BipartiteViewMILoss(nn.Module):
    """
    Core Method 3 Objective:
      1. Bipartite View-Pair Mutual Information: matches real view u -> syn view u_s
         and real view v -> syn view v_s, maximizing I(C_u; C_v).
      2. Intra-Pair View-Shift Matching: forces synthetic crop discrepancy
         (phi(u_s) - phi(v_s)) to match real crop discrepancy, eliminating the
         uniform texture shortcut in Fig. 1.
      3. Dual-Subspace Covariance Alignment: aligns both the invariant view-sum
         subspace (Sigma+) and view-difference subspace (Sigma-).
    """
    def __init__(self, tau=0.08, lambda_shift=1.5, lambda_cov=0.5, eps=1e-8):
        super().__init__()
        self.tau = tau
        self.lambda_shift = lambda_shift
        self.lambda_cov = lambda_cov
        self.eps = eps

    def forward(self, phi_u_real, phi_v_real, phi_u_syn, phi_v_syn):
        B, D = phi_u_real.shape
        M = phi_u_syn.shape[0]

        # 1. Bipartite Soft Assignments: u -> u_syn, v -> v_syn
        logits_u = torch.mm(phi_u_real, phi_u_syn.t()) / self.tau  # [B, M]
        logits_v = torch.mm(phi_v_real, phi_v_syn.t()) / self.tau  # [B, M]

        p_u = F.softmax(logits_u, dim=-1)  # [B, M]
        p_v = F.softmax(logits_v, dim=-1)  # [B, M]

        # Symmetrized empirical joint assignment matrix P_bar in R^{M x M}
        P = torch.mm(p_u.t(), p_v) / float(B)
        P_bar = 0.5 * (P + P.t())
        P_bar = torch.clamp(P_bar, min=self.eps)
        P_bar = P_bar / P_bar.sum()

        P_i = P_bar.sum(dim=1, keepdim=True)  # [M, 1]
        P_j = P_bar.sum(dim=0, keepdim=True)  # [1, M]

        # Negative Mutual Information: -I(C_u; C_v)
        mi = (P_bar * (torch.log(P_bar) - torch.log(P_i) - torch.log(P_j))).sum()
        loss_mi = -mi

        # 2. Intra-Pair View-Shift Vector & Norm Alignment
        resp = (p_u * p_v).detach()  # [B, M]
        resp_norm = resp / (resp.sum(dim=0, keepdim=True) + self.eps)

        delta_real = phi_u_real - phi_v_real  # [B, D]
        delta_syn = phi_u_syn - phi_v_syn     # [M, D]

        target_delta = torch.mm(resp_norm.t(), delta_real)  # [M, D]
        loss_shift_vec = F.mse_loss(delta_syn, target_delta) * D

        real_mag = delta_real.norm(dim=-1, keepdim=True)  # [B, 1]
        syn_mag = delta_syn.norm(dim=-1, keepdim=True)    # [M, 1]
        target_mag = torch.mm(resp_norm.t(), real_mag)
        loss_shift_mag = F.mse_loss(syn_mag, target_mag)

        loss_shift = loss_shift_vec + loss_shift_mag

        # 3. Dual-Subspace Alignment (Invariance z+ and Augmentation Variance z-)
        z_plus_r = (phi_u_real + phi_v_real) * 0.70710678
        z_minus_r = (phi_u_real - phi_v_real) * 0.70710678
        z_plus_s = (phi_u_syn + phi_v_syn) * 0.70710678
        z_minus_s = (phi_u_syn - phi_v_syn) * 0.70710678

        cov_plus_r = (z_plus_r.t() @ z_plus_r) / float(B)
        cov_minus_r = (z_minus_r.t() @ z_minus_r) / float(B)
        cov_plus_s = (z_plus_s.t() @ z_plus_s) / float(M)
        cov_minus_s = (z_minus_s.t() @ z_minus_s) / float(M)

        loss_cov = (
            F.mse_loss(cov_plus_s, cov_plus_r) * D
            + F.mse_loss(cov_minus_s, cov_minus_r) * D
        )

        total_loss = loss_mi + self.lambda_shift * loss_shift + self.lambda_cov * loss_cov
        stats = {
            "mi": mi.item(),
            "l_mi": loss_mi.item(),
            "l_shift": loss_shift.item(),
            "l_cov": loss_cov.item(),
        }
        return total_loss, stats


class SelfBootstrappingStudentPool:
    """
    Maintains a pool of K student encoders with staggered lifespans that
    ONLY ever train on the synthetic set S (100% teacher-free on T).
    """
    def __init__(self, arch="convnet", pool_size=3, max_age=150, lr_student=1e-3, device="cuda"):
        self.arch = arch
        self.pool_size = pool_size
        self.max_age = max_age
        self.lr_student = lr_student
        self.device = device

        self.models = []
        self.optimizers = []
        self.ages = []

        for i in range(pool_size):
            m, opt = self._init_student()
            self.models.append(m)
            self.optimizers.append(opt)
            self.ages.append(int((i / pool_size) * max_age))

    def _init_student(self):
        m = StudentModel(self.arch, proj_dim=128).to(self.device)
        opt = torch.optim.AdamW(m.parameters(), lr=self.lr_student, weight_decay=1e-4)
        return m, opt

    def step_on_synthetic(self, syn_01_detached, augmenter, inner_steps=2):
        for i in range(self.pool_size):
            if self.ages[i] >= self.max_age:
                self.models[i], self.optimizers[i] = self._init_student()
                self.ages[i] = 0

            model = self.models[i]
            opt = self.optimizers[i]
            model.train()

            for _ in range(inner_steps):
                u_s, v_s = augmenter(syn_01_detached)
                _, z1 = model.forward_ssl(u_s)
                _, z2 = model.forward_ssl(v_s)
                loss = simclr_nt_xent_loss(z1, z2, temperature=0.2)
                opt.zero_grad()
                loss.backward()
                opt.step()

            self.ages[i] += 1


def total_variation_loss(x):
    dh = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).mean()
    dw = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).mean()
    return dh + dw


# ============================================================================
# 5. Method 3 Distillation Loop
# ============================================================================
def distill_method3(
    real_images_01,
    init_syn_01,
    arch="convnet",
    num_steps=1000,
    batch_size_real=512,
    lr_pixels=0.02,
    tau=0.08,
    lambda_shift=1.5,
    lambda_cov=0.5,
    lambda_tv=2e-3,
    lambda_anchor=0.05,
    device="cuda",
    out_dir=None,
    dataset="cifar10",
    percentage=0.2,
    dsa=True,
    dsa_strategy='color_crop_cutout_flip_scale_rotate',
):
    """
    Optimizes synthetic images S in [0, 1] space using Self-Bootstrapping
    Bipartite View-Pair Mutual Information.
    """
    init_anchor = init_syn_01.clone().detach().to(device)
    syn_01 = nn.Parameter(init_syn_01.clone().detach().to(device))

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        visual_model = {"convnet": "ConvNet", "vgg11": "VGG11", "resnet18": "ResNet18"}[arch.lower()]

    opt_pixels = torch.optim.Adam([syn_01], lr=lr_pixels)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt_pixels, T_max=num_steps, eta_min=lr_pixels)

    # Disjoint crops on S ensure different parts of s_k represent distinct views
    aug_syn = DiffSSLAugment(img_size=32, scale=(0.45, 0.85), disjoint_crops=True).to(device)
    aug_real = DiffSSLAugment(img_size=32, scale=(0.45, 0.85), disjoint_crops=False).to(device)
    if dsa:
        aug_syn = DSAViews(dsa_strategy).to(device)
        aug_real = DSAViews(dsa_strategy).to(device)

    pool = SelfBootstrappingStudentPool(
        arch=arch, pool_size=3, max_age=150, lr_student=1e-3, device=device
    )
    criterion = BipartiteViewMILoss(
        tau=tau, lambda_shift=lambda_shift, lambda_cov=lambda_cov
    ).to(device)

    N = len(real_images_01)

    for step in range(1, num_steps + 1):
        # Sample random unlabeled mini-batch from T
        idx = torch.randint(0, N, (batch_size_real,))
        x_real_01 = real_images_01[idx].to(device, non_blocking=True)

        # 1. Self-bootstrap student encoders strictly on detached S
        pool.step_on_synthetic(syn_01.detach().clamp(0.0, 1.0), aug_syn, inner_steps=2)

        # 2. Sample augmented views for real batch X and synthetic set S
        syn_clamped = torch.clamp(syn_01, 0.0, 1.0)
        if dsa:
            seeds = DMDSA.seeds(2)
            with torch.no_grad():
                u_real, v_real = aug_real(x_real_01, seeds)
            u_syn, v_syn = aug_syn(syn_clamped, seeds)
        else:
            with torch.no_grad():
                u_real, v_real = aug_real(x_real_01)
            u_syn, v_syn = aug_syn(syn_clamped)

        # 3. Compute Bipartite MI + View-Shift + Dual-Subspace loss across student pool
        opt_pixels.zero_grad()
        loss_total = 0.0
        log_stats = {"mi": 0.0, "l_mi": 0.0, "l_shift": 0.0, "l_cov": 0.0}

        for model in pool.models:
            model.eval()
            with torch.no_grad():
                phi_u_r = model.forward_multi_features(u_real)
                phi_v_r = model.forward_multi_features(v_real)

            phi_u_s = model.forward_multi_features(u_syn)
            phi_v_s = model.forward_multi_features(v_syn)

            l_k, s_k = criterion(phi_u_r, phi_v_r, phi_u_s, phi_v_s)
            loss_total = loss_total + l_k / len(pool.models)
            for k in log_stats:
                log_stats[k] += s_k[k] / len(pool.models)

        # Spatial TV + weak structural anchor to preserve natural image geometry
        loss_tv = total_variation_loss(syn_01)
        loss_anchor = F.mse_loss(syn_01, init_anchor)
        loss_total = loss_total + lambda_tv * loss_tv + lambda_anchor * loss_anchor

        loss_total.backward()
        opt_pixels.step()
        scheduler.step()

        with torch.no_grad():
            syn_01.data.clamp_(0.0, 1.0)

        if step == 1 or step % 50 == 0 or step == num_steps:
            with torch.no_grad():
                pixel_change = (syn_01 - init_anchor).square().mean().sqrt().mul(255).item()
            print(
                f"[Distill Step {step:04d}/{num_steps}] "
                f"MI: {log_stats['mi']:.4f} | "
                f"L_MI: {log_stats['l_mi']:.4f} | "
                f"L_Shift: {log_stats['l_shift']:.4f} | "
                f"L_Cov: {log_stats['l_cov']:.4f} | RMS: {pixel_change:.2f}/255"
            )

        if out_dir is not None and (step % VISUAL_SAVE_EVERY == 0 or step == num_steps):
            grid_path = os.path.join(out_dir, "vis_IIC_%s_%s_%gpercent_iter%d_final.png" % (dataset.upper(), visual_model, percentage, step))
            visible = syn_01.detach().clamp(0, 1).cpu()  # Already in [0, 1] pixel space.
            save_image(visible, grid_path, nrow=int(np.ceil(np.sqrt(len(syn_01)))))

    return syn_01.detach().cpu()


# ============================================================================
# 6. Downstream SSL Pre-Training on S & Linear Evaluation (Table 1 Protocol)
# ============================================================================
def pretrain_ssl_on_subset(
    syn_images_01,
    arch="convnet",
    ssl_method="simclr",
    epochs=400,
    batch_size=128,
    lr=1e-3,
    device="cuda",
):
    """
    Trains a fresh target encoder from scratch on the distilled dataset S
    using either SimCLR or Barlow Twins (matching Table 1 columns).
    """
    model = StudentModel(arch=arch, proj_dim=128).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    augmenter = DiffSSLAugment(img_size=32, scale=(0.25, 1.0), disjoint_crops=False).to(device)
    dataset = TensorDataset(syn_images_01)
    loader = DataLoader(dataset, batch_size=min(batch_size, len(syn_images_01)), shuffle=True, drop_last=False)

    model.train()
    for ep in range(1, epochs + 1):
        for (x_b,) in loader:
            x_b = x_b.to(device)
            u, v = augmenter(x_b)
            _, z1 = model.forward_ssl(u)
            _, z2 = model.forward_ssl(v)

            if ssl_method.lower() == "simclr":
                loss = simclr_nt_xent_loss(z1, z2, temperature=0.2)
            elif ssl_method.lower() in ["barlow", "barlow_twins", "bt"]:
                loss = barlow_twins_loss(z1, z2, lambd=5e-3)
            else:
                raise ValueError(f"Unsupported SSL method: {ssl_method}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

    return model.backbone.eval()


def evaluate_semi_supervised(
    pretrained_backbone,
    train_images_01,
    train_labels,
    test_images_01,
    test_labels,
    labeled_fraction=0.01,
    num_classes=10,
    epochs=100,
    device="cuda",
):
    """
    Evaluates the frozen pre-trained encoder on 1% or 5% labeled data
    (matching the 1% and 5% Labeled Data rows in Tables 1-3).
    """
    N = len(train_labels)
    num_labeled = int(N * labeled_fraction)

    # Class-balanced sampling for the labeled subset
    indices = []
    per_class = max(1, num_labeled // num_classes)
    for c in range(num_classes):
        c_idx = torch.where(train_labels == c)[0]
        perm = torch.randperm(len(c_idx))[:per_class]
        indices.append(c_idx[perm])
    labeled_idx = torch.cat(indices, dim=0)

    x_tr = train_images_01[labeled_idx]
    y_tr = train_labels[labeled_idx]

    # Extract frozen representations
    pretrained_backbone.eval()
    with torch.no_grad():
        tr_feats = []
        for i in range(0, len(x_tr), 256):
            b = normalize_cifar(x_tr[i : i + 256].to(device))
            tr_feats.append(pretrained_backbone.forward_features(b, multi_stage=False))
        tr_feats = torch.cat(tr_feats, dim=0)

        te_feats = []
        for i in range(0, len(test_images_01), 512):
            b = normalize_cifar(test_images_01[i : i + 512].to(device))
            te_feats.append(pretrained_backbone.forward_features(b, multi_stage=False))
        te_feats = torch.cat(te_feats, dim=0)

    y_tr = y_tr.to(device)
    y_te = test_labels.to(device)

    classifier = nn.Linear(pretrained_backbone.feat_dim, num_classes).to(device)
    opt = torch.optim.AdamW(classifier.parameters(), lr=1e-2, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    for _ in range(epochs):
        classifier.train()
        logits = classifier(tr_feats)
        loss = F.cross_entropy(logits, y_tr)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()

    classifier.eval()
    with torch.no_grad():
        preds = classifier(te_feats).argmax(dim=-1)
        acc = (preds == y_te).float().mean().item() * 100.0
    return acc


# ============================================================================
# 7. Main Entry Point (End-to-End Execution for Table 1 & Figure 1)
# ============================================================================
def load_dataset_tensors(dataset_name="cifar10", data_root="/home/mmoslem3/scratch/data"):
    if dataset_name.lower() == "cifar10":
        ds_tr = torchvision.datasets.CIFAR10(root=data_root, train=True, download=True, transform=T.ToTensor())
        ds_te = torchvision.datasets.CIFAR10(root=data_root, train=False, download=True, transform=T.ToTensor())
        num_classes = 10
    elif dataset_name.lower() == "cifar100":
        ds_tr = torchvision.datasets.CIFAR100(root=data_root, train=True, download=True, transform=T.ToTensor())
        ds_te = torchvision.datasets.CIFAR100(root=data_root, train=False, download=True, transform=T.ToTensor())
        num_classes = 100
    else:
        raise ValueError("Use 'cifar10' or 'cifar100'")

    x_tr = torch.stack([ ds_tr[i][0] for i in range(len(ds_tr)) ], dim=0)
    y_tr = torch.tensor([ ds_tr[i][1] for i in range(len(ds_tr)) ], dtype=torch.long)
    x_te = torch.stack([ ds_te[i][0] for i in range(len(ds_te)) ], dim=0)
    y_te = torch.tensor([ ds_te[i][1] for i in range(len(ds_te)) ], dtype=torch.long)
    return x_tr, y_tr, x_te, y_te, num_classes


def main():
    parser = argparse.ArgumentParser(description="SSL-DD Method 3 Full Pipeline")
    parser.add_argument("--dataset", type=str.lower, default="cifar10", choices=["cifar10", "cifar100"])
    parser.add_argument("--percentage", type=float, default=None)
    parser.add_argument("--model", choices=['ConvNetBN', 'VGG11BN', 'ResNet18BN'], default=None)
    parser.add_argument("--subset_ratio", type=float, default=0.002, help="0.002 (0.2%), 0.01 (1%), or 0.02 (2%)")
    parser.add_argument("--arch", type=str, default="convnet", choices=["convnet", "vgg11", "resnet18"])
    parser.add_argument("--distill_steps", "--Iteration", type=int, default=8000)
    parser.add_argument("--pretrain_epochs", type=int, default=100)
    parser.add_argument("--lr_pixels", "--lr_img", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", "--save_path", type=str, default="./outputs_method3")
    parser.add_argument("--dsa", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dsa_strategy", '--dsa-strategy', default='color_crop_cutout_flip_scale_rotate')
    parser.add_argument("--num_eval", type=int, default=1)
    parser.add_argument("--data_path", default=str(ROOT.parent / 'data'))
    args = parser.parse_args()
    if args.percentage is not None:
        args.subset_ratio = args.percentage / 100
    if args.model is not None:
        args.arch = args.model.removesuffix('BN').lower()

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    x_tr, y_tr, x_te, y_te, num_classes = load_dataset_tensors(args.dataset, args.data_path)
    num_syn = int(len(x_tr) * args.subset_ratio)
    print(f"=== SSL-DD Method 3 | Dataset: {args.dataset} | Subset: {args.subset_ratio*100:.1f}% ({num_syn} imgs) | Arch: {args.arch} ===")

    # 1. Structural Initialization via Untrained Priors (Section 4.1)
    init_syn_01, _ = initialize_synthetic_via_untrained_priors(
        x_tr, num_syn=num_syn, arch=args.arch, device=device
    )

    # 2. Distill Synthetic Dataset S with Method 3
    syn_01 = distill_method3(
        real_images_01=x_tr,
        init_syn_01=init_syn_01,
        arch=args.arch,
        num_steps=args.distill_steps,
        batch_size_real=512,
        lr_pixels=args.lr_pixels,
        device=device,
        out_dir=args.out_dir,
        dataset=args.dataset,
        percentage=args.subset_ratio * 100,
        dsa=args.dsa,
        dsa_strategy=args.dsa_strategy,
    )

    # Pixels are already in [0, 1]; identity normalization preserves them in eval.py.
    torch.save({
        "data": syn_01,
        "dataset": args.dataset.upper(),
        "percentage": args.subset_ratio * 100,
        "normalization": [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
        "pixel_space": "unit_rgb",
        "method": "method3",
        "generation_arch": args.arch,
        "seed": args.seed,
        "iteration": args.distill_steps,
        "dsa": args.dsa,
        "dsa_strategy": args.dsa_strategy,
        "optimizer": "adam", "lr_img": args.lr_pixels,
    }, os.path.join(args.out_dir, f"syn_{args.dataset}_{args.subset_ratio}.pt"))
    if args.num_eval == 0:
        return

    # 3. Evaluate on both SimCLR and Barlow Twins under 1% and 5% Labeled Regimes (Table 1)
    for ssl_alg in ["simclr", "barlow_twins"]:
        print(f"\n--- Pre-training {args.arch} on Distilled S ({num_syn} imgs) using {ssl_alg.upper()} ---")
        backbone = pretrain_ssl_on_subset(
            syn_01, arch=args.arch, ssl_method=ssl_alg, epochs=args.pretrain_epochs, device=device
        )
        for labeled_frac in [0.01]:
            accs = []
            for trial in range(1):
                set_seed(args.seed + trial)
                acc = evaluate_semi_supervised(
                    backbone, x_tr, y_tr, x_te, y_te,
                    labeled_fraction=labeled_frac, num_classes=num_classes, device=device
                )
                accs.append(acc)
            print(
                f"[Result] {ssl_alg.upper()} | Labeled: {labeled_frac*100:.0f}% | "
                f"Test Acc: {np.mean(accs):.2f} +/- {np.std(accs):.2f}%"
            )










if __name__ == "__main__":
    main()
